"""Runtime state of the dynamic textures of a scene (see ``gs.textures.DynamicTexture``)."""

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.constants import TEXTURE_SLOT
from genesis.options.textures import ColorTexture, DynamicTexture, ImageTexture, texture_to_rgba
from genesis.utils.misc import broadcast_tensor, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidVisGeom
    from genesis.engine.scene import Scene


# The selection of an environment showing the pixels drawn on its canvas, in place of a declared image
DRAWN_IMAGE = -1


class DynamicTextureBinding(NamedTuple):
    """A visual geom drawing a dynamic texture, and the slot of its surface that the texture fills."""

    vgeom: "RigidVisGeom"
    slot: TEXTURE_SLOT


class DynamicTextureHandle:
    """The image each environment shows for one dynamic texture, and the renderers drawing it.

    Genesis holds the selection alone: the position of the declared image each environment shows, or 'DRAWN_IMAGE' for
    the pixels drawn on its canvas. The renderers hold the pixels, and each one implements the same contract:

    - at build, it allocates the declared images ('texture.get_initial_images()'), then the canvases that
      'texture.canvas' asks for (one per environment it renders, or one shared), binds them to the visual geoms in
      'bindings', and subscribes a callback with 'subscribe';
    - on every drawing, the callback receives the environments drawn and their pixels, as a uint8 RGBA tensor on the
      device of shape (len(envs_idx), height, width, 4), or (1, height, width, 4) for a canvas every environment shares;
    - at sync, it reads 'selection' when 'selection_version' moved since it last read it.
    """

    def __init__(self, scene: "Scene", texture: DynamicTexture, bindings: list[DynamicTextureBinding]):
        self._scene = scene
        self._texture = texture
        self._bindings = bindings
        (self._initial_image,) = texture.get_images_idx(texture.initial_image)
        self._selection = np.full(scene._B, self._initial_image, dtype=gs.np_int)
        self._selection_version = 0
        self._callbacks: list[Callable[[np.ndarray, torch.Tensor], None]] = []

    @property
    def name(self) -> str | None:
        """The name of the texture, by which the setters of an entity holding several dynamic textures pick it."""
        return self._texture.name

    @property
    def texture(self) -> DynamicTexture:
        """The declaration of the texture, as the surface of its visual geoms holds it."""
        return self._texture

    @property
    def bindings(self) -> list[DynamicTextureBinding]:
        """The visual geoms drawing the texture, each with the slot of its surface that the texture fills."""
        return self._bindings

    @property
    def selection(self) -> np.ndarray:
        """The image each environment shows, as a read-only array (see 'DynamicTextureHandle')."""
        selection = self._selection.view()
        selection.flags.writeable = False
        return selection

    @property
    def selection_version(self) -> int:
        """A counter incremented on every change of 'selection'."""
        return self._selection_version

    def subscribe(self, callback: Callable[[np.ndarray, torch.Tensor], None]) -> None:
        """Call 'callback' on every drawing (see 'DynamicTextureHandle').

        Genesis keeps no pixels, so a renderer subscribes before any environment shows a drawing.
        """
        if (self._selection == DRAWN_IMAGE).any():
            gs.raise_exception("A renderer cannot be added once pixels have been drawn on a dynamic texture.")
        self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[np.ndarray, torch.Tensor], None]) -> None:
        """Stop calling a callback given to 'subscribe'."""
        self._callbacks.remove(callback)

    def get_texture(self, envs_idx=None) -> torch.Tensor:
        """Return the position of the declared image each selected environment shows, 'DRAWN_IMAGE' for drawn pixels."""
        selection = self._selection[tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))]
        selection = torch.as_tensor(selection, dtype=gs.tc_int, device=gs.device)
        return selection[0] if self._scene.n_envs == 0 else selection

    def set_texture(self, image, envs_idx=None) -> None:
        """Show a declared image in the selected environments.

        'image' is a declared image given by position or by name, one for all the selected environments, or a sequence
        of them, one per selected environment.
        """
        envs_idx = tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))
        if torch.is_tensor(image):
            image = tensor_to_array(image)
        if np.ndim(image) > 1:
            gs.raise_exception("'set_texture' shows a declared image. Draw pixels with 'draw_texture' instead.")
        self._selection[envs_idx] = self._texture.get_images_idx(image)
        self._selection_version += 1

    def draw_texture(self, pixels, envs_idx=None) -> None:
        """Draw pixels on the canvas of each selected environment.

        'pixels' is a ``ColorTexture``, an ``ImageTexture``, or an RGB or RGBA array of uint8 values or of floats between
        0 and 1, of shape (height, width, channels) for all the selected environments or (len(envs_idx), height, width,
        channels) for one each. Integer and float values are clamped to the range of a channel. A texture whose canvas
        every environment shares takes a single image.
        """
        if self._texture.canvas is None:
            gs.raise_exception("Pixels are drawn on a dynamic texture declared with a 'canvas'.")
        envs_idx = tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))
        if isinstance(pixels, (ColorTexture, ImageTexture)):
            pixels = texture_to_rgba(pixels, self._texture.res)
        # Torch warns on wrapping a read-only array, such as one read from an image, which the copy below never writes
        if isinstance(pixels, np.ndarray) and not pixels.flags.writeable:
            pixels = pixels.copy()
        image = torch.as_tensor(pixels, device=gs.device)
        if image.ndim not in (3, 4):
            gs.raise_exception(
                f"Pixels are drawn as an array of shape (height, width, channels), got {image.ndim} dims."
            )
        if self._texture.canvas == "shared" and image.ndim == 4:
            gs.raise_exception("A dynamic texture whose canvas every environment shares takes a single image.")
        n_channels = image.shape[-1]
        if n_channels not in (3, 4):
            gs.raise_exception(f"Pixels are drawn as RGB or RGBA, got {n_channels} channels.")
        image = image.mul(255.0).clamp_(0.0, 255.0).round_() if image.is_floating_point() else image.clamp(0, 255)
        width, height = self._texture.res
        n_images = len(envs_idx) if self._texture.canvas == "per_env" else 1
        image = broadcast_tensor(image, torch.uint8, (n_images, height, width, n_channels))
        if n_channels == 3:
            image = torch.cat((image, image.new_full((n_images, height, width, 1), 255)), dim=-1)
        self._selection[envs_idx] = DRAWN_IMAGE
        self._selection_version += 1
        for callback in self._callbacks:
            callback(envs_idx, image)

    def reset(self, envs_idx=None) -> None:
        """Show the initial image again in the selected environments."""
        self.set_texture(self._initial_image, envs_idx)


def get_dynamic_texture(handles: list[DynamicTextureHandle], name: str | None, owner: str) -> DynamicTextureHandle:
    """Return the dynamic texture of 'owner' named 'name', or its only one if 'name' is None.

    'handles' are the dynamic textures that the visual geoms of 'owner' draw, and 'owner' describes it in the errors.
    """
    if name is not None:
        handles = [handle for handle in handles if handle.name == name]
    if len(handles) == 1:
        return handles[0]
    if not handles:
        target = "any dynamic texture" if name is None else f"a dynamic texture named '{name}'"
        gs.raise_exception(f"No visual geom of {owner} draws {target}.")
    names = [handle.name for handle in handles]
    gs.raise_exception(f"The visual geoms of {owner} draw {len(handles)} dynamic textures {names}. Pick one by 'name'.")
