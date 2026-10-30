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


# The selection of an environment showing the pixels drawn on its canvas, in place of a variant
DRAWN_IMAGE = -1


class DynamicTextureBinding(NamedTuple):
    """A visual geom drawing a dynamic texture, and the slot of its surface that the texture fills."""

    vgeom: "RigidVisGeom"
    slot: TEXTURE_SLOT


class DynamicTextureHandle:
    """The variant each environment shows for one dynamic texture, and the renderers drawing it.

    Genesis holds the selection alone: the position of the variant each environment shows, or 'DRAWN_IMAGE' for the
    pixels drawn on its canvas. The renderers hold the pixels, and each one implements the same contract:

    - at build, it allocates the variants ('texture.get_variants_image()'), then the canvases, one per
      environment it renders when 'texture.is_batched' and a single shared one otherwise, binds them to the visual
      geoms in 'bindings', and subscribes a callback with 'subscribe';
    - on every drawing, the callback receives the environments drawn and their pixels, as a uint8 RGBA tensor on the
      device of shape (len(envs_idx), height, width, 4), or (1, height, width, 4) when the texture is not batched;
    - at sync, it reads 'selection' when 'selection_version' moved since it last read it.
    """

    def __init__(self, scene: "Scene", texture: DynamicTexture, bindings: list[DynamicTextureBinding]):
        self._scene = scene
        self._texture = texture
        self._bindings = bindings
        (self._initial_variant,) = texture.get_variants_idx(texture.initial_variant)
        self._selection = np.full(scene._B, self._initial_variant, dtype=gs.np_int)
        self._selection_version = 0
        self._callbacks: list[Callable[[np.ndarray, torch.Tensor], None]] = []

    @property
    def texture_name(self) -> str | None:
        """The name of the texture, by which the setters of an entity holding several dynamic textures pick it."""
        return self._texture.texture_name

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
        """The variant each environment shows, as a read-only array (see 'DynamicTextureHandle')."""
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
        """Return the position of the variant each selected environment shows, 'DRAWN_IMAGE' for drawn pixels."""
        selection = self._selection[tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))]
        selection = torch.as_tensor(selection, dtype=gs.tc_int, device=gs.device)
        return selection[0] if self._scene.n_envs == 0 else selection

    def set_texture(self, texture, envs_idx=None) -> None:
        """Show a variant, or draw pixels, in the selected environments.

        'texture' is either variants or pixels. A variant is given by position or by name, one for all the selected
        environments, or a sequence of them, one per selected environment. Pixels are a ``ColorTexture``, an
        ``ImageTexture``, or an RGB or RGBA array of uint8 values or of floats between 0 and 1, of shape (height, width,
        channels) for all the selected environments or (len(envs_idx), height, width, channels) for one each. Integer
        and float values are clamped to the range of a channel. Pixels go to the canvas of each selected environment
        when the texture is batched, and to the canvas every environment shares otherwise, which takes a single image.
        """
        is_envs_selected = envs_idx is not None
        envs_idx = tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))
        if isinstance(texture, (ColorTexture, ImageTexture)):
            texture = texture_to_rgba(texture, self._texture.resolution)
        texture_ndim = texture.ndim if torch.is_tensor(texture) else np.ndim(texture)
        if texture_ndim <= 1:
            if torch.is_tensor(texture):
                texture = tensor_to_array(texture)
            self._selection[envs_idx] = self._texture.get_variants_idx(texture)
            self._selection_version += 1
            return
        if texture_ndim not in (3, 4):
            gs.raise_exception(
                "A dynamic texture is set to variants, or to pixels of shape (height, width, channels), got an "
                f"array of {texture_ndim} dims."
            )

        # Torch warns on wrapping a read-only array, such as one read from an image, which the copy below never writes
        if isinstance(texture, np.ndarray) and not texture.flags.writeable:
            texture = texture.copy()
        image = torch.as_tensor(texture, device=gs.device)
        if not self._texture.is_batched:
            if image.ndim == 4:
                gs.raise_exception("A dynamic texture that is not batched takes a single image.")
            if is_envs_selected:
                gs.logger.warning(
                    "Pixels drawn on a dynamic texture that is not batched are shared by every environment showing "
                    "them. Declare it with 'is_batched=True' to draw per environment."
                )
        n_channels = image.shape[-1]
        if n_channels not in (3, 4):
            gs.raise_exception(f"Pixels are drawn as RGB or RGBA, got {n_channels} channels.")
        image = image.mul(255.0).clamp_(0.0, 255.0).round_() if image.is_floating_point() else image.clamp(0, 255)
        width, height = self._texture.resolution
        n_images = len(envs_idx) if self._texture.is_batched else 1
        image = broadcast_tensor(image, torch.uint8, (n_images, height, width, n_channels))
        if n_channels == 3:
            image = torch.cat((image, image.new_full((n_images, height, width, 1), 255)), dim=-1)
        self._selection[envs_idx] = DRAWN_IMAGE
        self._selection_version += 1
        for callback in self._callbacks:
            callback(envs_idx, image)

    def reset(self, envs_idx=None) -> None:
        """Show the initial variant again in the selected environments."""
        self.set_texture(self._initial_variant, envs_idx)


def get_dynamic_texture(
    handles: list[DynamicTextureHandle], texture_name: str | None, owner: str
) -> DynamicTextureHandle:
    """Return the dynamic texture of 'owner' named 'texture_name', or its only one if 'texture_name' is None.

    'handles' are the dynamic textures that the visual geoms of 'owner' draw, and 'owner' describes it in the errors.
    """
    if texture_name is not None:
        handles = [handle for handle in handles if handle.texture_name == texture_name]
    if len(handles) == 1:
        return handles[0]
    if not handles:
        target = "any dynamic texture" if texture_name is None else f"a dynamic texture named '{texture_name}'"
        gs.raise_exception(f"No visual geom of {owner} draws {target}.")
    textures_name = [handle.texture_name for handle in handles]
    gs.raise_exception(
        f"The visual geoms of {owner} draw {len(handles)} dynamic textures {textures_name}. Pick one by 'texture_name'."
    )
