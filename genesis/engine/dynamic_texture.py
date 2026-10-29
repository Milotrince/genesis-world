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
    the pixels drawn on its canvas. The renderers hold the pixels. Each renderer builds the declared images from the
    declaration ('texture'), binds them to the visual geoms listed in 'bindings', and subscribes a callback, which every
    change calls with the environments it touches and the pixels drawn on their canvases, None for a change of the
    selection alone. A texture whose canvas every environment shares passes a single image.
    """

    def __init__(self, scene: "Scene", texture: DynamicTexture, bindings: list[DynamicTextureBinding]):
        self._scene = scene
        self._texture = texture
        self._bindings = bindings
        (self._initial_image,) = texture.get_images_idx(texture.initial_image)
        self._selection = np.full(scene._B, self._initial_image, dtype=gs.np_int)
        self._callbacks: list[Callable[[np.ndarray, torch.Tensor | None], None]] = []

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

    def subscribe(self, callback: Callable[[np.ndarray, torch.Tensor | None], None]) -> None:
        """Call 'callback' on every change of the texture (see 'DynamicTextureHandle')."""
        self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[np.ndarray, torch.Tensor | None], None]) -> None:
        """Stop calling a callback given to 'subscribe'."""
        self._callbacks.remove(callback)

    def get_texture(self, envs_idx=None) -> torch.Tensor:
        """Return the position of the declared image each selected environment shows, 'DRAWN_IMAGE' for drawn pixels."""
        selection = self._selection[tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))]
        selection = torch.as_tensor(selection, dtype=gs.tc_int, device=gs.device)
        return selection[0] if self._scene.n_envs == 0 else selection

    def set_texture(self, texture, envs_idx=None) -> None:
        """Show a declared image, or draw pixels, in the selected environments.

        'texture' is either declared images, given by position or by name, one for all the selected environments or one
        each, or pixels to draw: a ``ColorTexture`` or an ``ImageTexture``, or an RGB or RGBA array of uint8 values or
        of floats between 0 and 1, of shape (height, width, channels) for all the selected environments or
        (len(envs_idx), height, width, channels) for one each. Pixels are drawn on the canvas of each selected
        environment, which requires a texture declared drawable, and a texture whose canvas is shared by every
        environment takes a single image.
        """
        envs_idx = tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))
        if isinstance(texture, (ColorTexture, ImageTexture)):
            texture = texture_to_rgba(texture, self._texture.res)
        texture_ndim = texture.ndim if torch.is_tensor(texture) else np.ndim(texture)
        if texture_ndim <= 1:
            if torch.is_tensor(texture):
                texture = tensor_to_array(texture)
            self._selection[envs_idx] = self._texture.get_images_idx(texture)
            for callback in self._callbacks:
                callback(envs_idx, None)
            return

        if not self._texture.is_drawable:
            gs.raise_exception("Pixels are drawn on a dynamic texture declared with 'is_drawable=True'.")
        if not self._texture.is_batched and texture_ndim > 3:
            gs.raise_exception("A dynamic texture whose canvas every environment shares takes a single image.")
        # Torch warns on wrapping a read-only array, such as one read from an image, which the copy below never writes
        if isinstance(texture, np.ndarray) and not texture.flags.writeable:
            texture = texture.copy()
        image = torch.as_tensor(texture, device=gs.device)
        image = image.mul(255.0).clamp_(0.0, 255.0).round_() if image.is_floating_point() else image.clamp(0, 255)
        width, height = self._texture.res
        n_images = len(envs_idx) if self._texture.is_batched else 1
        n_channels = image.shape[-1]
        if n_channels not in (3, 4):
            gs.raise_exception(f"Pixels are drawn as RGB or RGBA, got {n_channels} channels.")
        image = broadcast_tensor(image, torch.uint8, (n_images, height, width, n_channels))
        if n_channels == 3:
            image = torch.cat((image, image.new_full((n_images, height, width, 1), 255)), dim=-1)
        self._selection[envs_idx] = DRAWN_IMAGE
        for callback in self._callbacks:
            callback(envs_idx, image)

    def reset(self, envs_idx=None) -> None:
        """Show the initial image again in the selected environments."""
        self.set_texture(self._initial_image, envs_idx)
