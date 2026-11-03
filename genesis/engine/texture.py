"""Runtime state of the dynamic textures of a scene (see ``gs.textures.DynamicTexture``)."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
from genesis.constants import DRAWN_IMAGE, TEXTURE_MAP_TYPE
from genesis.engine.state_change import StateChange, Subscriber
from genesis.options.textures import ColorTexture, DynamicTexture, ImageTexture, texture_to_rgba
from genesis.typing import IndexType
from genesis.utils.misc import broadcast_tensor, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidVisGeom
    from genesis.engine.scene import Scene


# The variants to show, by position or by name, or the pixels to draw (see 'Texture.set_texture')
TextureValueType = int | str | Sequence[int | str] | np.ndarray | torch.Tensor | ColorTexture | ImageTexture


class Texture:
    """The variant each environment shows for the dynamic texture of one map of a visual geom, and its renderers.

    Genesis holds the active variant indices alone: the position of the variant each environment shows, or
    'DRAWN_IMAGE' for the pixels drawn for it. The renderers hold the pixels, and each one implements the same contract:

    - at build, it allocates the variants ('options.get_variants_image(texture_map_type)'), then one drawn image per
      environment it renders when 'options.enable_drawing', binds them to the map of type 'texture_map_type' of
      'vgeom', subscribes a ``Subscriber`` to ``StateChange.VISUAL`` with 'subscribe', and a callback with
      'subscribe_drawing';
    - on every drawing, the callback receives the environments drawn and their pixels, as an RGBA tensor on the device
      of shape (len(envs_idx), height, width, 4);
    - at sync, when its subscriber holds ``StateChange.VISUAL`` pending, it clears it, then reads 'active_variants_idx'.

    The images are uint8 sRGB values in the diffuse map, and float16 linear values in the emissive map.
    """

    def __init__(
        self, scene: "Scene", options: DynamicTexture, texture_map_type: TEXTURE_MAP_TYPE, vgeom: "RigidVisGeom"
    ):
        self._scene = scene
        self._options = options
        self._texture_map_type = texture_map_type
        self._vgeom = vgeom
        (self._initial_variant,) = options.get_variants_idx(options.initial_variant)
        self._active_variants_idx = np.full(scene._B, self._initial_variant, dtype=gs.np_int)
        self._subscribers: set[Subscriber] = set()
        self._drawing_callbacks: list[Callable[[np.ndarray, torch.Tensor], None]] = []

    @property
    def options(self) -> DynamicTexture:
        """The declaration of the texture, as the surface of its visual geom holds it."""
        return self._options

    @property
    def texture_map_type(self) -> TEXTURE_MAP_TYPE:
        """The map of the surface that the texture fills, which sets the format of its images."""
        return self._texture_map_type

    @property
    def vgeom(self) -> "RigidVisGeom":
        """The visual geom drawing the texture."""
        return self._vgeom

    @property
    def active_variants_idx(self) -> np.ndarray:
        """The position of the variant each environment shows, as a read-only array (see 'Texture')."""
        active_variants_idx = self._active_variants_idx.view()
        active_variants_idx.flags.writeable = False
        return active_variants_idx

    def subscribe(self, subscriber: Subscriber) -> None:
        """Notify 'subscriber' of every change of 'active_variants_idx', as ``StateChange.VISUAL`` (see 'Texture')."""
        self._subscribers.add(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """Stop notifying a subscriber given to 'subscribe'."""
        self._subscribers.discard(subscriber)

    def subscribe_drawing(self, callback: Callable[[np.ndarray, torch.Tensor], None]) -> None:
        """Call 'callback' with the pixels of every drawing (see 'Texture').

        Genesis keeps no pixels, so a renderer subscribes before any environment shows a drawing.
        """
        if (self._active_variants_idx == DRAWN_IMAGE).any():
            gs.raise_exception("A renderer cannot be added once pixels have been drawn on a dynamic texture.")
        self._drawing_callbacks.append(callback)

    def unsubscribe_drawing(self, callback: Callable[[np.ndarray, torch.Tensor], None]) -> None:
        """Stop calling a callback given to 'subscribe_drawing'."""
        self._drawing_callbacks.remove(callback)

    def get_texture(self, envs_idx: IndexType | None = None) -> torch.Tensor:
        """Return the position of the variant each selected environment shows, 'DRAWN_IMAGE' for drawn pixels."""
        active_variants_idx = self._active_variants_idx[tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))]
        active_variants_idx = torch.as_tensor(active_variants_idx, dtype=gs.tc_int, device=gs.device)
        return active_variants_idx[0] if self._scene.n_envs == 0 else active_variants_idx

    def set_texture(self, texture: TextureValueType, envs_idx: IndexType | None = None) -> None:
        """Show a variant, or draw pixels, in the selected environments.

        'texture' is either variants or pixels. A variant is given by position or by name, one for all the selected
        environments, or a sequence of them, one per selected environment. Pixels are a ``ColorTexture``, an
        ``ImageTexture``, or an RGB or RGBA array of shape (height, width, channels) for all the selected environments
        or (len(envs_idx), height, width, channels) for one each, and require 'enable_drawing' in the declaration.

        An array holds uint8 values, or floats on the scale where 1 stands for 255. In the diffuse map, they are sRGB
        values clamped between 0 and 1. In the emissive map, they are linear values clamped at 0 alone, so that an
        emission may exceed 1.
        """
        envs_idx = tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))
        if isinstance(texture, (ColorTexture, ImageTexture)):
            texture = texture_to_rgba(texture, self._options.resolution, self._texture_map_type)
        texture_ndim = texture.ndim if torch.is_tensor(texture) else np.ndim(texture)
        if texture_ndim <= 1:
            if torch.is_tensor(texture):
                texture = tensor_to_array(texture)
            variants_idx = self._options.get_variants_idx(texture)
            if len(variants_idx) not in (1, len(envs_idx)):
                gs.raise_exception(
                    f"A dynamic texture shows one variant, or one per selected environment, got {len(variants_idx)} "
                    f"variants for {len(envs_idx)} environments."
                )
            self._active_variants_idx[envs_idx] = variants_idx
        else:
            if texture_ndim not in (3, 4):
                gs.raise_exception(
                    "A dynamic texture is set to variants, or to pixels of shape (height, width, channels), got an "
                    f"array of {texture_ndim} dims."
                )

            # Torch warns on wrapping a read-only array, such as an image read from a file, which the copy below
            # never writes
            if isinstance(texture, np.ndarray) and not texture.flags.writeable:
                texture = texture.copy()
            image = torch.as_tensor(texture, device=gs.device)
            n_channels = image.shape[-1]
            if n_channels not in (3, 4):
                gs.raise_exception(f"Pixels are drawn as RGB or RGBA, got {n_channels} channels.")
            if not self._options.enable_drawing:
                gs.raise_exception("Pixels are drawn on a dynamic texture declared with 'enable_drawing=True' alone.")
            if self._texture_map_type == TEXTURE_MAP_TYPE.EMISSIVE:
                dtype, alpha = torch.float16, 1.0
                if image.is_floating_point():
                    image = image.clamp(0.0, torch.finfo(dtype).max)
                else:
                    image = image.clamp(0, 255) / 255.0
            else:
                dtype, alpha = torch.uint8, 255
                if image.is_floating_point():
                    image = image.mul(255.0).clamp_(0.0, 255.0).round_()
                else:
                    image = image.clamp(0, 255)
            width, height = self._options.resolution
            n_images = len(envs_idx)
            image = broadcast_tensor(image, dtype, (n_images, height, width, n_channels))
            if n_channels == 3:
                image = torch.cat((image, image.new_full((n_images, height, width, 1), alpha)), dim=-1)
            self._active_variants_idx[envs_idx] = DRAWN_IMAGE
            for callback in self._drawing_callbacks:
                callback(envs_idx, image)
        for subscriber in self._subscribers:
            if StateChange.VISUAL in subscriber.to:
                subscriber.notify(StateChange.VISUAL, envs_idx)

    def reset(self, envs_idx: IndexType | None = None) -> None:
        """Show the initial variant again in the selected environments."""
        self.set_texture(self._initial_variant, envs_idx)


def get_dynamic_texture(textures: list[Texture], texture_map_type: TEXTURE_MAP_TYPE | None) -> Texture:
    """Return the dynamic texture filling the map of type 'texture_map_type', or the only one of 'textures' if None.

    'textures' are the dynamic textures of one visual geom, one per map at most.
    """
    if texture_map_type is not None:
        textures = [texture for texture in textures if texture.texture_map_type == texture_map_type]
    if len(textures) == 1:
        return textures[0]
    if not textures:
        target = "" if texture_map_type is None else f" in the {texture_map_type.name.lower()} map"
        gs.raise_exception(f"This visual geom draws no dynamic texture{target}.")
    gs.raise_exception("This visual geom draws a dynamic texture in both maps. Pick one by 'texture_map_type'.")
