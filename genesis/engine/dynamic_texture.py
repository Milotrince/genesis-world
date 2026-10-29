"""Runtime handles writing the images of the dynamic textures of a scene (see ``gs.textures.DynamicTexture``)."""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.constants import TEXTURE_SLOT
from genesis.options.textures import DynamicTexture
from genesis.utils.misc import broadcast_tensor, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidVisGeom
    from genesis.engine.scene import Scene


class DynamicTextureBinding(NamedTuple):
    """A visual geom drawing a dynamic texture, and the slot of its surface that the texture fills."""

    vgeom: "RigidVisGeom"
    slot: TEXTURE_SLOT


class DynamicTextureHandle:
    """Write the image of one dynamic texture while the scene runs.

    The image is an RGBA uint8 tensor on the device of the scene, of shape (n_envs, height, width, 4) when the texture
    is batched in a scene with parallel environments, and (height, width, 4) otherwise. 'set_image' copies an image in,
    while a write in place through 'image' is followed by 'notify_changed'. Both advance the counter of the
    environments they touch in 'versions', so that each renderer uploads the environments whose counter moved since it
    last drew them, whatever the other renderers did meanwhile.

    The handles of a scene are listed by ``scene.dynamic_textures``, each with the visual geoms drawing it
    ('bindings'), which is what a renderer outside of Genesis binds to its own textures.
    """

    def __init__(self, scene: "Scene", texture: DynamicTexture, bindings: list[DynamicTextureBinding]):
        self._scene = scene
        self._texture = texture
        self._bindings = bindings
        self._is_batched = texture.is_batched and scene.n_envs > 0
        width, height = texture.res
        self._initial_image = torch.as_tensor(texture.get_initial_image(), device=gs.device)
        shape = (scene.n_envs, height, width, 4) if self._is_batched else (height, width, 4)
        self._image = self._initial_image.expand(shape).clone()
        self._versions = np.zeros(scene.n_envs if self._is_batched else 1, dtype=gs.np_int)

    @property
    def texture(self) -> DynamicTexture:
        """The declaration of the texture, as the surface of its visual geoms holds it."""
        return self._texture

    @property
    def name(self) -> str | None:
        """The name of the texture, if it was given one."""
        return self._texture.name

    @property
    def res(self) -> tuple[int, int]:
        """The resolution of the image, as (width, height) in pixels."""
        return self._texture.res

    @property
    def encoding(self) -> str:
        """The encoding of the image, 'srgb' or 'linear'."""
        return self._texture.encoding

    @property
    def is_batched(self) -> bool:
        """Whether each parallel environment shows an image of its own."""
        return self._is_batched

    @property
    def bindings(self) -> list[DynamicTextureBinding]:
        """The visual geoms drawing the texture, each with the slot of its surface that the texture fills."""
        return self._bindings

    @property
    def image(self) -> torch.Tensor:
        """The image of the texture, written in place before calling 'notify_changed'."""
        return self._image

    @property
    def versions(self) -> np.ndarray:
        """The number of writes to the image of each environment, or of the shared image, so far."""
        return self._versions

    def set_image(self, image: "np.typing.ArrayLike | torch.Tensor", envs_idx=None) -> None:
        """Copy an image into the texture and notify the renderers.

        'image' is RGB or RGBA, as uint8 values or as floats between 0 and 1, of shape (height, width, channels) for
        every selected environment, or (len(envs_idx), height, width, channels) for one image each. A shared texture
        takes no 'envs_idx'.
        """
        envs_idx = self._sanitize_envs_idx(envs_idx)
        # Torch warns on wrapping a read-only array, such as one read from an image, which the copy below never writes
        if isinstance(image, np.ndarray) and not image.flags.writeable:
            image = image.copy()
        image = torch.as_tensor(image, device=gs.device)
        image = (image.clamp(0.0, 1.0) * 255.0).round() if image.is_floating_point() else image.clamp(0, 255)
        if image.shape[-1] == 3:
            image = torch.cat((image, torch.full_like(image[..., :1], 255)), dim=-1)
        width, height = self._texture.res
        if self._is_batched:
            self._image[envs_idx] = broadcast_tensor(image, torch.uint8, (len(envs_idx), height, width, 4))
        else:
            self._image[:] = broadcast_tensor(image, torch.uint8, (height, width, 4))
        self.notify_changed(envs_idx)

    def notify_changed(self, envs_idx=None) -> None:
        """Notify the renderers that the image of some environments was written in place through 'image'."""
        envs_idx = self._sanitize_envs_idx(envs_idx)
        if envs_idx is None:
            self._versions += 1
        else:
            self._versions[tensor_to_array(envs_idx)] += 1

    def reset(self, envs_idx=None) -> None:
        """Restore the initial image of the texture."""
        envs_idx = self._sanitize_envs_idx(envs_idx)
        if self._is_batched:
            self._image[envs_idx] = self._initial_image
        else:
            self._image[:] = self._initial_image
        self.notify_changed(envs_idx)

    def _sanitize_envs_idx(self, envs_idx) -> torch.Tensor | None:
        if not self._is_batched:
            if envs_idx is not None:
                gs.raise_exception("A dynamic texture shared by every environment takes no 'envs_idx'.")
            return None
        return self._scene._sanitize_envs_idx(envs_idx)
