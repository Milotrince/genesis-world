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
    """Select and write the images of one dynamic texture while the scene runs.

    The images are an RGBA uint8 tensor on the device of the scene, of shape (n_images, height, width, 4), fixed in
    number. 'selection' holds the image each environment shows, one per environment when the texture is batched in a
    scene with parallel environments and a single one otherwise. 'select' switches images, 'set_image' copies pixels
    into images, and a write in place through 'images' is followed by 'notify_changed'.

    Each write advances the counter of the images it touches in 'images_version', and each selection advances
    'selection_version', so that every renderer uploads the images and refreshes the selection that moved since it last
    drew them, whatever the other renderers did meanwhile. The handles of a scene are listed by
    ``scene.dynamic_textures``, each with the visual geoms drawing it ('bindings'), which is what a renderer outside of
    Genesis binds to its own textures.
    """

    def __init__(self, scene: "Scene", texture: DynamicTexture, bindings: list[DynamicTextureBinding]):
        self._scene = scene
        self._texture = texture
        self._bindings = bindings
        self._is_batched = texture.is_batched and scene.n_envs > 0
        self._initial_images = torch.as_tensor(texture.get_initial_images(), device=gs.device)
        self._images = self._initial_images.clone()
        (self._initial_selection,) = texture.get_images_idx(texture.initial_image)
        self._selection = np.full(scene.n_envs if self._is_batched else 1, self._initial_selection, dtype=gs.np_int)
        self._images_version = np.zeros(len(self._images), dtype=gs.np_int)
        self._selection_version = 0

    @property
    def texture(self) -> DynamicTexture:
        """The declaration of the texture, as the surface of its visual geoms holds it."""
        return self._texture

    @property
    def name(self) -> str | None:
        """The name of the texture, if it was given one."""
        return self._texture.name

    @property
    def is_batched(self) -> bool:
        """Whether each parallel environment selects an image of its own."""
        return self._is_batched

    @property
    def bindings(self) -> list[DynamicTextureBinding]:
        """The visual geoms drawing the texture, each with the slot of its surface that the texture fills."""
        return self._bindings

    @property
    def images(self) -> torch.Tensor:
        """The images of the texture, written in place before calling 'notify_changed'."""
        return self._images

    @property
    def images_name(self) -> tuple[str, ...] | None:
        """The name of each image, None for images given by position alone."""
        return self._texture.images_name

    @property
    def selection(self) -> np.ndarray:
        """The position of the image each environment shows, or of the image every environment shows if shared."""
        return self._selection

    @property
    def images_version(self) -> np.ndarray:
        """The number of writes to each image so far."""
        return self._images_version

    @property
    def selection_version(self) -> int:
        """The number of selections so far."""
        return self._selection_version

    def select(self, images_idx: "int | str | np.typing.ArrayLike", envs_idx=None) -> None:
        """Show an image in the selected environments, one for all of them or one each, by position or by name.

        A texture shared by every environment takes no 'envs_idx'.
        """
        envs_idx = self._sanitize_envs_idx(envs_idx)
        images_idx = self._texture.get_images_idx(
            tensor_to_array(images_idx) if torch.is_tensor(images_idx) else images_idx
        )
        if envs_idx is None:
            self._selection[:] = images_idx
        else:
            self._selection[tensor_to_array(envs_idx)] = images_idx
        self._selection_version += 1

    def set_image(
        self, image: "np.typing.ArrayLike | torch.Tensor", images_idx: "int | str | np.typing.ArrayLike" = 0
    ) -> None:
        """Copy pixels into images, given by position or by name, and notify the renderers.

        'image' is RGB or RGBA, as uint8 values or as floats between 0 and 1, of shape (height, width, channels) for
        every given image, or (len(images_idx), height, width, channels) for one each.
        """
        images_idx = self._texture.get_images_idx(images_idx)
        # Torch warns on wrapping a read-only array, such as one read from an image, which the copy below never writes
        if isinstance(image, np.ndarray) and not image.flags.writeable:
            image = image.copy()
        image = torch.as_tensor(image, device=gs.device)
        image = (image.clamp(0.0, 1.0) * 255.0).round() if image.is_floating_point() else image.clamp(0, 255)
        if image.shape[-1] == 3:
            image = torch.cat((image, torch.full_like(image[..., :1], 255)), dim=-1)
        width, height = self._texture.res
        self._images[images_idx] = broadcast_tensor(image, torch.uint8, (len(images_idx), height, width, 4))
        self._images_version[images_idx] += 1

    def notify_changed(self, images_idx: "int | str | np.typing.ArrayLike | None" = None) -> None:
        """Notify the renderers that images were written in place through 'images', every image if None."""
        if images_idx is None:
            self._images_version += 1
        else:
            self._images_version[self._texture.get_images_idx(images_idx)] += 1

    def reset(self, envs_idx=None) -> None:
        """Restore the initial selection of some environments, and the initial images along with it for all of them."""
        if envs_idx is None:
            self._images[:] = self._initial_images
            self._images_version += 1
        self.select(self._initial_selection, envs_idx)

    def _sanitize_envs_idx(self, envs_idx) -> torch.Tensor | None:
        if not self._is_batched:
            if envs_idx is not None:
                gs.raise_exception("A dynamic texture shared by every environment takes no 'envs_idx'.")
            return None
        return self._scene._sanitize_envs_idx(envs_idx)
