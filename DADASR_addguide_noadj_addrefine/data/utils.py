import random

from torchvision.transforms import RandomCrop, RandomRotation
import torchvision.transforms.functional as F

ROTATION_EXPAND = False
ROTATION_CENTER = None  # image center
ROTATION_FILL = 0.


def random_horizontal_flip(images, p=0.5):
    if random.random() < p:
        return [image.flip(-1) for image in images]
    return images


def random_rotate(images, max_rotation_angle, interpolation, crop_valid=False):
    angle = RandomRotation.get_params([-max_rotation_angle, max_rotation_angle])
    if crop_valid:
        rotated = [F.rotate(image, angle, interpolation, True, ROTATION_CENTER, ROTATION_FILL) for image in images]
        crop_params = np.floor(np.asarray(rotated[0].shape[1:3]) - 2. *
                      (np.sin(np.abs(angle * np.pi / 180.)) * np.asarray(images[0].shape[1:3][::-1]))).astype(int)
        return [F.center_crop(image, crop_params) for image in rotated]
    else:
        return [F.rotate(image, angle, interpolation, ROTATION_EXPAND, ROTATION_CENTER, ROTATION_FILL) for image in images]


def random_crop(images, crop_size):
    crop_params = RandomCrop.get_params(images[0], crop_size)
    return [F.crop(image, *crop_params) for image in images]
