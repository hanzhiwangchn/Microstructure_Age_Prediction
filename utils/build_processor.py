import logging
import torchio as tio

logger = logging.getLogger(__name__)
results_folder = 'model_ckpt_results'


def medical_augmentation_pt(images):
    training_transform = tio.Compose([
        tio.RandomBlur(p=0.5),  # blur 50% of times
        tio.RandomNoise(p=0.5),  # Gaussian noise 50% of times
        tio.RandomFlip(flip_probability=0.5),
    ])
    return training_transform(images)
