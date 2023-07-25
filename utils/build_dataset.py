import logging, os

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from torchvision import transforms

from utils.common_utils import TrainDataset, ValidationDataset, TestDataset, \
    ToTensor_MRI, medical_augmentation_pt

logger = logging.getLogger(__name__)


def build_dataset(args):
    """main function for dataset building"""
    if args.dataset == 'wand_compact':
        dataset_train, dataset_val, dataset_test, lim, input_shape, median_age = build_dataset_wand(args)
    # update arguments
    args.age_limits = lim
    args.input_shape = input_shape
    args.median_age = median_age
    return dataset_train, dataset_val, dataset_test, args


# ------------------- WAND Dataset ---------------------
def build_dataset_wand(args):
    """
    load WAND data
    """
    # load data(in .npy format)
    images = np.load(os.path.join(args.data_dir, f'subject_images_{args.image_modality}.npy'))
    age = np.load(os.path.join(args.data_dir, f'subject_age_{args.image_modality}.npy'))

    df = pd.DataFrame()
    df['Age'] = age
    # retrieve the minimum, maximum and median age for skewed loss
    lim = (df['Age'].min(), df['Age'].max())
    median_age = df['Age'].median()

    # add color channel dimension (bs, H, D, W) -> (bs, 1, H, D, W)
    images = np.expand_dims(images, axis=1)

    assert len(images.shape) == 5, images.shape
    assert images.shape[1] == 1
    assert len(images) == len(df)

    # assign a categorical label to Age for Stratified Split
    df['Age_categorical'] = pd.qcut(df['Age'], 10, labels=[i for i in range(10)])

    # Stratified train validation-test Split
    split = StratifiedShuffleSplit(test_size=args.val_test_size, random_state=args.random_state)
    train_index, validation_test_index = next(split.split(df, df['Age_categorical']))
    stratified_validation_test_set = df.loc[validation_test_index]
    assert sorted(train_index.tolist() + validation_test_index.tolist()) == list(range(len(df)))

    # Stratified validation test Split
    split2 = StratifiedShuffleSplit(test_size=args.test_size, random_state=args.random_state)
    validation_index, test_index = next(split2.split(stratified_validation_test_set,
                                                     stratified_validation_test_set['Age_categorical']))

    # NOTE: StratifiedShuffleSplit returns RangeIndex instead of the Original Index of the new DataFrame
    assert sorted(validation_index.tolist() + test_index.tolist()) == \
        list(range(len(stratified_validation_test_set.index)))
    assert sorted(validation_index.tolist() + test_index.tolist()) != \
        sorted(list(stratified_validation_test_set.index))

    # get the correct index of the original DataFrame for validation/test set
    validation_index = validation_test_index[validation_index]
    test_index = validation_test_index[test_index]

    # ensure there is no duplicated index in 3 data-sets
    assert sorted(train_index.tolist() + validation_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # get train/validation/test set
    train_images = images[train_index].astype(np.float32)
    validation_images = images[validation_index].astype(np.float32)
    test_images = images[test_index].astype(np.float32)
    input_shape = train_images.shape[1:]
    del images

    # add dimension for labels: (32,) -> (32, 1)
    train_labels = np.expand_dims(df.loc[train_index, 'Age'].values, axis=1).astype(np.float32)
    validation_labels = np.expand_dims(df.loc[validation_index, 'Age'].values, axis=1).astype(np.float32)
    test_labels = np.expand_dims(df.loc[test_index, 'Age'].values, axis=1).astype(np.float32)

    logger.info(f'Training images shape: {train_images.shape}, validation images shape: {validation_images.shape}, '
                f'testing images shape: {test_images.shape}, training labels shape: {train_labels.shape}, '
                f'validation labels shape: {validation_labels.shape}, testing labels shape: {test_labels.shape}')

    # Pytorch Data-set for train set. Apply data augmentation if needed using "torchio"
    dataset_train = TrainDataset(images=train_images, labels=train_labels, 
        transform=transforms.Compose([ToTensor_MRI()]), medical_augment=medical_augmentation_pt)
    del train_images, train_labels
    # Pytorch Data-set for validation set
    dataset_val = ValidationDataset(images=validation_images, labels=validation_labels,
                                           transform=transforms.Compose([ToTensor_MRI()]))
    del validation_images, validation_labels
    # Pytorch Data-set for test set
    dataset_test = TestDataset(images=test_images, labels=test_labels,
                               transform=transforms.Compose([ToTensor_MRI()]))
    del test_images, test_labels

    return dataset_train, dataset_val, dataset_test, lim, input_shape, median_age
