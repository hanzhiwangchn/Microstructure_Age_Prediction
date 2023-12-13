import logging, os

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from torchvision import transforms

from utils.image_utils.common_utils import TrainDataset, ValDataset, TestDataset, \
    ToTensor_MRI, medical_augmentation_pt

logger = logging.getLogger(__name__)


def build_dataset(args):
    """main function for dataset building"""
    if args.dataset == 'wand_micro':
        dataset_train, dataset_val, dataset_test, lim, input_shape, median_age = build_dataset_wand_micro(args)
    elif args.dataset == 'wand_t1w':
        dataset_train, dataset_val, dataset_test, lim, input_shape, median_age = build_dataset_wand_t1w(args)

    # update arguments
    args.age_limits = lim
    args.input_shape = input_shape
    args.median_age = median_age
    return dataset_train, dataset_val, dataset_test, args


# ------------------- WAND Dataset ---------------------
def build_dataset_wand_micro(args):
    """load WAND data"""
    # load data(in .npy format)
    images = np.load(os.path.join(args.data_dir, f'subject_images_{args.image_modality}.npy'))
    age = np.load(os.path.join(args.data_dir, f'subject_age_{args.image_modality}.npy'))

    df = pd.DataFrame(data=age, columns=['Age'])
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

    # train-val & test split
    split = StratifiedShuffleSplit(test_size=args.test_size, random_state=args.random_state)
    train_val_index, test_index = next(split.split(df, df['Age_categorical']))
    stratified_train_val_set = df.loc[train_val_index]
    assert sorted(train_val_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # train & val split
    split2 = StratifiedShuffleSplit(test_size=args.val_size, random_state=args.random_state)
    train_index, val_index = next(split2.split(stratified_train_val_set, stratified_train_val_set['Age_categorical']))

    # NOTE: StratifiedShuffleSplit returns RangeIndex instead of the Original Index of the new DataFrame
    assert sorted(train_index.tolist() + val_index.tolist()) == list(range(len(stratified_train_val_set.index)))
    assert sorted(train_index.tolist() + val_index.tolist()) != sorted(list(stratified_train_val_set.index))

    # get the correct index of original DataFrame for train/val set
    train_index = train_val_index[train_index]
    val_index = train_val_index[val_index]

    # ensure there is no duplicated index
    assert sorted(train_index.tolist() + val_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # split data based on index
    train_images = images[train_index].astype(np.float32)
    val_images = images[val_index].astype(np.float32)
    test_images = images[test_index].astype(np.float32)
    input_shape = train_images.shape[1:]
    del images

    # add dimension for labels: (batch_size,) -> (batch_size, 1)
    train_labels = np.expand_dims(df.loc[train_index, 'Age'].values, axis=1).astype(np.float32)
    val_labels = np.expand_dims(df.loc[val_index, 'Age'].values, axis=1).astype(np.float32)
    test_labels = np.expand_dims(df.loc[test_index, 'Age'].values, axis=1).astype(np.float32)

    logger.info(f'Training images shape: {train_images.shape}, validation images shape: {val_images.shape}, '
                f'testing images shape: {test_images.shape}, training labels shape: {train_labels.shape}, '
                f'validation labels shape: {val_labels.shape}, testing labels shape: {test_labels.shape}')

    # Pytorch Dataset for train set. Apply data augmentation if needed using "torchio"
    dataset_train = TrainDataset(images=train_images, labels=train_labels,
                                 transform=transforms.Compose([ToTensor_MRI()]),
                                 medical_augment=medical_augmentation_pt)
    del train_images, train_labels

    # Pytorch Dataset for validation set
    dataset_val = ValDataset(images=val_images, labels=val_labels,
                             transform=transforms.Compose([ToTensor_MRI()]))
    del val_images, val_labels

    # Pytorch Dataset for test set
    dataset_test = TestDataset(images=test_images, labels=test_labels,
                               transform=transforms.Compose([ToTensor_MRI()]))
    del test_images, test_labels

    return dataset_train, dataset_val, dataset_test, lim, input_shape, median_age


# ------------------- WAND Dataset ---------------------
def build_dataset_wand_t1w(args):
    """load t1w data"""
    # load data(in .npy format)
    images = np.load(os.path.join(args.data_dir, 'wand_t1w_cropped.npy'))
    age = np.load(os.path.join(args.data_dir, 'tract_age_compact.npy'))

    df = pd.DataFrame(data=age, columns=['Age'])
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

    # train-val & test split
    split = StratifiedShuffleSplit(test_size=args.test_size, random_state=args.random_state)
    train_val_index, test_index = next(split.split(df, df['Age_categorical']))
    stratified_train_val_set = df.loc[train_val_index]
    assert sorted(train_val_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # train & val split
    split2 = StratifiedShuffleSplit(test_size=args.val_size, random_state=args.random_state)
    train_index, val_index = next(split2.split(stratified_train_val_set, stratified_train_val_set['Age_categorical']))

    # NOTE: StratifiedShuffleSplit returns RangeIndex instead of the Original Index of the new DataFrame
    assert sorted(train_index.tolist() + val_index.tolist()) == list(range(len(stratified_train_val_set.index)))
    assert sorted(train_index.tolist() + val_index.tolist()) != sorted(list(stratified_train_val_set.index))

    # get the correct index of original DataFrame for train/val set
    train_index = train_val_index[train_index]
    val_index = train_val_index[val_index]

    # ensure there is no duplicated index
    assert sorted(train_index.tolist() + val_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # split data based on index
    train_images = images[train_index].astype(np.float32)
    val_images = images[val_index].astype(np.float32)
    test_images = images[test_index].astype(np.float32)
    input_shape = train_images.shape[1:]
    del images

    # add dimension for labels: (batch_size,) -> (batch_size, 1)
    train_labels = np.expand_dims(df.loc[train_index, 'Age'].values, axis=1).astype(np.float32)
    val_labels = np.expand_dims(df.loc[val_index, 'Age'].values, axis=1).astype(np.float32)
    test_labels = np.expand_dims(df.loc[test_index, 'Age'].values, axis=1).astype(np.float32)

    logger.info(f'Training images shape: {train_images.shape}, validation images shape: {val_images.shape}, '
                f'testing images shape: {test_images.shape}, training labels shape: {train_labels.shape}, '
                f'validation labels shape: {val_labels.shape}, testing labels shape: {test_labels.shape}')

    # Pytorch Dataset for train set. Apply data augmentation if needed using "torchio"
    dataset_train = TrainDataset(images=train_images, labels=train_labels,
                                 transform=transforms.Compose([ToTensor_MRI()]),
                                 medical_augment=medical_augmentation_pt)
    del train_images, train_labels

    # Pytorch Dataset for validation set
    dataset_val = ValDataset(images=val_images, labels=val_labels,
                             transform=transforms.Compose([ToTensor_MRI()]))
    del val_images, val_labels

    # Pytorch Dataset for test set
    dataset_test = TestDataset(images=test_images, labels=test_labels,
                               transform=transforms.Compose([ToTensor_MRI()]))
    del test_images, test_labels

    return dataset_train, dataset_val, dataset_test, lim, input_shape, median_age
