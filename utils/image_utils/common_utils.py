from collections import OrderedDict
import time, os, json, logging, torch

from scipy import stats
import pandas as pd
import torchio as tio

logger = logging.getLogger(__name__)
results_folder = 'model_ckpt_results/images'
WAND_NPY_micro_DATA_DIR = '/cubric/data/c1809127/314_wand_compact'
WAND_NPY_t1w_DATA_DIR = '../micro_images'
WAND_NPY_t1w_DATA_DIR = '/cubric/data/c1809127/314_wand_mri_preprocessed_npy'
# WAND_NPY_t1w_DATA_DIR = '/Users/hanzhiwang/Datasets'


# ------------------- Pytorch Dataset ---------------------
class TrainDataset(torch.utils.data.Dataset):
    """build training dataset"""
    def __init__(self, images, labels, transform=None, medical_augment=None):
        self.transform = transform
        self.medical_augment = medical_augment
        self.dict = []
        for i in range(len(images)):
            temp_dict = {'image': images[i], 'label': labels[i]}
            self.dict.append(temp_dict)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        image, label = self.dict[idx]['image'], self.dict[idx]['label']
        if self.transform:
            image, label = self.transform([image, label])
        if self.medical_augment:
            # medical augmentation can only be used on 3D medical images
            image = self.medical_augment(image)
        return {'image': image, 'label': label}


class ValDataset(torch.utils.data.Dataset):
    """
    build validation dataset
    Note that Huggingface requires that __getitem__ method returns dict object
    """
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.dict = []
        for i in range(len(images)):
            temp_dict = {'image': images[i], 'label': labels[i]}
            self.dict.append(temp_dict)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        image, label = self.dict[idx]['image'], self.dict[idx]['label']
        if self.transform:
            image, label = self.transform([image, label])
        return {'image': image, 'label': label}


class TestDataset(torch.utils.data.Dataset):
    """
    build test dataset
    Note that Huggingface requires that __getitem__ method returns dict object
    """
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.dict = []
        for i in range(len(images)):
            temp_dict = {'image': images[i], 'label': labels[i]}
            self.dict.append(temp_dict)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        image, label = self.dict[idx]['image'], self.dict[idx]['label']
        if self.transform:
            image, label = self.transform([image, label])
        return {'image': image, 'label': label}


class ToTensor_MRI(object):
    """Convert ndarrays in sample to Tensors for MRI"""
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        return torch.from_numpy(image), torch.from_numpy(label)


def medical_augmentation_pt(images):
    training_transform = tio.Compose([
        # tio.RandomNoise(p=0.5),  # Gaussian noise 50% of times
        tio.RandomFlip(flip_probability=0.5),
    ])
    return training_transform(images)


# ------------------- Run Manager ---------------------
class RunManager:
    """capture model stats"""
    def __init__(self, args):
        self.epoch_num_count = 0
        self.epoch_start_time = None
        self.args = args

        # train/validation/test metrics
        self.train_epoch_loss = 0

        self.run_correlation_train = []
        self.run_correlation_val = []
        self.run_correlation_test = []

        self.run_metrics_train = None
        self.run_metrics_val = None
        self.run_metrics_test = None

        # run data saves the stats from train/validation/test set
        self.run_data = []
        self.run_start_time = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.epoch_stats = None

    def begin_run(self, train_loader, val_loader, test_loader):
        self.run_start_time = time.time()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        logger.info('Begin Run!')

    def end_run(self):
        """save metrics from test set"""
        self.epoch_num_count = 0
        logger.info('End Run!')

    def begin_epoch(self):
        self.epoch_num_count += 1
        self.epoch_start_time = time.time()
        # initialize metrics
        self.train_epoch_loss = 0
        logger.info(f'Start epoch {self.epoch_num_count}')

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        # calculate loss
        train_loss = self.train_epoch_loss / len(self.train_loader.dataset)
        logger.info(f'End epoch {self.epoch_num_count}')

        # add stats from current epoch to run data
        self.epoch_stats = OrderedDict()
        self.epoch_stats['epoch'] = self.epoch_num_count
        self.epoch_stats['train_loss'] = float(f'{train_loss:.2f}')
        self.epoch_stats.update(self.run_metrics_train)
        self.epoch_stats.update(self.run_metrics_val)
        self.epoch_stats.update(self.run_metrics_test)
        self.epoch_stats['train_correlation'] = float(f'{self.run_correlation_train[-1]:.2f}')
        self.epoch_stats['val_correlation'] = float(f'{self.run_correlation_val[-1]:.2f}')
        self.epoch_stats['test_correlation'] = float(f'{self.run_correlation_test[-1]:.2f}')
        self.epoch_stats['epoch_duration'] = float(f'{epoch_duration:.1f}')
        self.epoch_stats['run_duration'] = float(f'{run_duration:.1f}')
        self.run_data.append(self.epoch_stats)

    def track_train_loss(self, loss):
        # accumulate training loss for all batches
        self.train_epoch_loss += loss.item() * self.train_loader.batch_size

    def collect_train_metrics(self, metric_results):
        self.run_metrics_train = {f'train_{k}': round(v, 3) for k, v in metric_results.items()}

    def collect_val_metrics(self, metric_results):
        self.run_metrics_val = {f'val_{k}': round(v, 3) for k, v in metric_results.items()}
        
    def collect_test_metrics(self, metric_results):
        self.run_metrics_test = {f'test_{k}': round(v, 3) for k, v in metric_results.items()}

    def collect_train_correlation(self, correlation):
        self.run_correlation_train.append(correlation)

    def collect_val_correlation(self, correlation):
        self.run_correlation_val.append(correlation)

    def collect_test_correlation(self, correlation):
        self.run_correlation_test.append(correlation)

    def display_epoch_results(self):
        # display stats from the current epoch
        logger.info(self.epoch_stats)

    def save(self, filename):
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{filename}.csv')


# ------------------- model functions ---------------------
def update_args(args):
    """update arguments"""
    # use CUDA if possible
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Found device: {args.device}')

    if args.dataset == 'wand_micro':
        args.data_dir = os.path.join(WAND_NPY_micro_DATA_DIR, args.image_modality)
        args.num_train_epochs = 200
        args.batch_size = 4
        args.update_lambda_start_epoch = 50
        args.update_lambda_second_phase_start_epoch = 100
        args.save_best_start_epoch = 10

    elif args.dataset == 'wand_t1w':
        args.data_dir = os.path.join(WAND_NPY_t1w_DATA_DIR)
        args.num_train_epochs = 200
        args.batch_size = 4
        args.update_lambda_start_epoch = 50
        args.update_lambda_second_phase_start_epoch = 100
        args.save_best_start_epoch = 10

    # quick pipeline check setting
    if args.run_code_test:
        # training
        args.data_dir = WAND_NPY_t1w_DATA_DIR
        args.num_train_epochs = 10
        args.batch_size = 32
        args.update_lambda_start_epoch = 10
        args.update_lambda_second_phase_start_epoch = 20
        args.save_best_start_epoch = 1
        args.comment = 'test_run'
        args.acceptable_correlation_threshold = 0.99

    args.out_dir_main = results_folder
    # model name using a single image modality
    args.model_name = f'{args.model}_loss_{args.loss_type}_skewed_{args.skewed_loss}_modality_{args.image_modality}_' \
                      f'{args.comment}_rnd_state_{args.random_state}'
    args.out_dir = f'{args.out_dir_main}/{args.model_name}'
    os.makedirs(args.out_dir, exist_ok=True)
    return args


def calculate_correlation(args, model, m, train_loader, val_loader, test_loader):
    """calculate correlation after current epoch"""
    train_preds_list = []
    train_labels_list = []
    val_preds_list = []
    val_labels_list = []
    test_preds_list = []
    test_labels_list = []

    model.eval()
    # training set
    with torch.no_grad():
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(batch['image'])

            assert outputs.shape == batch['label'].shape
            assert len(outputs.shape) == 2

            train_preds_list.append(outputs)
            train_labels_list.append(batch['label'])

        # preds and labels will have shape (*, 1)
        preds = torch.cat(train_preds_list, 0)
        labels = torch.cat(train_labels_list, 0)
        assert preds.shape == labels.shape
        assert preds.shape[1] == 1

        correlation = calculate_correlation_coefficient(preds=preds, labels=labels, args=args)
        # track train correlation
        m.collect_train_correlation(correlation=correlation.item())

    # validation set
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(batch['image'])

            assert outputs.shape == batch['label'].shape
            assert len(outputs.shape) == 2

            val_preds_list.append(outputs)
            val_labels_list.append(batch['label'])

        # preds and labels will have shape (*, 1)
        preds = torch.cat(val_preds_list, 0)
        labels = torch.cat(val_labels_list, 0)
        assert preds.shape == labels.shape
        assert preds.shape[1] == 1

        correlation = calculate_correlation_coefficient(preds=preds, labels=labels, args=args)
        # track train correlation
        m.collect_val_correlation(correlation=correlation.item())

    # test set
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(batch['image'])

            assert outputs.shape == batch['label'].shape
            assert len(outputs.shape) == 2

            test_preds_list.append(outputs)
            test_labels_list.append(batch['label'])

        # preds and labels will have shape (*, 1)
        preds = torch.cat(train_preds_list, 0)
        labels = torch.cat(train_labels_list, 0)
        assert preds.shape == labels.shape
        assert preds.shape[1] == 1

        correlation = calculate_correlation_coefficient(preds=preds, labels=labels, args=args)
        # track test correlation
        m.collect_test_correlation(correlation=correlation.item())


def calculate_correlation_coefficient(preds, labels, args):
    """calculate correlation coefficient"""
    if args.correlation_type == 'pearson':
        error = preds - labels
        vx = error - torch.mean(error)
        vy = labels - torch.mean(labels)
        corr_coef = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    elif args.correlation_type == 'spearman':
        error = preds - labels
        corr_coef = stats.spearmanr(error.cpu(), labels.cpu())[0]
    return corr_coef
