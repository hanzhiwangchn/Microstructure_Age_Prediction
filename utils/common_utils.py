from collections import OrderedDict
import time, os, json ,logging, torch

from scipy import stats
import pandas as pd

logger = logging.getLogger(__name__)
results_folder = 'model_ckpt_results'


# ------------------- Pytorch Dataset ---------------------
class TrainDataset(torch.utils.data.Dataset):
    """
    build training dataset
    Note that Huggingface requires that __getitem__ method returns dict object
    """
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


class ValidationDataset(torch.utils.data.Dataset):
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


# ------------------- Run Manager for no Trainer training ---------------------
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

    def end_run(self, dirs):
        """save metrics from test set"""
        test_results = {f"eval_{k}": v for k, v in self.run_metrics_test.items()}
        with open(os.path.join(dirs, "test_results.json"), "w") as f:
            json.dump(test_results, f)
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

    # original setting for camcan dataset
    if args.dataset == 'camcan':
        args.data_dir = '../shared/camcan/mri_concat.pickle'
        args.num_train_epochs = 400
        args.batch_size = 32
        args.update_lambda_start_epoch = 150
        args.update_lambda_second_phase_start_epoch = 250
        args.save_best_start_epoch = 100

    # quick pipeline check setting
    if args.run_code_test:
        # training
        args.model = 'resnet'
        args.num_train_epochs = 10
        args.batch_size = 32
        args.update_lambda_start_epoch = 2
        args.update_lambda_second_phase_start_epoch = 4
        args.save_best_start_epoch = 1
        args.comment = 'test_run'
        # dataset
        args.val_test_size = 0.8
        args.test_size = 0.5

    args.out_dir = results_folder
    args.model_name_no_trainer = args.model+ f'-pt-{args.comment}' 
    args.out_dir_no_trainer = f'{args.out_dir}/{args.model_name_no_trainer}'
    os.makedirs(args.out_dir_no_trainer, exist_ok=True)
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