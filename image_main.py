import argparse, logging, os

from utils.image_utils.common_utils import RunManager, update_args
from utils.image_utils.build_dataset import build_dataset
from utils.image_utils.build_model import build_model
from utils.image_utils.build_loss_function import build_loss_function
from utils.image_utils.build_training_loop import build_loader, build_optimizer, train_val_test

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# create results folder
results_folder = 'model_ckpt_results/images'
os.makedirs(results_folder, exist_ok=True)


def build_parser_image_training():
    """build parser for training models using microstructure and t1w images"""
    parser = argparse.ArgumentParser(description='build parser for image training')
    parser.add_argument('--model', type=str, default='densenet', choices=['densenet', 'resnet'],
                        help='model configurations')
    parser.add_argument('--loss-type', type=str, default='L1', choices=['L1'],
                        help='ordinary loss function configurations')
    parser.add_argument('--correlation-type', type=str, default='pearson', choices=['pearson', 'spearman'],
                        help='correlation metric configurations')
    parser.add_argument('--skewed-loss', action='store_true', default=False,
                        help='use skewed loss function')
    parser.add_argument('--run-code-test', action='store_true', default=False,
                        help='run code test')
    # dynamic lambda strategy config
    parser.add_argument('--compact-dynamic', action='store_true', default=False,
                        help='a compact dynamic-lambda algorithm for the skewed loss')
    parser.add_argument('--compact-target', type=str, default='validation', choices=['validation'],
                        help='compact dynamic-lambda config: '
                             'specify on which dataset we want the correlation to move toward zero')
    parser.add_argument('--compact-update-interval', type=int, default=5,
                        help='compact dynamic-lambda config: '
                             'update lambda value every a certain number of epoch')
    parser.add_argument('--compact-init-multiplier', type=float, default=1.4,
                        help='compact dynamic-lambda config: '
                             'initialize a multiplier in the stage-2 when updating lambda')
    # apply the two-stage bias correction algorithm
    parser.add_argument('--two-stage-correction', action='store_true', default=False,
                        help='use the two-stage correction approach for the normal loss')
    # frequently used settings
    parser.add_argument('--dataset', type=str, default='wand_t1w', choices=['wand_micro', 'wand_t1w'],
                        help='specify which dataset to use')
    parser.add_argument('--image-modality', type=str, default='t1w',
                        choices=['KFA_DKI', 'ICVF_NODDI', 'FA_CHARMED', 'RD_CHARMED', 'MD_CHARMED',
                                 'AD_CHARMED', 'FRtot_CHARMED', 'MWF_mcDESPOT', 't1w'],
                        help='specify which dataset to use')
    parser.add_argument('--random-state', type=int, default=0,
                        help='used in train test dataset split')
    parser.add_argument('--comment', type=str, default='run0',
                        help='comments to distinguish different runs')
    # default settings
    parser.add_argument('--val-size', type=float, default=0.1)
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--init-lambda', type=float, default=1.0,
                        help='default lambda value for the skewed loss')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-train-epochs', type=int, default=20, help='number of epoch')
    parser.add_argument('--params-init', type=str, default='kaiming_uniform',
                        choices=['default', 'kaiming_uniform', 'kaiming_normal'],
                        help='weight initializations')
    parser.add_argument('--acceptable-correlation-threshold', type=float, default=0.1,
                        help='acceptable threshold for correlation when selecting best model')
    parser.add_argument('--save-best', action='store_true', default=True,
                        help='save models with the lowest validation loss in training to prevent over-fitting')
    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--lr-scheduler-type', type=str, default='linear', help='lr scheduler type')
    parser.add_argument('--num-warmup-steps', type=int, default=0, help='num warmup steps')
    # testing
    parser.add_argument('--test', action='store_true', default=False,
                        help='testing')
    return parser


def baseline_model_training_images():
    """main pipeline for image training"""
    # build parser
    args = build_parser_image_training().parse_args()
    args = update_args(args=args)
    logger.info(f'Parser arguments are {args}')

    # build dataset
    dataset_train, dataset_val, dataset_test, args = build_dataset(args=args)
    logger.info('Dataset loaded')

    # build model
    model = build_model(args=args)
    logger.info('Model loaded')

    # build loss function
    loss_fn_train, _, _ = build_loss_function(args=args)

    # build dataloader
    train_loader, val_loader, test_loader = build_loader(args=args, dataset_train=dataset_train,
                                                         dataset_val=dataset_val, dataset_test=dataset_test)

    # build optimizer
    optimizer, lr_scheduler = build_optimizer(model=model, train_loader=train_loader, args=args)

    # build RunManager to save stats from training
    m = RunManager(args=args)
    m.begin_run(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    # train and evaluate
    train_val_test(args=args, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                   model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, m=m, loss_fn_train=loss_fn_train)

    logger.info('Model finished!')


if __name__ == '__main__':
    baseline_model_training_images()
