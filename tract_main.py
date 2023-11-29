def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse, logging, os
from utils.tract_utils.tract_training_utils import *

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# create results folder
results_folder = 'model_ckpt_results/tracts'
os.makedirs(results_folder, exist_ok=True)


def build_parser_tract_training():
    parser = argparse.ArgumentParser(description='build parser for tract metrics training')
    # load and split data parameters
    parser.add_argument('--tract-data-dir', type=str, default='tract_data/temp', help='tract data dirs')
    parser.add_argument('--val-size', type=float, default=0.1, help='val set size')
    parser.add_argument('--test-size', type=float, default=0.1, help='test set size')
    parser.add_argument('--random-state', type=int, default=0, help='random state')
    # Decomposition parameters
    parser.add_argument('--decomposition', action='store_true', default=False)
    parser.add_argument('--decomposition-feature-names', type=str, default='d_measures', 
                        choices=['d_measures', 'tracts', 'both'], 
                        help='select on which axis to perform decomposition')
    parser.add_argument('--decomposition-method', type=str, default='pca', choices=['pca', 'umap', 'kernel_pca'])
    parser.add_argument('--pca-component', type=int, default=5, help='pca components')
    parser.add_argument('--umap-component', type=int, default=5, help='umap components')
    parser.add_argument('--kernel-pca-component', type=int, default=5, help='kernel pca components')
    # SMOTE parameters
    parser.add_argument('--smote', action='store_true', default=True)
    parser.add_argument('--smote-method', type=str, default='extreme', choices=['extreme', 'balance'],
                        help='sampling methods for SMOGN. In most cases, "extreme" is better.')
    parser.add_argument('--smote-threshold', type=float, default=0.5, help='threshold for SMOGN')
    # less interesting parameters
    parser.add_argument('--eda', action='store_true', default=False)
    parser.add_argument('--derek-paper-plots', action='store_true', default=False)
    parser.add_argument('--outlier-removal', action='store_true', default=False, 
                        help='remove subjects whose age is larger than 60. It becomes less interesting after we use SMOTE')
    # default parameters
    parser.add_argument('--baseline-model-training', action='store_true', default=True)
    parser.add_argument('--scatter-prediction-plot', action='store_true', default=True)
    # test parameters
    parser.add_argument('--test-only', action='store_true', default=False, help='load trained models and do testing')
    return parser


def tract_training_main():
    """main pipeline for tract training"""
    # build parser
    args = build_parser_tract_training().parse_args()
    args = update_args(args=args)
    logger.info(f'Parser arguments are {args}')

    # load and split data
    train_features, val_features, test_features, train_labels, val_labels, test_labels = load_tract_data(args)
    logger.info('Dataset loaded')

    # Exploratory Data Analysis
    # TODO: check it later
    if args.eda:
        perform_exploratory_data_analysis(args, train_features, train_labels)

    # possible feature engineering should happen here

    # scale data
    train_features, val_features, test_features = scale_data(train_features, val_features, test_features)
    logger.info('Dataset scaled')

    # perform decomposition: (PCA, UMAP or Kernel PCA)
    if args.decomposition:
        train_features, val_features, test_features = apply_decomposition(args, train_features, val_features, test_features)
    logger.info('Dataset decomposed')
    logger.info(f"Training feature shape: {train_features.shape}, val feature shape: {val_features.shape}, test feature shape: {test_features.shape}."
                f"Training label shape: {train_labels.shape}, val label shape: {val_labels.shape}, test label shape: {test_labels.shape}")

    # plot age vs. principle component 1 for each tract region (Derek's paper)
    # TODO: check it later
    if args.derek_paper_plots and args.decomposition == 'd_measures':
        paper_plots_derek(args, train_features, train_labels)

    # When decomposition method is 'd_measures' and 'both', tract-wise baseline is trained.
    # When decomposition method is 'tracts', d_measure-wise baseline is trained.
    if args.baseline_model_training:
        args = training_baseline_model(args, train_features, val_features, train_labels, val_labels)
        load_trained_model_ensemble(args, train_features, val_features, test_features, val_labels, test_labels)
          

if __name__ == '__main__':
    tract_training_main()
