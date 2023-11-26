def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse, logging, os
from tract_training.tract_training_utils import *

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# create results folder
results_folder = 'tract_training/model_ckpt_results'
os.makedirs(results_folder, exist_ok=True)


def build_parser_tract_training():
    parser = argparse.ArgumentParser(description='build parser for tract metrics training')
    # default parameters
    parser.add_argument('--data-dir', type=str, default='tract_training/tract_data', help='data dirs')
    parser.add_argument('--test-size', type=float, default=0.1, help='test set size')
    parser.add_argument('--random-state', type=int, default=5, help='random state')
    # Decomposition parameters
    parser.add_argument('--decomposition', action='store_true', default=True)
    parser.add_argument('--decomposition-feature-names', type=str, default='d_measures', 
                        choices=['d_measures', 'tracts', 'both'])
    parser.add_argument('--decomposition-method', type=str, default='pca', choices=['pca', 'umap', 'kernel_pca'])
    parser.add_argument('--pca-component', type=int, default=5, help='pca components')
    parser.add_argument('--umap-component', type=int, default=5, help='umap components')
    parser.add_argument('--kernel-pca-component', type=int, default=5, help='kernel pca components')
    # other parameters
    parser.add_argument('--derek-paper-plots', action='store_true', default=False)
    parser.add_argument('--tractwise-baseline-model-training', action='store_true', default=True)
    # test parameters
    parser.add_argument('--eda', action='store_true', default=True)
    parser.add_argument('--smote', action='store_true', default=False)
    parser.add_argument('--outlier-removal', action='store_true', default=True)
    parser.add_argument('--scatter-prediction-plot', action='store_true', default=False)
    parser.add_argument('--to-categorical-label', action='store_true', default=False)
    return parser


def tract_training_main():
    """main pipeline for tract training"""
    # build parser
    args = build_parser_tract_training().parse_args()
    args.keyword_dict = build_keyword_dict()
    logger.info(f'Parser arguments are {args}')

    # load and split data
    train_features, test_features, train_labels, test_labels = load_tract_data(args)
    logger.info('Dataset loaded')

    # Exploratory Data Analysis
    if args.eda:
        perform_exploratory_data_analysis(args, train_features, train_labels)

    # possible feature engineering should happen here

    # scale data
    train_features, test_features = scale_data(train_features, test_features)
    logger.info('Dataset scaled')

    # perform decomposition: (PCA, UMAP or Kernel PCA)
    if args.decomposition:
        train_features, test_features = apply_decomposition(args, train_features, test_features)
    logger.info('Dataset decomposed')
    logger.info(f"Training feature shape: {train_features.shape}, test feature shape: {test_features.shape}. "
                f"Training label shape: {train_labels.shape}, test label shape: {test_labels.shape}")

    # plot age vs. principle component 1 for each tract region (Derek's paper)
    if args.derek_paper_plots and args.decomposition == 'd_measures':
        paper_plots_derek(args, train_features, train_labels)

    # When decomposition method is 'd_measures' and 'both', tract-wise baseline is trained.
    # When decomposition method is 'tracts', d_measure-wise baseline is trained.
    if args.tractwise_baseline_model_training:
        res_dict = training_tractwise_baseline_model(args, train_features, test_features, train_labels, test_labels)
        load_trained_model_ensemble(args, train_features, test_features, test_labels, res_dict)

    # add SHAP plot for each baseline model
          

if __name__ == '__main__':
    tract_training_main()
