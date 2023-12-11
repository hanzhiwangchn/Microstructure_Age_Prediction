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
    parser = argparse.ArgumentParser(description='build parser for tract training')
    # load and split data parameters
    parser.add_argument('--tract-data-dir', type=str, default='Datasets/tract_data', help='tract data dirs')
    parser.add_argument('--val-size', type=float, default=0.1, help='val set size')
    parser.add_argument('--test-size', type=float, default=0.1, help='test set size')
    parser.add_argument('--random-state', type=int, default=16, help='random state')
    # Decomposition parameters
    parser.add_argument('--decomposition', action='store_true', default=False)
    parser.add_argument('--decomposition-axis', type=str, default='d_measures', 
                        choices=['d_measures', 'tracts', 'both'], 
                        help='select on which axis to perform decomposition')
    parser.add_argument('--decomposition-method', type=str, default='pca', choices=['pca', 'umap'])
    parser.add_argument('--n-component', type=int, default=5, help='number of components for decomposition')
    # SMOTE parameters
    parser.add_argument('--smote', action='store_true', default=True)
    parser.add_argument('--smote-method', type=str, default='extreme', choices=['extreme', 'balance'],
                        help='sampling methods for SMOGN. In most cases, "extreme" is better.')
    parser.add_argument('--smote-threshold', type=float, default=0.5, help='threshold for SMOGN')
    parser.add_argument('--smote-label-control', action='store_true', default=True, 
                        help='control the range of synthetic label values for SMOGN')
    # less interesting parameters
    parser.add_argument('--eda', action='store_true', default=False)
    parser.add_argument('--derek-paper-plots', action='store_true', default=False)
    # default parameters
    parser.add_argument('--baseline-model-training', action='store_true', default=True)
    # test parameters
    parser.add_argument('--multi-runs-ensemble', action='store_true', default=False, 
                        help='ensemble same model from multiple runs for testing')
    # run time parameter
    parser.add_argument('--runtime', type=int, default=0, help='rum time indicator')
    return parser


def tract_training_main():
    """main pipeline for tract training"""
    # build parser
    args = build_parser_tract_training().parse_args()
    args = update_args(args=args)
    logger.info(f'Parser arguments are {args}')

    # load and split data
    train_features, val_features, test_features, train_labels, val_labels, test_labels = load_tract_data(args=args)
    logger.info('Dataset loaded')

    # Exploratory Data Analysis
    if args.eda:
        exploratory_data_analysis(args=args, train_features=train_features, train_labels=train_labels)

    # scale data
    train_features, val_features, test_features = scale_data(args=args, train_features=train_features, val_features=val_features, test_features=test_features)
    logger.info('Dataset scaled')

    # perform decomposition: (PCA or UMAP)
    if args.decomposition:
        train_features, val_features, test_features = apply_decomposition(args=args, train_features=train_features, val_features=val_features, test_features=test_features)
        logger.info('Dataset decomposed')
    else:
        logger.info('No decomposition')
    logger.info(f"Training feature shape: {train_features.shape}, val feature shape: {val_features.shape}, test feature shape: {test_features.shape}."
                f"Training label shape: {train_labels.shape}, val label shape: {val_labels.shape}, test label shape: {test_labels.shape}")

    # plot age vs. principle component 1 for each tract region (Derek's paper).
    if args.derek_paper_plots and args.decomposition == 'd_measures':
        paper_plots_derek(args=args, train_features=train_features, train_labels=train_labels)

    # Main Model Training
    if args.baseline_model_training:
        training_baseline_model(args=args, train_features=train_features, val_features=val_features, train_labels=train_labels, val_labels=val_labels)
    
    # It is a workaround to disentangle training and testing functions. 
    # trained_model_list needs to be changed accordingly if the actual applied models are changed.
    trained_model_list = ['svr', 'xgb', 'stack_reg_1']
    load_trained_model_ensemble(args=args, val_features=val_features, test_features=test_features, 
                                val_labels=val_labels, test_labels=test_labels, trained_model_list=trained_model_list)
          
    # ensemble the same model from different runtime. It should be used after training.
    if args.multi_runs_ensemble:
        ensemble_model_from_multiple_runs(args=args, val_features=val_features, test_features=test_features, 
                                          val_labels=val_labels, test_labels=test_labels, trained_model_list=trained_model_list)


if __name__ == '__main__':
    tract_training_main()
