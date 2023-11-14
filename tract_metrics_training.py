# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

import argparse, logging, os

from utils.build_dataset import load_tract_data
from utils.tract_training_utils import apply_pca, paper_plots_derek, training_tractwise_baseline_model

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# create results folder
results_folder = 'model_ckpt_results'
os.makedirs(results_folder, exist_ok=True)


def build_parser_tract_training():
    parser = argparse.ArgumentParser(description='build parser for tract metrics training')
    parser.add_argument('--data-dir', type=str, default='tract_data', help='data dirs')
    parser.add_argument('--test-size', type=float, default=0.1, help='test set size')
    parser.add_argument('--random-state', type=int, default=42, help='random state')
    parser.add_argument('--pca', action='store_true', default=True)
    parser.add_argument('--derek-paper-plots', action='store_true', default=False)
    parser.add_argument('--tractwise-baseline-model-training', action='store_true', default=True)
    return parser


def build_keyword_dict():
    keyword_dict = dict()
    # tract of interest
    keyword_dict['ROI'] = ['AF_left', 'AF_right', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 
                           'CG_left', 'CG_right', 'CST_left', 'CST_right', 'FX_left', 'FX_right', 'IFO_left', 
                           'IFO_right', 'ILF_left', 'ILF_right', 'SLF_III_left', 'SLF_III_right', 'SLF_II_left', 
                           'SLF_II_right', 'SLF_I_left', 'SLF_I_right', 'T_PREF_left', 'T_PREF_right', 'UF_left', 
                           'UF_right']
    # diffusion measures
    keyword_dict['d_measures'] = ['KFA_DKI', 'ICVF_NODDI', 'AD_CHARMED', 'FA_CHARMED', 'RD_CHARMED', 
                                  'MD_CHARMED', 'FRtot_CHARMED', 'MWF_mcDESPOT']                   
    return keyword_dict


def tract_training_main():
    """main pipeline for tract training"""
    args = build_parser_tract_training().parse_args()
    keyword_dict = build_keyword_dict()

    # load and standardize data
    train_features, test_features, train_labels, test_labels = load_tract_data(args)

    # perform PCA
    if args.pca:
        train_features, test_features = apply_pca(train_features, test_features, keyword_dict)

    # plot age vs. principle component 1 for each tract region (Derek paper)
    if args.derek_paper_plots:
        paper_plots_derek(train_features, train_labels, keyword_dict)

    # for each tract region, we training a baseline model
    if args.tractwise_baseline_model_training:
        training_tractwise_baseline_model(train_features, test_features, train_labels, test_labels, keyword_dict)

    pass


if __name__ == '__main__':
    tract_training_main()
