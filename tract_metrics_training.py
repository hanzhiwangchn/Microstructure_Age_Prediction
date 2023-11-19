def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse, logging, os, json

from tract_training.tract_training_utils import *

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# create results folder
results_folder = 'model_ckpt_results'
os.makedirs(results_folder, exist_ok=True)


def build_parser_tract_training():
    parser = argparse.ArgumentParser(description='build parser for tract metrics training')
    parser.add_argument('--data-dir', type=str, default='tract_training/tract_data', help='data dirs')
    parser.add_argument('--test-size', type=float, default=0.1, help='test set size')
    parser.add_argument('--random-state', type=int, default=42, help='random state')
    # Decomposition
    parser.add_argument('--decomposition', action='store_true', default=True)
    parser.add_argument('--decomposition-feature-names', type=str, default='d_measures', choices=['d_measures', 'both'])
    parser.add_argument('--decomposition-method', type=str, default='pca', choices=['pca', 'umap', 'kernel_pca'])
    parser.add_argument('--pca-component', type=int, default=5, help='pca components')
    parser.add_argument('--pca-component-1', type=int, default=5, help='pca components')
    parser.add_argument('--umap-component', type=int, default=5, help='umap components')
    parser.add_argument('--kernel-pca-component', type=int, default=5, help='kernel pca components')
    # others
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
    # build parser
    args = build_parser_tract_training().parse_args()
    args.keyword_dict = build_keyword_dict()
    logger.info(args)

    # load and split data
    train_features, test_features, train_labels, test_labels = load_tract_data(args)

    # possible feature engineering should happen here

    # standardize data
    train_features, test_features = standardize_data(train_features, test_features)

    # perform decomposition (PCA or UMAP)
    if args.decomposition:
        train_features, test_features = apply_decomposition(args, train_features, test_features)

    # plot age vs. principle component 1 for each tract region (Derek paper)
    if args.derek_paper_plots:
        paper_plots_derek(args, train_features, train_labels)

    # for each tract region, we training a baseline model
    if args.tractwise_baseline_model_training:
        training_tractwise_baseline_model(args, train_features, test_features, train_labels, test_labels)

    # model averaging on all tracts

    # add SHAP plot for each baseline model
    pass

def test():
    with open('baseline_model_performance.json', 'r') as f:
        data = json.load(f)

    count_dict = dict()
    count_dict['xgb'] = 0
    count_dict['rfr'] = 0
    count_dict['kernelridge'] = 0
    count_dict['svr'] = 0
    count_dict['stack_reg'] = 0

    for each_tract in data.keys():
        for each_model in data[each_tract].keys():
            performance = float(data[each_tract][each_model].split('.')[0])
            if performance >= 9:
                count_dict[each_model] += 1
    print(count_dict)

    # print(data)
    pass

if __name__ == '__main__':
    tract_training_main()
