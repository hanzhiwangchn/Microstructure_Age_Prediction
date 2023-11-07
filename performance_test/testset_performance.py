import argparse
from test_utils import get_model_results, print_stats, calculate_mae, calculate_correlation, \
    mae_mean_std, corr_mean_std

# Test set MAE and correlation evaluation
IMAGE_MODALITY_LIST = ['KFA_DKI', 'FA_CHARMED', 'RD_CHARMED', 'MD_CHARMED', 'FRtot_CHARMED']


def build_parser_for_evaluation():
    parser = argparse.ArgumentParser(description='Microstructure Age Prediction Evaluation')
    parser.add_argument('--model', type=str, default='densenet', choices=['densenet', 'resnet'])
    parser.add_argument('--num-runs', type=int, default=6)
    return parser


def single_model_evaluation_for_each_modality():
    """A evaluation pipeline for each image modality"""
    args = build_parser_for_evaluation().parse_args()

    for image_modality in IMAGE_MODALITY_LIST:
        args.image_modality = image_modality

        print(f"{image_modality}")
        dfs = get_model_results(args)
        assert dfs[1][0][1]['ground_truth'].values.tolist() == dfs[2][0][3]['ground_truth'].values.tolist()

        corr_list = calculate_correlation(dfs)
        mae_list = calculate_mae(dfs)
        print_stats(mae_list, category='mae')
        print_stats(corr_list, category='correlation')

        # calculate mean and std for each type of loss
        mae_mean_std(mae_list)
        corr_mean_std(corr_list)


if __name__ == '__main__':
    single_model_evaluation_for_each_modality()
