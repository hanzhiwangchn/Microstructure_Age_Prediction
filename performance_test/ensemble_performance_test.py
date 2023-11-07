import argparse
import numpy as np
from test_utils import get_model_results, print_stats

# ensemble model test using same train/test split
IMAGE_MODALITY_LIST = ['KFA_DKI', 'FA_CHARMED', 'RD_CHARMED', 'MD_CHARMED', 'FRtot_CHARMED']
LOSS_TYPE_LIST = ['L1_normal', 'L1_normal_corrected', 'L1_skewed']


def build_parser_for_ensemble_evaluation():
    parser = argparse.ArgumentParser(description='Microstructure Age Prediction Evaluation')
    parser.add_argument('--model', type=str, default='densenet', choices=['densenet', 'resnet'])
    parser.add_argument('--num-runs', type=int, default=6)
    return parser


def ensemble_evaluation_for_each_modality():
    """A evaluation pipeline for each image modality"""
    args = build_parser_for_ensemble_evaluation().parse_args()

    for image_modality in IMAGE_MODALITY_LIST:
        args.image_modality = image_modality

        mae_list = [[] for _ in range(len(LOSS_TYPE_LIST))]
        corr_list = [[] for _ in range(len(LOSS_TYPE_LIST))]

        print(f"{image_modality}")
        dfs = get_model_results(args)
        assert dfs[1][0][1]['ground_truth'].values.tolist() == dfs[2][0][3]['ground_truth'].values.tolist()

        mae_list, corr_list = ensemble_predictions(dfs, mae_list, corr_list, args)
        print_stats(mae_list, category='mae')
        print_stats(corr_list, category='correlation')


def ensemble_predictions(dfs, mae_list, corr_list, args):
    for i in range(len(LOSS_TYPE_LIST)):
        for j in range(len(dfs[i])):
            df0 = dfs[i][j][0].copy(deep=True)
            for k in range(1, args.num_runs):
                df0[f'predicted_value_{k}'] = dfs[i][j][k]['predicted_value']

            df0['ensemble_mean'] = df0[['predicted_value'] + [f'predicted_value_{m}' for m in range(1, args.num_runs)]].\
                mean(axis=1)
            df0['ensemble_mean_std'] = df0[['predicted_value'] + [f'predicted_value_{n}' for n in range(1, args.num_runs)]].\
                std(axis=1)
            df0['diff'] = df0['ensemble_mean'] - df0['ground_truth']
            df0['diff_abs'] = df0['diff'].abs()

            mae_list[i].append(df0['diff_abs'].mean())
            corr_list[i].append(np.corrcoef(df0['ground_truth'], df0['diff'])[0][1])

    return mae_list, corr_list


if __name__ == '__main__':
    ensemble_evaluation_for_each_modality()
