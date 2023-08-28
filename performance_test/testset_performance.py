import os, argparse
import numpy as np
import pandas as pd
from scipy import stats
import itertools
from matplotlib import pyplot as plt

# Test set MAE and correlation evaluation

LOSS_TYPE_LIST = ['L1_normal', 'L1_normal_corrected']
model_config = ['resnet']
num_runs = 6
dataset = 'abide_symmetric'
RESULT_DIRS = '~/model_ckpt_results/model_ckpt_results'
save_plot_dir = '../Plots/temp'

# f'{args.model}_loss_{args.loss_type}_skewed_{args.skewed_loss}_' \
#                       f'modality_{args.image_modality}_dataset_{args.image_modality}_' \
#                       f'{args.comment}_rnd_state_{args.random_state}'
def build_wand_image_modality_dir_dict():
    """build a dict with keys being image modality and values being corresponding dirs"""
    image_modality_dir_dict = dict()
    image_modality_dir_dict['KFA_DKI'] = 'derivatives/DKI_dipywithgradcorr/preprocessed/'
    # image_modality_dir_dict['ICVF_NODDI'] = 'derivatives/NODDI_MDT/preprocessed/'
    image_modality_dir_dict['FA_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_modality_dir_dict['RD_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_modality_dir_dict['MD_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    # image_modality_dir_dict['AD_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_modality_dir_dict['FRtot_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    # image_modality_dir_dict['MWF_mcDESPOT'] = 'derivatives/mcDESPOT/preprocessed/'

    return image_modality_dir_dict


def build_parser_for_evaluation():
    parser = argparse.ArgumentParser(description='Microstructure Age Prediction Evaluation')
    parser.add_argument('--model', type=str, default='resnet', choices=['densenet', 'resnet'])
    parser.add_argument('--num-runs', type=int, default=6)
    return parser

def get_model_results(args):
    """read csv files for all models and all runs"""
    dfs = []
    for loss_type in LOSS_TYPE_LIST:
        dfs_temp = []
        for random_state in range(1000, 1001):
            dfs_temp_2 = []
            for i in range(args.num_runs):
                if loss_type == 'L1_normal':
                    model_name = f'{args.model}_loss_L1_skewed_False_' \
                                 f'modality_{args.image_modality}_dataset_{args.image_modality}_' \
                                 f'run{i}_rnd_state_{random_state}'
                    df_temp = pd.read_csv(
                        os.path.join(RESULT_DIRS, model_name, 'performance_summary.csv'))
                elif loss_type == 'L1_skewed':
                    model_name = f'{args.model}_loss_L1_skewed_True_' \
                                 f'modality_{args.image_modality}_dataset_{args.image_modality}_' \
                                 f'run{i}_rnd_state_{random_state}'
                    df_temp = pd.read_csv(
                        os.path.join(RESULT_DIRS, model_name, 'performance_summary.csv'))
                elif loss_type == 'L1_normal_corrected':
                    model_name = f'{args.model}_loss_L1_skewed_False_' \
                                 f'modality_{args.image_modality}_dataset_{args.image_modality}_' \
                                 f'run{i}_rnd_state_{random_state}'
                    df_temp = pd.read_csv(
                        os.path.join(RESULT_DIRS, model_name, 'corrected_performance_summary.csv'))

                df_temp['error'] = df_temp['predicted_value'] - df_temp['ground_truth']
                df_temp['abs_error'] = df_temp['error'].abs()
                dfs_temp_2.append(df_temp)
            dfs_temp.append(dfs_temp_2)
        dfs.append(dfs_temp)
    return dfs


def calculate_correlation(dfs):
    corr_list = []
    for i in range(len(dfs)):
        temp = []
        for j in range(len(dfs[i])):
            temp_inner = []
            for k in range(len(dfs[i][j])):
                # corr = float(f"{stats.spearmanr(dfs[i][j][k]['ground_truth'], dfs[i][j][k]['error'])[0]:.2f}")
                corr = float(f"{np.corrcoef(dfs[i][j][k]['ground_truth'], dfs[i][j][k]['error'])[0][1]:.2f}")
                temp_inner.append(corr)
            temp.append(temp_inner)
        corr_list.append(temp)
    assert len(corr_list) == len(dfs)
    return corr_list


def calculate_mae(dfs):
    mae_list = []
    for i in range(len(dfs)):
        temp = []
        for j in range(len(dfs[i])):
            temp_inner = []
            for k in range(len(dfs[i][j])):
                mae = float(f"{dfs[i][j][k]['abs_error'].mean():.2f}")
                temp_inner.append(mae)
            temp.append(temp_inner)
        mae_list.append(temp)
    assert len(mae_list) == len(dfs)
    return mae_list


def print_stats(stats_list, category='mae'):
    print()
    if category == 'mae':
        print('MAE summary')
    elif category == 'correlation':
        print("Correlation summary")
    for i in range(len(LOSS_TYPE_LIST)):
        print()
        print(LOSS_TYPE_LIST[i])
        print(stats_list[i])


def analysis_stats(metric, analysis_metric, stats_list, desired_pairs, method):
    # stats_list here is a 3 dimensional array:loss_type, random_state, runs
    stats_list = np.array(stats_list)
    print()
    print(f'{metric} {analysis_metric} test')
    if method == 'individual':
        # 100 models test
        stats_list = stats_list.reshape(stats_list.shape[0], -1)
    elif method == 'average':
        stats_list = np.mean(stats_list, axis=2)
        print(stats_list)
    elif method == 'ensemble':
        # ensemble test is implemented in the ensemble_performance_test.py
        print('pls check the method')
        exit(0)
    for each_pair in desired_pairs:
        print()
        print(f'{LOSS_TYPE_LIST[each_pair[0]]} & {LOSS_TYPE_LIST[each_pair[1]]}')
        if analysis_metric == 'wilcoxon':
            print(stats.wilcoxon(stats_list[each_pair[0]], stats_list[each_pair[1]]))
        elif analysis_metric == 'ttest':
            print(stats.ttest_rel(stats_list[each_pair[0]], stats_list[each_pair[1]]))


def mae_mean_std(mae_list):
    print()
    mae_list = np.array(mae_list)
    mae_list = mae_list.reshape(mae_list.shape[0], -1)
    means = mae_list.mean(axis=1)
    stds = mae_list.std(axis=1)
    assert means.shape == (len(LOSS_TYPE_LIST), )
    for i in range(len(LOSS_TYPE_LIST)):
        print(f'{LOSS_TYPE_LIST[i]}: {means[i]}, {stds[i]}')


def corr_mean_std(corr_list):
    print()
    corr_list = np.array(corr_list)
    corr_list = corr_list.reshape(corr_list.shape[0], -1)
    means = corr_list.mean(axis=1)
    stds = corr_list.std(axis=1)
    assert means.shape == (len(LOSS_TYPE_LIST), )
    for i in range(len(LOSS_TYPE_LIST)):
        print(f'{LOSS_TYPE_LIST[i]}: {means[i]}, {stds[i]}')


def get_comparison_plots(mae_list, corr_list):
    plt.figure()
    for each_loss_idx in range(len(LOSS_TYPE_LIST)):
        plt.scatter(list(range(1, 1 + num_runs)), mae_list[each_loss_idx], label=f'{LOSS_TYPE_LIST[each_loss_idx]}')
    plt.xlabel('times', fontsize=20)
    plt.ylabel('MAE', fontsize=20)
    plt.legend(prop={'size': 10})
    plt.savefig(os.path.join(save_plot_dir,
                             f'{"MAE_comparison.jpg"}'),
                bbox_inches='tight')
    plt.close()
    plt.figure()
    for each_loss_idx in range(len(LOSS_TYPE_LIST)):
        plt.scatter(list(range(1, 1 + num_runs)), corr_list[each_loss_idx], label=f'{LOSS_TYPE_LIST[each_loss_idx]}')
    plt.xlabel('times', fontsize=20)
    plt.ylabel('correlation', fontsize=20)
    plt.hlines(y=0, xmin=1, xmax=5, colors='y')
    plt.legend(prop={'size': 10})
    plt.savefig(os.path.join(save_plot_dir,
                             f'{"corr_comparison.jpg"}'),
                bbox_inches='tight')
    plt.close()


def evaluation_for_each_modality():
    """A evaluation pipeline for each image modality"""
    args = build_parser_for_evaluation().parse_args()

    image_modality_list = list(build_wand_image_modality_dir_dict().keys())
    for image_modality in image_modality_list:
        args.image_modality = image_modality
        # read all csv files to collect results

        print(f"{image_modality}")
        dfs = get_model_results(args)

        # assert dfs[0][7]['ground_truth'].values.tolist() == dfs[1][7]['ground_truth'].values.tolist()
        # assert dfs[1][7]['ground_truth'].values.tolist() != dfs[2][4]['ground_truth'].values.tolist()
        assert len(dfs) == len(LOSS_TYPE_LIST)
        # assert len(dfs[-1]) == num_runs

        corr_list = calculate_correlation(dfs)
        mae_list = calculate_mae(dfs)
        print_stats(mae_list, category='mae')
        print_stats(corr_list, category='correlation')

        desired_pairs = [(0, 1)]
        desired_pairs = [list(itertools.combinations(i, 2)) for i in desired_pairs]
        desired_pairs = [j for i in desired_pairs for j in i]

        method = 'average'
        analysis_stats(metric='MAE', analysis_metric='wilcoxon',
                    stats_list=mae_list, desired_pairs=desired_pairs, method=method)
        analysis_stats(metric='MAE', analysis_metric='ttest',
                    stats_list=mae_list, desired_pairs=desired_pairs, method=method)
        analysis_stats(metric='correlation', analysis_metric='wilcoxon',
                    stats_list=corr_list, desired_pairs=desired_pairs, method=method)
        analysis_stats(metric='correlation', analysis_metric='ttest',
                    stats_list=corr_list, desired_pairs=desired_pairs, method=method)

        mae_mean_std(mae_list)
        corr_mean_std(corr_list)


if __name__ == '__main__':
    evaluation_for_each_modality()
