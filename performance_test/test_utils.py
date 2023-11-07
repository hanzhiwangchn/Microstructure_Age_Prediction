import pandas as pd
import numpy as np
import os

IMAGE_MODALITY_LIST = ['KFA_DKI', 'FA_CHARMED', 'RD_CHARMED', 'MD_CHARMED', 'FRtot_CHARMED']
LOSS_TYPE_LIST = ['L1_normal', 'L1_normal_corrected', 'L1_skewed']
RESULT_DIRS = '~/model_ckpt_results/'


def get_model_results(args):
    """read csv files for all runs"""
    dfs = []
    for loss_type in LOSS_TYPE_LIST:
        dfs_temp = []
        for random_state in range(1000, 1001):
            dfs_temp_2 = []
            for i in range(args.num_runs):
                if loss_type == 'L1_normal':
                    model_name = f'{args.model}_loss_L1_skewed_False_modality_{args.image_modality}_' \
                                 f'run{i}_rnd_state_{random_state}'
                    df_temp = pd.read_csv(
                        os.path.join(RESULT_DIRS, model_name, 'performance_summary.csv'))
                elif loss_type == 'L1_skewed':
                    model_name = f'{args.model}_loss_L1_skewed_True_modality_{args.image_modality}_' \
                                 f'run{i}_rnd_state_{random_state}'
                    df_temp = pd.read_csv(
                        os.path.join(RESULT_DIRS, model_name, 'performance_summary.csv'))
                elif loss_type == 'L1_normal_corrected':
                    model_name = f'{args.model}_loss_L1_skewed_False_modality_{args.image_modality}_' \
                                 f'run{i}_rnd_state_{random_state}'
                    df_temp = pd.read_csv(
                        os.path.join(RESULT_DIRS, model_name, 'corrected_performance_summary.csv'))

                df_temp['error'] = df_temp['predicted_value'] - df_temp['ground_truth']
                df_temp['abs_error'] = df_temp['error'].abs()
                dfs_temp_2.append(df_temp)
            dfs_temp.append(dfs_temp_2)
        dfs.append(dfs_temp)
    return dfs


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

def calculate_correlation(dfs):
    """single model"""
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
    """single model"""
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


def mae_mean_std(mae_list):
    """single model"""
    print()
    mae_list = np.array(mae_list)
    mae_list = mae_list.reshape(mae_list.shape[0], -1)
    means = mae_list.mean(axis=1)
    stds = mae_list.std(axis=1)
    assert means.shape == (len(LOSS_TYPE_LIST), )
    for i in range(len(LOSS_TYPE_LIST)):
        print(f'{LOSS_TYPE_LIST[i]}: {means[i]}, {stds[i]}')


def corr_mean_std(corr_list):
    """single model"""
    print()
    corr_list = np.array(corr_list)
    corr_list = corr_list.reshape(corr_list.shape[0], -1)
    means = corr_list.mean(axis=1)
    stds = corr_list.std(axis=1)
    assert means.shape == (len(LOSS_TYPE_LIST), )
    for i in range(len(LOSS_TYPE_LIST)):
        print(f'{LOSS_TYPE_LIST[i]}: {means[i]}, {stds[i]}')

