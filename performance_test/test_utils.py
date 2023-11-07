import pandas as pd
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
