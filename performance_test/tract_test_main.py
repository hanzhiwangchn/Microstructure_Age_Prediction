import numpy as np
import json, os, logging, collections
import scipy
import matplotlib.pyplot as plt

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

result_dir_prefix = '/Users/hanzhiwang/tract_full_res/tracts'
models = ['svr', 'xgb', 'stack_reg_1', 'ensemble']
runtime = 3

model_setting_1 = 'Decom_False_d_measures_Smote_True_0.5_control_True'
model_setting_2 = 'Decom_True_d_measures_Smote_True_0.5_control_True'

model_settings = [model_setting_1, ]
selected_state_list = [0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 15, 
                       16, 17, 18, 20, 22, 23, 26, 27]


def model_performance_single_run():
    """
    It is used to demonstrate averaged model performance (MAE) from different model settings.
    """
    total_res_list = []
    total_res_dict = dict()
    for each_model_setting in model_settings:
        total_res_dict[each_model_setting] = dict()
        for rnd_state in selected_state_list:
            temp_list = []
            for runtime_idx in range(runtime):
                res_folder_name = f'{each_model_setting}_state_{rnd_state}_runtime_{runtime_idx}'
                with open(os.path.join(result_dir_prefix, 'baseline_models', res_folder_name, 'baseline_model_performance.json'), 'r') as f:
                    res_dict = json.load(f)
                temp_list.append(float(res_dict['val_averaged_top3_ensemble']))
            # total_res_dict[each_model_setting][f'state_{rnd_state}'] = np.mean(np.array(temp_list))
            total_res_dict[each_model_setting][f'state_{rnd_state}'] = temp_list
    
    for each_model_setting in model_settings:
        temp = []
        for key in total_res_dict[each_model_setting].keys():
            # temp.append(total_res_dict[each_model_setting][key])
            temp.extend(total_res_dict[each_model_setting][key])
        total_res_list.append(temp)

    total_res_list = np.array(total_res_list)

    for i in range(total_res_list.shape[0]):
        logger.info(f'MAEs: {total_res_list[i]}')
        logger.info(f'MAEs: {total_res_list[i].min()}')
        logger.info(f'MAEs: {total_res_list[i].max()}')
        logger.info(f'Mean MAE: {total_res_list[i].mean()}')
        logger.info(f'Std MAE: {total_res_list[i].std()}')
        logger.info('\n')

    # logger.info(f"{scipy.stats.ttest_rel(total_res_list[0], total_res_list[1])}")
    # logger.info('\n')
    # logger.info(f"{scipy.stats.ttest_rel(total_res_list[1], total_res_list[2])}")
    # logger.info('\n')
    # logger.info(f"{scipy.stats.ttest_rel(total_res_list[0], total_res_list[2])}")
    # logger.info('\n')


def model_performance_multiple_run():
    """
    It is used to demonstrate averaged model performance (MAE) from multiple runs.
    Model predictions from the same random state are averaged together.
    """
    res_dict_total = dict()
    for each_model in models:
        res_dict_total[each_model] = list()
    for i in selected_state_list:
        with open(os.path.join(result_dir_prefix, f'ensemble_runs_results/ensemble_baseline_model_performance_{i}.json')) as f:
            res_dict = json.load(f)
        for each_model in models:
            res_dict_total[each_model].append(float(res_dict[f'ensemble_runs_{each_model}']))
    
    for i in range(len(models)):
        logger.info(f'{models[i]}')
        logger.info(f'MAEs: {res_dict_total[models[i]]}')
        logger.info(f'Mean MAE: {np.mean(res_dict_total[models[i]])}')
        logger.info(f'Std MAE: {np.std(res_dict_total[models[i]])}')
        logger.info(f'min MAE: {np.min(res_dict_total[models[i]])}')
        logger.info(f'max MAE: {np.max(res_dict_total[models[i]])}')
        logger.info('\n')

        
def tract_model_best_performing_regions():
    """collect top regions from trained results"""
    best_region_dict = dict()
    for each_model in models[:-1]:
        best_region_dict[each_model] = list()
    best_region_dict['total'] = list()
    for each_folder in sorted(os.listdir('/Users/hanzhiwang/tract_full_res/tracts/baseline_models')):
        if not each_folder.startswith('.') and not each_folder.startswith('Decom_True'):
            state = int(each_folder.split('_')[-3])
            if state in selected_state_list:
                with open(os.path.join('/Users/hanzhiwang/tract_full_res/tracts/baseline_models', each_folder, 'baseline_model_performance.json')) as f:
                    res = json.load(f)
                for each_model in models[:-1]:
                    best_region_dict[each_model].extend(res[f'top_features_{each_model}'])
                    best_region_dict['total'].extend(res[f'top_features_{each_model}'])
    
    for key, value in best_region_dict.items():
        counter = collections.Counter(value)
        sorted_counter = dict(sorted(counter.items(), key=lambda item: item[1], reverse=False))

        # combine=True
        # # combine left and right regions
        # if combine:
        #     combine_dict = collections.defaultdict(int)
        #     for key_1, value_1 in sorted_counter.items():
        #         if key_1[:2] != "CC":
        #             new_key = '_'.join(key_1.split('_')[:-1])
        #             combine_dict[new_key] += value_1
        #         else:
        #             combine_dict[key_1] += value_1
        #     sorted_counter = dict(sorted(combine_dict.items(), key=lambda item: item[1]))

        # Extract x-labels and frequencies
        x_labels = list(sorted_counter.keys())
        frequencies = list(sorted_counter.values())

        # Plotting the histogram
        plt.figure(figsize=(10, 10))
        plt.barh(x_labels, frequencies, color='skyblue')
        plt.ylabel('Tracts', fontsize=20)
        plt.xlabel('Frequency', fontsize=20)
        if key == 'total':
            plt.title(f'Histogram of Prioritized Tracts', fontsize=30)
        else:
            plt.title(f'Histogram of Prioritized Tracts - {key}', fontsize=30)
        plt.yticks(list(sorted_counter.keys()))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=14)
        if key == 'total':
            plt.axvline(x=30, color='red', linestyle='--')
        else:
            plt.axvline(x=10, color='red', linestyle='--')
        plt.savefig(f'temp1/{key}_importance_features.jpg', bbox_inches='tight')

if __name__ == '__main__':
    tract_model_best_performing_regions()
    # model_performance_multiple_run()
