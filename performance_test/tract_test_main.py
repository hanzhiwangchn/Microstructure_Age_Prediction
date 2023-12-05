import numpy as np
import json, os, logging
import scipy

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Given a specific parameter config combination, 
# load relevant json file to calculate their mean and std of the MAE
result_dir_prefix = '/Users/hanzhiwang/res/tracts/baseline_models'
decomposition_axis = 'd_measures'
models = ['svr']

model_setting_1 = f'Decom_False_d_measures_Smote_True_0.5_control_True'
model_setting_2 = f'Decom_True_d_measures_Smote_True_0.5_control_True'

model_settings = [model_setting_1, model_setting_2]


def test():
    total_res_list = []
    total_res_dict = dict()
    for each_model_setting in model_settings:
        total_res_dict[each_model_setting] = dict()
        for rnd_state in [0,1,2,3,4,5,6,7,10,12,13,15,16,17,18,20,22,23,26,27]:
            temp_list = []
            for runtime in range(3):
                res_folder_name = f'{each_model_setting}_state_{rnd_state}_runtime_{runtime}'
                with open(os.path.join(result_dir_prefix, res_folder_name, 'baseline_model_performance.json'), 'r') as f:
                    res_dict = json.load(f)
                temp_list.append(float(res_dict[f'val_average_top3_ensemble']))
            # total_res_dict[each_model_setting][f'state_{rnd_state}'] = np.mean(np.array(temp_list))
            total_res_dict[each_model_setting][f'state_{rnd_state}'] = temp_list
    
    for each_model_setting in model_settings:
        temp = []
        for key in total_res_dict[each_model_setting].keys():
            # temp.append(total_res_dict[each_model_setting][key])
            temp.extend(total_res_dict[each_model_setting][key])
        total_res_list.append(temp)

    total_res_list = np.array(total_res_list)
    # print(total_res_list.shape)

    for i in range(total_res_list.shape[0]):
        logger.info(f'MAEs: {total_res_list[i]}')
        logger.info(f'Mean MAE: {total_res_list[i].mean()}')
        logger.info(f'Std MAE: {total_res_list[i].std()}')
        logger.info('\n')

    logger.info(f"{scipy.stats.ttest_rel(total_res_list[0], total_res_list[1])}")
    logger.info('\n')
    # logger.info(f"{scipy.stats.ttest_rel(total_res_list[1], total_res_list[2])}")
    # logger.info('\n')
    # logger.info(f"{scipy.stats.ttest_rel(total_res_list[0], total_res_list[2])}")
    # logger.info('\n')


if __name__ == '__main__':
    test()