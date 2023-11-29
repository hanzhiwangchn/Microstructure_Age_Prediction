import numpy as np
import json, os, logging
import scipy

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Given a specific parameter config combination, 
# load relevant json file to calculate their mean and std of the MAE
result_dir_prefix = '/Users/hanzhiwang/model_ckpt_results/tracts/baseline_models'
decomposition_feature_name = 'd_measures'
models = ['svr', 'rfr', 'xgb']


def test():
    total_res_dict = dict()
    for each_model in models:
        total_res_dict[each_model] = []

    # d_measures 8 ,14 [0,2,3,4,7,12,13,16]
    # both 9 ,11, 14 [0,1,2,3,4,5,6,7,10,12,13,15,16,17,18,19]
    # tract
    for rnd_state in [0,2,3,4,7,12,13,16]:
        res_folder_name = f'{decomposition_feature_name}_{rnd_state}'
        with open(os.path.join(result_dir_prefix, res_folder_name, 'baseline_model_performance.json'), 'r') as f:
            res_dict = json.load(f)
        for each_model in models:
            total_res_dict[each_model].append(float(res_dict[f'val_averaged_top3_{each_model}']))

    for key in total_res_dict.keys():
        total_res_dict[key] = np.array(total_res_dict[key])
        logger.info(f'{key}')
        logger.info(f'MAEs: {total_res_dict[key]}')
        logger.info(f'Mean MAE: {total_res_dict[key].mean()}')
        logger.info(f'Std MAE: {total_res_dict[key].std()}')
        logger.info('\n')

    logger.info(f"{scipy.stats.ttest_rel(total_res_dict['svr'], total_res_dict['xgb'])}")
    logger.info('\n')
    logger.info(f"{scipy.stats.ttest_rel(total_res_dict['svr'], total_res_dict['rfr'])}")
    logger.info('\n')
    logger.info(f"{scipy.stats.ttest_rel(total_res_dict['rfr'], total_res_dict['xgb'])}")



if __name__ == '__main__':
    test()