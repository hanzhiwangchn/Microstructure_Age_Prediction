import numpy as np
import json, os, logging

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Given a specific parameter config combination, 
# load relevant json file to calculate their mean and std of the MAE
result_dir_prefix = '/Users/hanzhiwang/model_ckpt_results/tracts/baseline_models'
decomposition_feature_name = 'd_measures'


def test():
    total_res_list = []
    for rnd_state in range(20):
        res_folder_name = f'{decomposition_feature_name}_{rnd_state}'
        with open(os.path.join(result_dir_prefix, res_folder_name, 'baseline_model_performance.json'), 'r') as f:
            res_dict = json.load(f)
        total_res_list.append(float(res_dict['val_averaged_top3']))
    total_res_list = np.array(total_res_list)
    logger.info(f'MAEs: {total_res_list}')
    logger.info(f'Mean MAE: {total_res_list.mean()}')
    logger.info(f'Std MAE: {total_res_list.std()}')
    



if __name__ == '__main__':
    test()