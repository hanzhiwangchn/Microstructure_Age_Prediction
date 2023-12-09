import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error


image_res_dir_prefix = '/Users/hanzhiwang/image_full_res/images'
model_names = ['densenet', 'resnet']

random_state = [0,1,2,3,5,6,7,10,13,15,16,17,20,22,23,26]
runs = 3


def load_results():
    image_results_dict = dict()
    for each_model in model_names:
        image_results_dict[each_model] = dict()
        for each_state in random_state:
            image_results_dict[each_model][f'state_{each_state}'] = list()
            for i in range(runs):
                df = pd.read_csv(os.path.join(image_res_dir_prefix, 
                                              f'{each_model}_loss_L1_skewed_False_modality_t1w_run{i}_rnd_state_{each_state}', 
                                              'performance_summary_val.csv'))
                image_results_dict[each_model][f'state_{each_state}'].append(df.predicted_value.tolist())

    ground_truth_dict = dict()
    for each_state in random_state:
        ground_truth_dict[f'state_{each_state}'] = list()
        df = pd.read_csv(os.path.join(image_res_dir_prefix, 
                                      f'densenet_loss_L1_skewed_False_modality_t1w_run0_rnd_state_{each_state}', 
                                      'performance_summary_val.csv'))
        ground_truth_dict[f'state_{each_state}'].extend(df.ground_truth.tolist())
    return image_results_dict, ground_truth_dict


def test():
    """calculate MAE for single model and ensemble model"""
    image_results_dict, ground_truth_dict = load_results()
    image_score_dict = dict()
    for each_state in random_state:
        image_score_dict[f'state_{each_state}'] = dict()
        ground_truth = ground_truth_dict[f'state_{each_state}']
        for each_model in model_names:
            image_score_dict[f'state_{each_state}'][each_model] = list()
            total_pred = list()
            for i in range(runs):
                predictions = image_results_dict[each_model][f'state_{each_state}'][i]
                total_pred.append(predictions)
                mae = mean_squared_error(ground_truth, predictions, squared=False)
                image_score_dict[f'state_{each_state}'][each_model].append(mae)
            total_pred_averaged = np.stack(total_pred, axis=-1).mean(axis=-1)
            mae = mean_squared_error(ground_truth, total_pred_averaged, squared=False)
            image_score_dict[f'state_{each_state}'][each_model].append(mae)

    print(image_score_dict)








if __name__ == '__main__':
    test()