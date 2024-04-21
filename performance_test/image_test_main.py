def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

image_res_dir_prefix = '/Users/hanzhiwang/image_full_res/images'
model_names = ['densenet', 'resnet']

random_state = [0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 15, 
                16, 17, 18, 20, 22, 23, 26, 27]
# random_state = [0,1]
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


def test_multi_model():
    """calculate MAE for single model and ensemble model"""
    image_results_dict, ground_truth_dict = load_results()

    image_results_dict['SFCN'] = image_results_dict['resnet']
    image_results_dict['DenseNet'] = image_results_dict['densenet']
    model_names = ['DenseNet', 'SFCN']
    # print(image_results_dict)

    image_results_dict_reshape = dict()
    for each_state in random_state:
        image_results_dict_reshape[f'state_{each_state}'] = dict()
        image_results_dict_reshape[f'state_{each_state}']['Averaged model ensemble'] = list()
        for each_model in model_names:
            image_results_dict_reshape[f'state_{each_state}'][f'{each_model} ensemble'] = list()
            total_pred = list()
            for i in range(runs):
                predictions = image_results_dict[each_model][f'state_{each_state}'][i]
                total_pred.append(predictions)
            total_pred_averaged = np.stack(total_pred, axis=-1).mean(axis=-1).tolist()
            image_results_dict_reshape[f'state_{each_state}'][f'{each_model} ensemble'].extend(total_pred_averaged)
            image_results_dict_reshape[f'state_{each_state}']['Averaged model ensemble'].append(total_pred_averaged)
        image_results_dict_reshape[f'state_{each_state}']['Averaged model ensemble'] = \
            np.array(image_results_dict_reshape[f'state_{each_state}']['Averaged model ensemble']).mean(axis=0).tolist()
    print(image_results_dict_reshape)
    for each_state in random_state:
        plot_predictions_scatter_helper_func(image_results_dict_reshape[f'state_{each_state}'], 
                                             ground_truth_dict[f'state_{each_state}'], 
                                             each_state, 'multi')    
    image_score_dict = dict()
    new_model_names = ['densenet', 'resnet', 'ensemble']
    for each_model in new_model_names:
        image_score_dict[f'ensemble_{each_model}'] = list()
        for each_state in random_state:
            ground_truth = ground_truth_dict[f'state_{each_state}']
            predictions = image_results_dict_reshape[f'state_{each_state}'][f'ensemble_{each_model}']
            mae = mean_absolute_error(ground_truth, predictions)
            image_score_dict[f'ensemble_{each_model}'].append(mae)
    # print(image_score_dict)

    for key, value in image_score_dict.items():
        print(key)
        print(np.array(image_score_dict[key]).mean())
        print(np.array(image_score_dict[key]).std())
        print(np.array(image_score_dict[key]).min())
        print(np.array(image_score_dict[key]).max())


def test_single_model():
    image_results_dict, ground_truth_dict = load_results()

    image_results_dict_reshape = dict()
    for each_state in random_state:
        image_results_dict_reshape[f'state_{each_state}'] = dict()
        for run_idx in range(runs):
            image_results_dict_reshape[f'state_{each_state}'][f'run_{run_idx}'] = dict()
            temp = list()
            for each_model in model_names:
                temp.append(image_results_dict[each_model][f'state_{each_state}'][run_idx])
                image_results_dict_reshape[f'state_{each_state}'][f'run_{run_idx}'][each_model] = image_results_dict[each_model][f'state_{each_state}'][run_idx]
            image_results_dict_reshape[f'state_{each_state}'][f'run_{run_idx}']['ensemble'] = np.array(temp).mean(axis=0).tolist()

    image_results_dict_reshape_score = dict()
    new_model_names = ['densenet', 'resnet', 'ensemble']
    for each_model in new_model_names:
        image_results_dict_reshape_score[each_model] = dict()
        for each_state in random_state:
            image_results_dict_reshape_score[each_model][f'state_{each_state}'] = list()
            for run_idx in range(runs):
                mae = mean_absolute_error(ground_truth_dict[f'state_{each_state}'], 
                                          image_results_dict_reshape[f'state_{each_state}'][f'run_{run_idx}'][each_model])
                image_results_dict_reshape_score[each_model][f'state_{each_state}'].append(round(mae, 2))
    print(image_results_dict_reshape_score)
    print('\n')

    temp_dict = dict()
    for each_model in new_model_names:
        temp_dict[each_model] = list()
        for each_state in random_state:
            temp_dict[each_model].extend(image_results_dict_reshape_score[each_model][f'state_{each_state}'])
    print(temp_dict)
    print('\n')

    for each_model in new_model_names:
        a = np.array(temp_dict[each_model])
        print(each_model)
        print(len(a))
        print(a.mean())
        print(a.std())
        print(a.min())
        print(a.max())

    for each_state in random_state:
        for run_idx in range(runs):
            plot_predictions_scatter_helper_func(image_results_dict_reshape[f'state_{each_state}'][f'run_{run_idx}'], 
                                                 ground_truth_dict[f'state_{each_state}'], 
                                                 each_state, run_idx)

def plot_predictions_scatter_helper_func(prediction_dict, ground_truth, state, run):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot([15, 65], [15, 65])
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 5)))
    for key, value in prediction_dict.items():
        c = next(color)
        ax.scatter(ground_truth, value, s=200.0, c=c, label=key)
    ax.set_title("Prediction scatter plot for all models", fontsize=50)
    ax.set_xlabel('Age', fontsize=50)
    ax.set_ylabel('Predictions', fontsize=50)
    ax.legend(fontsize=35)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(os.path.join('/Users/hanzhiwang/Projects/Microstructure_Age_Prediction/temp', f'scatter_test_performance_{state}_{run}.png'), 
                bbox_inches='tight')

if __name__ == '__main__':
    test_multi_model()
