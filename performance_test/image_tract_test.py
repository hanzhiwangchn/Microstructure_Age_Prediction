import numpy as np
import pandas as pd
import os, json
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import scipy

image_res_dir_prefix = '/Users/hanzhiwang/image_full_res/images'
tract_res_dir_prefix = '/Users/hanzhiwang/tract_full_res/tracts'
model_names = ['densenet', 'resnet']

random_state = [0,1,2,3,4,5,6,7,10,12,13,15,16,17,18,20,22,23,26,27]
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


def load_image_data():
    """calculate MAE for single model and ensemble model"""
    image_results_dict, ground_truth_dict = load_results()

    image_results_dict_reshape = dict()
    for each_state in random_state:
        image_results_dict_reshape[f'state_{each_state}'] = dict()
        image_results_dict_reshape[f'state_{each_state}']['ensemble_ensemble'] = list()
        for each_model in model_names:
            image_results_dict_reshape[f'state_{each_state}'][f'ensemble_{each_model}'] = list()
            total_pred = list()
            for i in range(runs):
                predictions = image_results_dict[each_model][f'state_{each_state}'][i]
                total_pred.append(predictions)
            total_pred_averaged = np.stack(total_pred, axis=-1).mean(axis=-1).tolist()
            image_results_dict_reshape[f'state_{each_state}'][f'ensemble_{each_model}'].extend(total_pred_averaged)
            image_results_dict_reshape[f'state_{each_state}']['ensemble_ensemble'].append(total_pred_averaged)
        image_results_dict_reshape[f'state_{each_state}']['ensemble_ensemble'] = \
            np.array(image_results_dict_reshape[f'state_{each_state}']['ensemble_ensemble']).mean(axis=0).tolist()
    # print(image_results_dict_reshape)

    image_ensemble_res = list()
    for each_state in random_state:
        image_ensemble_res.append(image_results_dict_reshape[f'state_{each_state}']['ensemble_ensemble'])

    # print(image_ensemble_res)
    return image_ensemble_res


def load_tract_data():
    res = []
    for each_state in random_state:
        with open(os.path.join(tract_res_dir_prefix, 'ensemble_runs_results', f'ensemble_model_predictions_{each_state}.json'), 'r') as f:
            temp = json.load(f)
        predictions = temp['ensemble_ensemble']
        res.append(predictions)
    return res

def main_test():
    image_ensemble_res = load_image_data()
    image_ensemble_res = np.array(image_ensemble_res)
    tract_ensemble_res = load_tract_data()
    tract_ensemble_res = np.array(tract_ensemble_res)
    _, ground_truth_dict = load_results()

    image_tract_ensemble = (image_ensemble_res + tract_ensemble_res) / 2

    # print(image_ensemble_res)
    # print(tract_ensemble_res)
    # print(image_tract_ensemble)

    image_score_list = list()
    tract_score_list = list()
    image_tract_score_list = list()
    for each_state_idx in range(len(random_state)):
        ground_truth = ground_truth_dict[f'state_{random_state[each_state_idx]}']
        mae_image = mean_absolute_error(ground_truth, image_ensemble_res[each_state_idx])
        image_score_list.append(mae_image)
        mae_tract = mean_absolute_error(ground_truth, tract_ensemble_res[each_state_idx])
        tract_score_list.append(mae_tract)
        mae_total = mean_absolute_error(ground_truth, image_tract_ensemble[each_state_idx])
        image_tract_score_list.append(mae_total)

        temp_dict = dict()
        temp_dict['ensemble_images'] = image_ensemble_res[each_state_idx]
        temp_dict['ensemble_tracts'] = tract_ensemble_res[each_state_idx]
        temp_dict['ensemble_both'] = image_tract_ensemble[each_state_idx]

        plot_predictions_scatter_helper_func(temp_dict, ground_truth, random_state[each_state_idx])

    for each in [image_score_list, tract_score_list, image_tract_score_list]:
        print(each)
        print(np.array(each).mean())
        print(np.array(each).std())
        print(np.array(each).max())
        print(np.array(each).min())

    print(scipy.stats.ttest_rel(image_tract_score_list, image_score_list))
    print(scipy.stats.ttest_rel(image_tract_score_list, tract_score_list))


def plot_predictions_scatter_helper_func(prediction_dict, ground_truth, state):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.plot([10, 70], [10, 70])
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 5)))
    for key, value in prediction_dict.items():
        c = next(color)
        ax.scatter(ground_truth, value, s=200.0, c=c, label=key)
    ax.set_title(f"Prediction scatter plot for all models", fontsize=40)
    ax.set_xlabel('Age', fontsize=40)
    ax.set_ylabel('Predictions',fontsize=40)
    ax.legend(fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(os.path.join('/Users/hanzhiwang/PycharmProjects/Microstructure_Age_Prediction/total_preds_plots', f'scatter_test_performance_{state}.png'), bbox_inches='tight')











if __name__ == '__main__':
    main_test()