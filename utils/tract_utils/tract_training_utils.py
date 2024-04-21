import logging, json, os, joblib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import r2_score, mean_absolute_error

import smogn
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

results_folder = '/Users/hanzhiwang/tract_full_res/tracts'
logger = logging.getLogger(__name__)


def update_args(args):
    """update arguments"""
    args.keyword_dict = build_keyword_dict()
    args.result_dir = results_folder
    args.model_config = f'Decom_{args.decomposition}_{args.decomposition_axis}_' \
                        f'Smote_{args.smote}_{args.smote_threshold}_control_{args.smote_label_control}_' \
                        f'state_{args.random_state}_runtime_{args.runtime}'
    args.baseline_model_dir = f'{args.result_dir}/baseline_models/{args.model_config}'
    if not args.multi_runs_ensemble:
        os.makedirs(args.baseline_model_dir, exist_ok=True)

    return args


def build_keyword_dict():
    """A keyword dict for parser"""
    keyword_dict = dict()
    # tract of interest
    keyword_dict['ROI'] = ['AF_left', 'AF_right', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 
                           'CG_left', 'CG_right', 'CST_left', 'CST_right', 'FX_left', 'FX_right', 'IFO_left', 
                           'IFO_right', 'ILF_left', 'ILF_right', 'SLF_III_left', 'SLF_III_right', 'SLF_II_left', 
                           'SLF_II_right', 'SLF_I_left', 'SLF_I_right', 'T_PREF_left', 'T_PREF_right', 'UF_left', 
                           'UF_right']
    # diffusion measures
    keyword_dict['d_measures'] = ['KFA_DKI', 'ICVF_NODDI', 'AD_CHARMED', 'FA_CHARMED', 'RD_CHARMED', 
                                  'MD_CHARMED', 'FRtot_CHARMED', 'MWF_mcDESPOT']                   
    return keyword_dict


def load_tract_data(args):
    """load tract data and do a train-val-test-split"""
    features = np.load(os.path.join(args.tract_data_dir, 'tract_value_compact.npy'))
    labels = np.load(os.path.join(args.tract_data_dir, 'tract_age_compact.npy'))

    # Stratified train val test Split based on age
    df = pd.DataFrame(data=labels, columns=['Age'])
    df['Age_categorical'] = pd.qcut(df['Age'], 10, labels=[i for i in range(10)])

    # train-val & test split
    split = StratifiedShuffleSplit(test_size=args.test_size, random_state=args.random_state)
    train_val_index, test_index = next(split.split(df, df['Age_categorical']))
    stratified_train_val_set = df.loc[train_val_index]
    assert sorted(train_val_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # train & val split
    split2 = StratifiedShuffleSplit(test_size=args.val_size, random_state=args.random_state)
    train_index, val_index = next(split2.split(stratified_train_val_set, stratified_train_val_set['Age_categorical']))

    # NOTE: StratifiedShuffleSplit returns RangeIndex instead of the Original Index of the new DataFrame
    assert sorted(train_index.tolist() + val_index.tolist()) == list(range(len(stratified_train_val_set.index)))
    assert sorted(train_index.tolist() + val_index.tolist()) != sorted(list(stratified_train_val_set.index))
    
    # get the correct index of original DataFrame for train/val set
    train_index = train_val_index[train_index]
    val_index = train_val_index[val_index]

    # ensure there is no duplicated index
    assert sorted(train_index.tolist() + val_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # split data based on index
    train_features = features[train_index].astype(np.float32)
    val_features = features[val_index].astype(np.float32)
    test_features = features[test_index].astype(np.float32)
    train_labels = df.loc[train_index, 'Age'].values.astype(np.float32)
    val_labels = df.loc[val_index, 'Age'].values.astype(np.float32)
    test_labels = df.loc[test_index, 'Age'].values.astype(np.float32)
    logger.info(f"Training feature shape: {train_features.shape}, val feature shape: {val_features.shape}, test feature shape: {test_features.shape}."
                f"Training label shape: {train_labels.shape}, val label shape: {val_labels.shape}, test label shape: {test_labels.shape}")
    
    # visualize train-val-test distributions
    plt.figure()
    sns.kdeplot(data=train_labels, label = "Train")
    sns.kdeplot(data=val_labels, label = "Val")
    sns.kdeplot(data=test_labels, label = "Test")
    plt.legend()
    plt.savefig(f'{args.result_dir}/train_val_test_dist_state_{args.random_state}.png')
    plt.close()

    return train_features, val_features, test_features, train_labels, val_labels, test_labels


def exploratory_data_analysis(args, train_features, train_labels):
    """
    perform EDA on the training dataset. 
    The shape of features is (num_subjects, num_tracts, num_d_measures)
    """
    # Check outliers. Since we usually perform standardization on each d-measure channel, 
    # we plot histogram per measures.
    for d_measure_idx in range(train_features.shape[-1]):
        plt.subplots(figsize=(20, 20))
        plt.hist(train_features[:, :, d_measure_idx].flatten(), bins=20)
        plt.title(f"Histogram on {args.keyword_dict['d_measures'][d_measure_idx]}", fontsize=40)
        plt.xlabel('Frequency', fontsize=30)
        plt.ylabel('Value',fontsize=30)
        plt.savefig(f"tract_training/tract_eda/Hist_{args.keyword_dict['d_measures'][d_measure_idx]}.png", bbox_inches='tight')
        plt.close()

    # check age distribution
    plt.hist(train_labels, bins=20)
    plt.title('Age Distribution', fontsize=30)
    plt.savefig("tract_training/tract_eda/Train_Age.png", bbox_inches='tight')

    # plot d-measures correlation
    train_features_temp = train_features.reshape(-1, train_features.shape[-1])
    df_temp = pd.DataFrame.from_records(train_features_temp, columns=args.keyword_dict['d_measures'])
    plt.subplots(figsize=(20, 20))
    sns.heatmap(df_temp.corr(), cmap="PiYG", annot=True, annot_kws={"size": 30})
    plt.savefig('tract_training/tract_eda/d_measures_correlation_total.png', bbox_inches='tight')
    plt.close()
    
    # plot tract region correlation
    train_features_temp = np.transpose(train_features, (0, 2, 1))
    train_features_temp = train_features_temp.reshape(-1, train_features_temp.shape[-1])
    df_temp = pd.DataFrame.from_records(train_features_temp, columns=args.keyword_dict['ROI'])
    plt.subplots(figsize=(20, 20))
    sns.heatmap(df_temp.corr(), cmap="PiYG", annot=True)
    plt.savefig('tract_training/tract_eda/ROI_correlation_total.png', bbox_inches='tight')
    plt.close()


def scale_data(args, train_features, val_features, test_features):
    """Scale d_measures features using RobustScaler for each tract region"""
    # Input shape: [subject, tracts, d_measures]
    num_train_sub, num_tract_region, num_d_measures = train_features.shape
    num_val_sub, _, _ = val_features.shape
    num_test_sub, _, _  = test_features.shape

    args.scale = 1
    train_features, val_features, test_features = transformation_helper(args=args, train_features=train_features, val_features=val_features, test_features=test_features)
    args.scale = 0

    # Output shape: [subject, tracts, d_measures]
    assert train_features.shape == (num_train_sub, num_tract_region, num_d_measures)
    assert val_features.shape == (num_val_sub, num_tract_region, num_d_measures)
    assert test_features.shape == (num_test_sub, num_tract_region, num_d_measures)

    return train_features, val_features, test_features


def apply_decomposition(args, train_features, val_features, test_features):
    """
    perform decomposition based on a certain axis or both axises.
    Decomposition methods: {PCA, UMAP}. 
    Output shape: [num_subject, num_tracts or num_d_measures, n_principle_component]
    We perform decomposition individually for each tract or each d_measure by default.
    When decomposing d_measures, we will train model on each tract and select top-performing models.
    When decomposing tracts, we will train model on each d_measure and select top-performing models.
    """
    # Input shape: [subject, tracts, d_measures]
    num_train_sub, num_tract_region, num_d_measures = train_features.shape
    num_val_sub, _, _ = val_features.shape
    num_test_sub, _, _  = test_features.shape

    # Decomposition on d_measures
    if args.decomposition_axis == 'd_measures':
        train_features, val_features, test_features = transformation_helper(args=args, train_features=train_features, val_features=val_features, test_features=test_features)

        assert train_features.shape == (num_train_sub, num_tract_region, args.n_component)
        assert val_features.shape == (num_val_sub, num_tract_region, args.n_component)
        assert test_features.shape == (num_test_sub, num_tract_region, args.n_component)
    
    # Decomposition on tracts
    elif args.decomposition_axis == 'tracts':
        # reshape to (subject, d-measures, tracts), since decomposition is based on tracts.
        train_features = np.transpose(train_features, (0, 2, 1))
        val_features = np.transpose(val_features, (0, 2, 1))
        test_features = np.transpose(test_features, (0, 2, 1))

        train_features, val_features, test_features = transformation_helper(args=args, train_features=train_features, val_features=val_features, test_features=test_features)

        assert train_features.shape == (num_train_sub, num_d_measures, args.n_component)
        assert val_features.shape == (num_val_sub, num_d_measures, args.n_component)
        assert test_features.shape == (num_test_sub, num_d_measures, args.n_component)

    # Decomposition on tracts and d-measures
    elif args.decomposition_axis == 'both':
        # since we observe that different tracts give different performance, 
        # we perform decomposition firstly by tract then by d_measures.
        train_features = np.transpose(train_features, (0, 2, 1))
        val_features = np.transpose(val_features, (0, 2, 1))
        test_features = np.transpose(test_features, (0, 2, 1))

        train_features, val_features, test_features = transformation_helper(args=args, train_features=train_features, val_features=val_features, test_features=test_features)

        assert train_features.shape == (num_train_sub, num_d_measures, args.n_component)
        assert val_features.shape == (num_val_sub, num_d_measures, args.n_component)
        assert test_features.shape == (num_test_sub, num_d_measures, args.n_component)

        # reshape back to (num_sub, n_pc_tract, num_d_measures)
        train_features = np.transpose(train_features, (0, 2, 1))
        val_features = np.transpose(val_features, (0, 2, 1))
        test_features = np.transpose(test_features, (0, 2, 1))

        train_features, val_features, test_features = transformation_helper(args=args, train_features=train_features, val_features=val_features, test_features=test_features)
        
        assert train_features.shape == (num_train_sub, args.n_component, args.n_component)
        assert val_features.shape == (num_val_sub, args.n_component, args.n_component)
        assert test_features.shape == (num_test_sub, args.n_component, args.n_component)
        
    return train_features, val_features, test_features


def transformation_helper(args, train_features, val_features, test_features):
    """helper function for scale and decompose data"""
    train_features_list, val_features_list, test_features_list = [], [], []
    # Iterate on the second last dimension (The transformed data should appear on the final dimension)
    for idx in range(train_features.shape[-2]):
        train_features_ROI = train_features[:, idx, :]
        val_features_ROI = val_features[:, idx, :]
        test_features_ROI = test_features[:, idx, :]
        
        if args.scale == 1:        
            transformer = RobustScaler()
            transformer.fit(train_features_ROI)
        elif args.decomposition_method == 'pca':
            transformer = PCA(n_components=args.n_component)
            transformer.fit(train_features_ROI)
        elif args.decomposition_method == 'umap':
            umap_decomposer = umap.UMAP(n_components=args.n_component)
            transformer = umap_decomposer.fit(train_features_ROI)

        # transform
        train_features_ROI = transformer.transform(train_features_ROI)
        val_features_ROI = transformer.transform(val_features_ROI)
        test_features_ROI = transformer.transform(test_features_ROI)

        train_features_list.append(train_features_ROI)
        val_features_list.append(val_features_ROI)
        test_features_list.append(test_features_ROI)
    
    train_features = np.stack(train_features_list, axis=-2)
    val_features = np.stack(val_features_list, axis=-2)
    test_features = np.stack(test_features_list, axis=-2)

    return train_features, val_features, test_features


def paper_plots_derek(args, train_features, train_labels):
    """for each tract region, fit linear regression and calculate its R2 score"""
    for tract_idx in range(len(args.keyword_dict['ROI'])):
        # PC1 value on a specific tract
        train_features_ROI = train_features[:, tract_idx, 0]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x=train_labels, y=train_features_ROI)
        ax.xaxis('Age')
        ax.yaxis('PC1')
  
        lr = LinearRegression()
        lr.fit(train_labels, train_features_ROI)
        ax.text(10, 20, f'{r2_score(train_features_ROI, lr.predict(train_labels))}', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})

        # draw LR
        x = np.array([[20], [70]])
        ax.plot(x, lr.predict(x))
        plt.savefig(f"derek_paper_plots_dir/{args.keyword_dict['ROI'][tract_idx]}.jpg")


def apply_smote(args, train_features, train_labels, feature_name):
    """apply smogn to create synthetic examples"""
    df = pd.DataFrame(data=train_features, columns=[f'feature_{i}' for i in range(train_features.shape[-1])])
    df['label'] = train_labels

    # SMOGN
    n_tries = 0
    done = False
    while not done:
        try:
            data_smogn = smogn.smoter(
                data=df, y = 'label', k = 5, samp_method = args.smote_method,
                rel_thres = args.smote_threshold, rel_method = 'manual',
                rel_ctrl_pts_rg=[[20, 1, 0], 
                                 [30, 0, 0], 
                                 [40, 0, 0], 
                                 [50, 1, 0], 
                                 [60, 1, 0]]
                )
            done = True
        except ValueError:
            if n_tries < 999:
                n_tries += 1
            else:
                raise

    if args.smote_label_control:
        # no subject should have age larger than 70
        data_smogn = data_smogn[data_smogn['label'] <= 70]

    # visualize modified distribution
    plt.figure()
    sns.kdeplot(df['label'], label = "Original")
    sns.kdeplot(data_smogn['label'], label = "Modified")
    plt.legend()
    plt.savefig(f"{args.result_dir}/smote_comparison/{feature_name}_dist.png")
    plt.close()

    return data_smogn.iloc[:, :-1].values, data_smogn.iloc[:, -1].values


def training_baseline_model(args, train_features, val_features, train_labels, val_labels):
    """
    Build a baseline tract-wise or measure-wise model using RandomizedSearch
    After the baseline training, averaging top-performing models based on val set
    Train_features shape: (subject, tracts or d_measures, n_PCs)
    
    When decomposition method is 'd_measures' and 'both', tract-wise baseline is trained.
    When decomposition method is 'tracts', d_measure-wise baseline is trained.
    """
    res_dict = dict()
    for idx in range(train_features.shape[-2]):
        if args.decomposition_axis == 'd_measures':
            feature_of_interest = args.keyword_dict['ROI'][idx]
        elif args.decomposition_axis == 'tracts':
            feature_of_interest = args.keyword_dict['d_measures'][idx]
        elif args.decomposition_axis == 'both':
            feature_of_interest = f"Tract_PC_{idx}"
        logger.info(f"Current feature: {feature_of_interest}")
        res_dict[feature_of_interest] = dict()

        # input shape (subject, num_features) Pc1, Pc2 ...
        train_features_ROI = train_features[:, idx, :]
        val_features_ROI = val_features[:, idx, :]

        # apply SMOTE 
        if args.smote:
            train_features_ROI, train_labels_ROI = apply_smote(args=args, train_features=train_features_ROI, train_labels=train_labels, feature_name=feature_of_interest)
        else:
            train_labels_ROI = train_labels

        # models
        svr = SVR()
        xgb = XGBRegressor()
        estimators_1 = [('svr', SVR())]
        stack_reg_1 = StackingRegressor(estimators=estimators_1,
                                        final_estimator=XGBRegressor())
        reg_configs = list(zip((svr, xgb, stack_reg_1), 
                               ('svr', 'xgb', 'stack_reg_1')))
        
        # scoring function
        scoring = "neg_mean_absolute_error"
        # params dict
        params = dict()
        params['svr'] = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
                         'gamma': ['auto', 'scale'], 
                         'C': [i/10.0 for i in range(5, 50, 2)]}
        params['xgb'] = {'max_depth':range(3, 20, 2), 
                         'min_child_weight':range(1, 9, 2), 
                         'gamma':[i/10.0 for i in range(0, 5)], 
                         'subsample':[i/10.0 for i in range(6, 10)]}
        params['stack_reg_1'] = {'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
                                 'svr__gamma': ['auto', 'scale'], 
                                 'svr__C': [i/10.0 for i in range(5, 50, 2)], 
                                 'final_estimator__max_depth':range(3, 20, 2), 
                                 'final_estimator__min_child_weight':range(1, 9, 2),  
                                 'final_estimator__gamma':[i/10.0 for i in range(0, 5)], 
                                 'final_estimator__subsample':[i/10.0 for i in range(6, 10)]}

        # fit
        for reg, reg_name in reg_configs:
            grid = RandomizedSearchCV(estimator=reg, param_distributions=params[reg_name], cv=10, 
                                      scoring=scoring, refit=True, n_iter=300, n_jobs=-1)
            grid.fit(train_features_ROI, train_labels_ROI)

            # test best estimator
            logger.info(f'Mean CV score for {reg_name}: {-grid.best_score_}')
            val_mae = mean_absolute_error(val_labels, grid.best_estimator_.predict(val_features_ROI))
            logger.info(f"Val set MAE for {reg_name}: {val_mae}")

            # save results and models
            res_dict[feature_of_interest][reg_name] = str(val_mae)
            trained_model_name = f"trained_{reg_name}_{feature_of_interest}.sav"              
            joblib.dump(grid.best_estimator_, os.path.join(args.baseline_model_dir, trained_model_name))
    
    with open(os.path.join(args.baseline_model_dir, 'baseline_model_performance.json'), 'w') as f:
        json.dump(res_dict, f)


def load_trained_model_ensemble(args, val_features, test_features, val_labels, test_labels, trained_model_list):
    """load trained models and create ensemble models"""
    val_res_dict ,test_res_dict, top_prediction_dict = dict(), dict(), dict()

    # load each trained model and give predictions on val and test set.
    for reg_name in trained_model_list:
        val_res_dict[reg_name] = list()
        test_res_dict[reg_name] = list()
        for idx in range(val_features.shape[-2]):
            val_features_ROI = val_features[:, idx, :] 
            test_features_ROI = test_features[:, idx, :]

            # fetch the right model
            feature_name = get_feature_name_helper_func(args=args, idx=idx) 
            trained_model_name = f'trained_{reg_name}_{feature_name}.sav'
            loaded_model = joblib.load(os.path.join(args.baseline_model_dir, trained_model_name))
            
            val_preds = loaded_model.predict(val_features_ROI)
            test_preds = loaded_model.predict(test_features_ROI)
            val_res_dict[reg_name].append(val_preds)
            test_res_dict[reg_name].append(test_preds)

        # select idx with top 3 best performance and best features based on val results
        smallest_idx = np.argpartition(np.array([mean_absolute_error(val_labels, i) for i in val_res_dict[reg_name]]), 3)[:3]
        if args.decomposition_axis == 'd_measures':
            top_features = [args.keyword_dict['ROI'][i] for i in smallest_idx]
        elif args.decomposition_axis == 'tracts':
            top_features = [args.keyword_dict['d_measures'][i] for i in smallest_idx]

        # average predictions for all features or only top features
        average_res_val = np.stack(val_res_dict[reg_name], axis=-1).mean(axis=-1)
        top3_res_val = np.stack(val_res_dict[reg_name], axis=-1)[:, smallest_idx].mean(axis=-1)
        average_res_test = np.stack(test_res_dict[reg_name], axis=-1).mean(axis=-1)
        top3_res_test = np.stack(test_res_dict[reg_name], axis=-1)[:, smallest_idx].mean(axis=-1)
        # save prediction for current model
        top_prediction_dict[reg_name] = top3_res_val

        # load previous result json and add averaged performance
        with open(os.path.join(args.baseline_model_dir, 'baseline_model_performance.json'), 'r+') as f:
            res_dict = json.load(f)
            res_dict[f'val_averaged_all_{reg_name}'] = str(mean_absolute_error(val_labels, average_res_val))
            res_dict[f'val_averaged_top3_{reg_name}'] = str(mean_absolute_error(val_labels, top3_res_val))
            res_dict[f'test_averaged_all_{reg_name}'] = str(mean_absolute_error(test_labels, average_res_test))
            res_dict[f'test_averaged_top3_{reg_name}'] = str(mean_absolute_error(test_labels, top3_res_test))
            res_dict[f'top_features_{reg_name}'] = top_features
        with open(os.path.join(args.baseline_model_dir, 'baseline_model_performance.json'), 'w') as f:
            json.dump(res_dict, f)

    # ensemble predictions for all models
    total_pred_list = [each_prediction for each_prediction in top_prediction_dict.values()]
    top_prediction_dict['ensemble'] = np.stack(total_pred_list, axis=-1).mean(axis=-1)
    with open(os.path.join(args.baseline_model_dir, 'baseline_model_performance.json'), 'r+') as f:
        res_dict = json.load(f)
        res_dict[f'val_averaged_top3_ensemble'] = str(mean_absolute_error(val_labels, top_prediction_dict['ensemble']))
    with open(os.path.join(args.baseline_model_dir, 'baseline_model_performance.json'), 'w') as f:
        json.dump(res_dict, f)

    # add scatter plot for test set
    plot_predictions_scatter_helper_func(args=args, is_multi_runs=False, prediction_dict=top_prediction_dict, ground_truth=val_labels)


def ensemble_model_from_multiple_runs(args, val_features, test_features, val_labels, test_labels, trained_model_list):
    """This function is essentially the above function with another loop"""
    val_res_dict ,test_res_dict, top_prediction_dict = dict(), dict(), dict()
    num_runs = 3

    # load each trained model and give predictions on val and test set.
    for reg_name in trained_model_list:
        top_prediction_dict[reg_name] = dict()
        for run_idx in range(num_runs):
            val_res_dict[reg_name] = list()
            test_res_dict[reg_name] = list()
            for idx in range(val_features.shape[-2]):
                val_features_ROI = val_features[:, idx, :] 
                test_features_ROI = test_features[:, idx, :] 

                # fetch the right model
                feature_name = get_feature_name_helper_func(args=args, idx=idx) 
                args.model_config = f'Decom_{args.decomposition}_{args.decomposition_axis}_' \
                        f'Smote_{args.smote}_{args.smote_threshold}_control_{args.smote_label_control}_' \
                        f'state_{args.random_state}_runtime_{run_idx}'
                args.baseline_model_dir = f'{args.result_dir}/baseline_models/{args.model_config}'
                trained_model_name = f'trained_{reg_name}_{feature_name}.sav'
                loaded_model = joblib.load(os.path.join(args.baseline_model_dir, trained_model_name))
                
                val_preds = loaded_model.predict(val_features_ROI)
                test_preds = loaded_model.predict(test_features_ROI)
                val_res_dict[reg_name].append(val_preds)
                test_res_dict[reg_name].append(test_preds)

            # select idx with top 3 best performance based on val results
            smallest_idx = np.argpartition(np.array([mean_absolute_error(val_labels, i) for i in val_res_dict[reg_name]]), 3)[:3]
            # save prediction for current model
            top_prediction_dict[reg_name][f'run{run_idx}'] = np.stack(val_res_dict[reg_name], axis=-1)[:, smallest_idx].mean(axis=-1)

        # ensemble predictions for all runs
        total_pred_list = [each_prediction for each_prediction in top_prediction_dict[reg_name].values()]
        top_prediction_dict[reg_name][f'ensemble_runs_{reg_name}'] = np.stack(total_pred_list, axis=-1).mean(axis=-1)

    top_prediction_dict['ensemble'] = np.stack([top_prediction_dict[each_model][f'ensemble_runs_{each_model}'] for each_model in trained_model_list], axis=-1).mean(axis=-1)

    args.save_ensemble_dir = '/Users/hanzhiwang/tract_full_res/tracts/ensemble_runs_results'
    res_dict = dict()
    for reg_name in trained_model_list:
        res_dict[f'ensemble_runs_{reg_name}'] = str(mean_absolute_error(val_labels, top_prediction_dict[reg_name][f'ensemble_runs_{reg_name}']))
    res_dict[f'ensemble_runs_ensemble'] = str(mean_absolute_error(val_labels, top_prediction_dict['ensemble']))
    with open(os.path.join(args.save_ensemble_dir, f'ensemble_baseline_model_performance_{args.random_state}.json'), 'w') as f:
        json.dump(res_dict, f)

    # add scatter plot for test set
    temp_dict = dict()
    for each_model in trained_model_list:
        if each_model == 'xgb':
            temp_dict[f'XGBoost ensemble'] = top_prediction_dict[f'{each_model}'][f'ensemble_runs_{each_model}']
        elif each_model == 'svr':
            temp_dict[f'SVR ensemble'] = top_prediction_dict[f'{each_model}'][f'ensemble_runs_{each_model}']
        elif each_model == 'stack_reg_1':
            temp_dict[f'stacking model ensemble'] = top_prediction_dict[f'{each_model}'][f'ensemble_runs_{each_model}']
    temp_dict['Averaged model ensemble'] = top_prediction_dict['ensemble']
    converted_dict = {key: value.tolist() for key, value in temp_dict.items()}
    with open(os.path.join(args.save_ensemble_dir, f'ensemble_model_predictions_{args.random_state}.json'), 'w') as f:
        json.dump(converted_dict, f)
    plot_predictions_scatter_helper_func(args=args, is_multi_runs=True, prediction_dict=temp_dict, ground_truth=val_labels)


def get_feature_name_helper_func(args, idx):
    if args.decomposition_axis == 'd_measures':
        feature_name = args.keyword_dict['ROI'][idx]
    elif args.decomposition_axis == 'both':
        feature_name = f"Tract_PC_{idx}"
    elif args.decomposition_axis == 'tracts':
        feature_name = args.keyword_dict['d_measures'][idx]

    return feature_name


def plot_predictions_scatter_helper_func(args, is_multi_runs, prediction_dict, ground_truth):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot([15, 65], [15, 65])
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 5)))
    for key, value in prediction_dict.items():
        c = next(color)
        ax.scatter(ground_truth, value, s=200.0, c=c, label=key)
    ax.set_title(f"Prediction scatter plot for all models", fontsize=50)
    ax.set_xlabel('Age', fontsize=50)
    ax.set_ylabel('Predictions',fontsize=50)
    ax.legend(fontsize=35)
    ax.tick_params(axis='both', which='major', labelsize=25)
    if is_multi_runs:
        plt.savefig(os.path.join(args.save_ensemble_dir, f'ensemble_scatter_test_performance_{args.random_state}.png'), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(args.baseline_model_dir, f'scatter_test_performance.png'), bbox_inches='tight')
