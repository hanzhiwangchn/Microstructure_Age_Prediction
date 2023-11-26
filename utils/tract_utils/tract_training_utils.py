import logging, json, os
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import joblib

from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, KernelPCA
import umap
from sklearn.metrics import r2_score , mean_squared_error

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor


logger = logging.getLogger(__name__)


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
    """load tract data and do a train-test-split"""
    features = np.load(os.path.join(args.data_dir, 'tract_value_compact.npy'))
    labels = np.load(os.path.join(args.data_dir, 'tract_age_compact.npy'))

    # Stratified train test Split based on age
    df = pd.DataFrame(data=labels, columns=['Age'])
    df['Age_categorical'] = pd.qcut(df['Age'], 10, labels=[i for i in range(10)])

    split = StratifiedShuffleSplit(test_size=args.test_size, random_state=args.random_state)
    train_index, test_index = next(split.split(df, df['Age_categorical']))
    assert sorted(train_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # train test split
    train_features = features[train_index].astype(np.float32)
    test_features = features[test_index].astype(np.float32)
    train_labels = np.expand_dims(df.loc[train_index, 'Age'].values, axis=1).astype(np.float32)
    test_labels = np.expand_dims(df.loc[test_index, 'Age'].values, axis=1).astype(np.float32)
    logger.info(f"Training feature shape: {train_features.shape}, test feature shape: {test_features.shape}. "
                f"Training label shape: {train_labels.shape}, test label shape: {test_labels.shape}")
    
    # remove training subjects whose age is larger than 60
    if args.outlier_removal:
        remaining_idx = np.where(train_labels.flatten() < 60)
        train_features = train_features[remaining_idx]
        train_labels = train_labels[remaining_idx]
    
    return train_features, test_features, train_labels, test_labels


def perform_exploratory_data_analysis(args, train_features, train_labels):
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


def scale_data(train_features, test_features):
    """Scale features using statistics that are robust to outliers based on diffusion measures"""
    scaler = RobustScaler()
    train_features = scaler.fit_transform(train_features.reshape(-1, train_features.shape[-1])).reshape(train_features.shape)
    test_features = scaler.transform(test_features.reshape(-1, test_features.shape[-1])).reshape(test_features.shape)

    return train_features, test_features


def apply_decomposition(args, train_features, test_features):
    """
    perform decomposition based on a certain axe or both axes.
    Decomposition methods: {PCA, UMAP, Kernel_PCA}
    Output shape: [num_subject, n_component_tracts, n_component_d_measures]
    When decomposing d_measures, we will train model on each tract and select top-performing models.
    When decomposing tracts, we will train model on each d_measure and select top-performing models.
    """
    # Input shape: [subject, tracts, d_measures]
    num_train_sub, num_tract_region, num_d_measures  = train_features.shape
    num_test_sub, _, _  = test_features.shape

    # d_measures decomposition
    if args.decomposition_feature_names == 'd_measures':
        # reshape to (subject * tracts, d-measures), since decomposition is based on d-measures.
        train_features = train_features.reshape(-1, train_features.shape[-1])
        test_features = test_features.reshape(-1, test_features.shape[-1])

        # PCA
        if args.decomposition_method == 'pca':
            pca = PCA(n_components=args.pca_component)
            pca.fit(train_features)
            train_features = pca.transform(train_features)
            test_features = pca.transform(test_features)
            logger.info(pca.explained_variance_ratio_)
            logger.info(pca.singular_values_)
            logger.info(pca.components_)

            # reshape back to original shape (subject, tracts, n_component)
            train_features = train_features.reshape(num_train_sub, num_tract_region, args.pca_component)
            test_features = test_features.reshape(num_test_sub, num_tract_region, args.pca_component)

        # Kernel PCA
        elif args.decomposition_method == 'kernel_pca':
            kernel_pca = KernelPCA(n_components=args.kernel_pca_component)
            kernel_pca.fit(train_features)
            train_features = kernel_pca.transform(train_features)
            test_features = kernel_pca.transform(test_features)

            # reshape back to original shape (subject, regions, n_component)
            train_features = train_features.reshape(num_train_sub, num_tract_region, args.kernel_pca_component)
            test_features = test_features.reshape(num_test_sub, num_tract_region, args.kernel_pca_component)

        # UMAP
        elif args.decomposition_method == 'umap':
            umap_reducer = umap.UMAP(n_components=args.umap_component)
            trans = umap_reducer.fit(train_features)
            train_features = trans.transform(train_features)
            test_features = trans.transform(test_features)

            # reshape back to original shape (subject, regions, n_component)
            train_features = train_features.reshape(num_train_sub, num_tract_region, args.umap_component)
            test_features = test_features.reshape(num_test_sub, num_tract_region, args.umap_component)

        assert train_features.shape == (num_train_sub, num_tract_region, args.pca_component)
        assert test_features.shape == (num_test_sub, num_tract_region, args.pca_component)
    
    # tracts decomposition
    elif args.decomposition_feature_names == 'tracts':
        # reshape to (subject * d-measures, tracts), since decomposition is based on tracts.
        train_features = np.transpose(train_features, (0, 2, 1))
        train_features = train_features.reshape(-1, train_features.shape[-1])
        test_features = np.transpose(test_features, (0, 2, 1))
        test_features = test_features.reshape(-1, test_features.shape[-1])

        # PCA
        if args.decomposition_method == 'pca':
            pca = PCA(n_components=args.pca_component)
            pca.fit(train_features)
            train_features = pca.transform(train_features)
            test_features = pca.transform(test_features)
            # logger.info(pca.explained_variance_ratio_)
            # logger.info(pca.singular_values_)
            # logger.info(pca.components_)

            # reshape back to shape (subject, d_measures, n_component)
            train_features = train_features.reshape(num_train_sub, num_d_measures, args.pca_component)
            test_features = test_features.reshape(num_test_sub, num_d_measures, args.pca_component)

        # Kernel PCA
        elif args.decomposition_method == 'kernel_pca':
            kernel_pca = KernelPCA(n_components=args.kernel_pca_component)
            kernel_pca.fit(train_features)
            train_features = kernel_pca.transform(train_features)
            test_features = kernel_pca.transform(test_features)

            # reshape back to shape (subject, d_measures, n_component)
            train_features = train_features.reshape(num_train_sub, num_d_measures, args.kernel_pca_component)
            test_features = test_features.reshape(num_test_sub, num_d_measures, args.kernel_pca_component)

        # UMAP
        elif args.decomposition_method == 'umap':
            umap_reducer = umap.UMAP(n_components=args.umap_component)
            trans = umap_reducer.fit(train_features)
            train_features = trans.transform(train_features)
            test_features = trans.transform(test_features)

            # reshape back to shape (subject, d_measures, n_component)
            train_features = train_features.reshape(num_train_sub, num_d_measures, args.umap_component)
            test_features = test_features.reshape(num_test_sub, num_d_measures, args.umap_component)

        assert train_features.shape == (num_train_sub, num_d_measures, args.pca_component)
        assert test_features.shape == (num_test_sub, num_d_measures, args.pca_component)

    # d_measures and tracts decomposition
    if False:
        train_features = np.transpose(train_features, (0, 2, 1)).reshape(-1, num_tract_region)
        test_features = np.transpose(test_features, (0, 2, 1)).reshape(-1, num_tract_region)

        if args.decomposition_method == 'pca':
            # PCA
            pca = PCA(n_components=args.pca_component_1)
            pca.fit(train_features)
            train_features = pca.transform(train_features)
            test_features = pca.transform(test_features)
            logger.info(pca.explained_variance_ratio_)
            logger.info(pca.singular_values_)
            logger.info(pca.components_)

            # reshape back to original shape (subject, regions, n_component)
            train_features = train_features.reshape(num_train_sub, args.pca_component_1, args.pca_component)
            test_features = test_features.reshape(num_test_sub, args.pca_component_1, args.pca_component)

        elif args.decomposition_method == 'kernel_pca':
            # PCA
            pca = KernelPCA(n_components=args.kernel_pca_component)
            pca.fit(train_features)
            train_features = pca.transform(train_features)
            test_features = pca.transform(test_features)

            # reshape back to original shape (subject, regions, n_component)
            train_features = train_features.reshape(num_train_sub, num_tract_region, args.kernel_pca_component)
            test_features = test_features.reshape(num_test_sub, num_tract_region, args.kernel_pca_component)

        elif args.decomposition_method == 'umap':
            # UMAP
            umap_reducer = umap.UMAP(n_components=args.umap_component)
            trans = umap_reducer.fit(train_features)
            train_features = trans.transform(train_features)
            test_features = trans.transform(test_features)

            # reshape back to original shape (subject, regions, n_component)
            train_features = train_features.reshape(num_train_sub, num_tract_region, args.umap_component)
            test_features = test_features.reshape(num_test_sub, num_tract_region, args.umap_component)

            pass


    return train_features, test_features


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


def training_tractwise_baseline_model(args, train_features, test_features, train_labels, test_labels):
    """
    For each tract region, build a baseline model using RandomizedSearch. 
    After the baseline training, averaging top-performing models.
    Train_features shape: (subject, tracts or d_measures, *)
    """
    res_dict = dict()
    for idx in range(train_features.shape[1]):
        if args.decomposition_feature_names in ['d_measures', 'both']:
            logger.info(f"Current tract region: {args.keyword_dict['ROI'][idx]}")
            res_dict[f"{args.keyword_dict['ROI'][idx]}"] = dict()
        elif args.decomposition_feature_names == 'tracts':
            logger.info(f"Current d_measure: {args.keyword_dict['d_measures'][idx]}")
            res_dict[f"{args.keyword_dict['d_measures'][idx]}"] = dict()

        # input shape (subject, num_features) Pc1, Pc2 ...
        train_features_ROI = train_features[:, idx, :]
        test_features_ROI = test_features[:, idx, :]

        # models
        svr = SVR()
        # xgb = XGBRegressor()
        # rfr = RandomForestRegressor()
        estimators = [('svr', SVR())]
        stack_reg = StackingRegressor(estimators=estimators,
                                      final_estimator=RandomForestRegressor())
        reg_configs = list(zip((svr,), 
                               ('svr',)))
        
        scoring = "neg_root_mean_squared_error"
        # params dict
        params = dict()
        params['svr'] = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
                         'gamma': ['auto', 'scale'], 
                         'C': [i/10.0 for i in range(5, 50, 2)]}
        params['rfr'] = {'n_estimators': range(50, 150, 10), 
                         'max_features': ['sqrt', 'log2', 1.0], 
                         'max_depth': range(3, 30, 3)}
        params['xgb'] = {'max_depth':range(3, 15, 2), 
                         'min_child_weight':range(1, 9, 2), 
                         'gamma':[i/10.0 for i in range(0, 5)], 
                         'subsample':[i/10.0 for i in range(6, 10)]}
        params['lgb'] = {'num_leaves':range(3, 10, 2), 
                         'max_bin':range(20, 200, 20), 
                         'bagging_fraction':[i/10.0 for i in range(6, 10)]}
        params['stack_reg'] = {'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
                               'svr__gamma': ['auto', 'scale'], 
                               'svr__C': [i/10.0 for i in range(5, 50, 2)], 
                               'final_estimator__n_estimators': range(50, 150, 10), 
                               'final_estimator__max_features': ['sqrt', 'log2', 1.0], 
                               'final_estimator__max_depth': range(3, 30, 3)}

        # fit
        for reg, reg_name in reg_configs:
            grid = RandomizedSearchCV(estimator=reg, param_distributions=params[reg_name], cv=10, 
                                      scoring=scoring, refit=True, n_iter=300, n_jobs=-1)
            grid.fit(train_features_ROI, train_labels.ravel())

            # test best estimator
            logger.info(f'Mean CV score for {reg_name}: {-grid.best_score_}')
            test_rmse = mean_squared_error(test_labels, grid.best_estimator_.predict(test_features_ROI), squared=False)
            logger.info(f"Test set RMSE for {reg_name}: {test_rmse}")

            if args.decomposition_feature_names in ['d_measures', 'both']:
                res_dict[f"{args.keyword_dict['ROI'][idx]}"][reg_name] = str(test_rmse)
            elif args.decomposition_feature_names == 'tracts':
                res_dict[f"{args.keyword_dict['d_measures'][idx]}"][reg_name] = str(test_rmse)

            # save the trained model
            model_dir = f'tract_training/baseline_models/{args.decomposition_feature_names}_{args.random_state}'
            os.makedirs(model_dir, exist_ok=True)
            if args.decomposition_feature_names in ['d_measures', 'both']:
                trained_model_name = f"trained_{reg_name}_{args.keyword_dict['ROI'][idx]}.sav"
            elif args.decomposition_feature_names == 'tracts':
                trained_model_name = f"trained_{reg_name}_{args.keyword_dict['d_measures'][idx]}.sav"
            joblib.dump(grid.best_estimator_, os.path.join(model_dir, trained_model_name))

    return res_dict


def load_trained_model_ensemble(args, train_features, test_features, test_labels, res_dict):
    res = []
    model_dir = f'tract_training/baseline_models/{args.decomposition_feature_names}_{args.random_state}'
    # load each trained model again and give predictions on test set again.
    for idx in range(train_features.shape[1]):
        test_features_ROI = test_features[:, idx, :] 
        if args.decomposition_feature_names in ['d_measures', 'both']:
            name = args.keyword_dict['ROI'][idx]
        elif args.decomposition_feature_names == 'tracts':
            name = args.keyword_dict['d_measures'][idx]
        # iterate all model names
        for trained_model_name in os.listdir(model_dir):
            if name in trained_model_name:                
                loaded_model = joblib.load(os.path.join(model_dir, trained_model_name))
                preds = loaded_model.predict(test_features_ROI)
                res.append(preds)

    # count the best performing regions
    best_performing_tracts = []
    # select idx with best test loss
    smallest_idx = np.argpartition(np.array([mean_squared_error(test_labels, i, squared=False) for i in res]), 3)[:3]
    
    best_performing_tracts.extend(args.keyword_dict['ROI'][smallest_idx])
    logger.info(f'Best performing tracts are {Counter(best_performing_tracts)}')
    
    # ensemble model performance
    average_res = np.stack(res, axis=-1).mean(axis=-1)
    top3_res = np.stack(res, axis=-1)[:, smallest_idx].mean(axis=-1)
    res_dict['averaged_all'] = str(mean_squared_error(test_labels, average_res, squared=False))
    res_dict['averaged_top3'] = str(mean_squared_error(test_labels, top3_res, squared=False))
    with open(os.path.join(model_dir, 'baseline_model_performance.json'), 'w') as f:
        json.dump(res_dict, f)

    # add scatter plot here
    # plt.subplots(figsize=(20, 20))
    # plt.plot([10, 60], [10, 60])
    # plt.scatter(test_labels, top3_res)
    # plt.title(f"Prediction scatter plot", fontsize=40)
    # plt.xlabel('Age', fontsize=30)
    # plt.ylabel('Predictions',fontsize=30)
    # plt.savefig(f"tract_training/tract_eda/Hist_{args.keyword_dict['d_measures'][d_measure_idx]}.png", bbox_inches='tight')
    # plt.close()


