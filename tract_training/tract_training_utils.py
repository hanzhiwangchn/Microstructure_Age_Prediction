import logging, json, os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import joblib

from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, KernelPCA
import umap
from sklearn.metrics import r2_score , mean_squared_error

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor


logger = logging.getLogger(__name__)


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
    
    return train_features, test_features, train_labels, test_labels


def standardize_data(train_features, test_features):
    """z-normalization based on diffusion measures"""
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features.reshape(-1, train_features.shape[-1])).reshape(train_features.shape)
    test_features = scaler.transform(test_features.reshape(-1, test_features.shape[-1])).reshape(test_features.shape)

    return train_features, test_features


def apply_decomposition(args, train_features, test_features):
    """
    perform decomposition based on a certain axe or both axes.
    Decomposition methods: {PCA, UMAP, Kernel_PCA}
    Output shape: [num_subject, n_component_tracts, n_component_d_measures]
    """
    # Input shape: [subject, tracts, d_measures]
    num_train_sub, num_tract_region, num_d_measures  = train_features.shape
    num_test_sub, _, _  = test_features.shape

    if args.decomposition_feature_names == 'd_measures':
        # reshape to (subject * tracts, d-measures), since decomposition is based on d-measures.
        train_features = train_features.reshape(-1, train_features.shape[-1])
        test_features = test_features.reshape(-1, test_features.shape[-1])

    # elif args.decomposition_feature_names == 'tracts':
    #     # reshape to (subject * d-measures, tracts), since decomposition is based on tracts.
    #     train_features = np.transpose(train_features, (0, 2, 1)).reshape(-1, train_features.shape[-1])
    #     test_features = np.transpose(test_features, (0, 2, 1)).reshape(-1, test_features.shape[-1])

    # # plot d-measures correlation
    # df_temp = pd.DataFrame.from_records(train_features, columns=args.keyword_dict['d_measures'])
    # plt.subplots(figsize=(15,15))
    # sns.heatmap(df_temp.corr(), cmap="Blues", annot=True)
    # plt.savefig('tract_training/d_measures_correlation.png')

    # plot d-measures correlation
    # df_temp = pd.DataFrame.from_records(train_features, columns=args.keyword_dict['ROI'])
    # plt.subplots(figsize=(15,15))
    # sns.heatmap(df_temp.corr(), cmap="Blues", annot=True)
    # plt.savefig('tract_training/ROI_correlation.png')

        if args.decomposition_method == 'pca':
            # PCA
            pca = PCA(n_components=args.pca_component)
            pca.fit(train_features)
            train_features = pca.transform(train_features)
            test_features = pca.transform(test_features)
            # logger.info(pca.explained_variance_ratio_)
            # logger.info(pca.singular_values_)
            # logger.info(pca.components_)

            # reshape back to original shape (subject, regions, n_component)
            train_features = train_features.reshape(num_train_sub, num_tract_region, args.pca_component)
            test_features = test_features.reshape(num_test_sub, num_tract_region, args.pca_component)

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


        if True:
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
    After the baseline training, averaging results on all tracts.
    """
    res = []
    res_dict = dict()
    # for tract_idx in [3, 27, 28]:
    for tract_idx in range(args.pca_component_1):
    # for tract_idx in range(len(args.keyword_dict['ROI'])):
        # logger.info(f"Current tract region: {args.keyword_dict['ROI'][tract_idx]}")
        logger.info(f"Current tract region: tract PC{tract_idx}")
        res_dict[f"tract PC{tract_idx}"] = dict()

        # input shape (subject, num_features) Pc1, Pc2 ...
        train_features_ROI = train_features[:, tract_idx, :]
        test_features_ROI = test_features[:, tract_idx, :]

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

            res_dict[f"tract PC{tract_idx}"][reg_name] = str(test_rmse)

            # save the trained model
            model_dir = 'tract_training/baseline_models'
            trained_model_name = f"trained_{reg_name}_{args.keyword_dict['ROI'][tract_idx]}.sav"
            joblib.dump(grid.best_estimator_, os.path.join(model_dir, trained_model_name))

            loaded_model = joblib.load(os.path.join(model_dir, trained_model_name))
            preds = loaded_model.predict(test_features_ROI)
            assert test_rmse == mean_squared_error(test_labels, preds, squared=False), test_rmse - mean_squared_error(test_labels, preds, squared=False)

            res.append(preds)
    
    average_res = np.stack(res, axis=-1).mean(axis=-1)
    res_dict['averaged'] = str(mean_squared_error(test_labels, average_res, squared=False))

    with open('baseline_model_performance.json', 'w') as f:
        json.dump(res_dict, f)
