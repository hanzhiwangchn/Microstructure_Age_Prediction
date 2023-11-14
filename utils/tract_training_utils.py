import logging, json

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error


logger = logging.getLogger(__name__)


def apply_pca(train_features, test_features, keyword_dict):
    """perform pca based on diffusion measures (tract values), since difference d-measures are not orthogonal"""
    # reshape to (subject * regions, d-measures)
    train_features_pca = train_features.reshape(-1, train_features.shape[-1])
    test_features_pca = test_features.reshape(-1, test_features.shape[-1])

    # plot d-measures correlation
    df_temp = pd.DataFrame.from_records(train_features_pca, columns=keyword_dict['d_measures'])
    plt.subplots(figsize=(15,15))
    sns.heatmap(df_temp.corr(), cmap="Blues", annot=True)
    plt.savefig('d_measures_correlation.png')

    # pca fit_transform
    pca = PCA(n_components=3)
    pca.fit_transform(train_features_pca)
    pca.transform(test_features_pca)

    # print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)
    # print(pca.components_)

    # reshape back to original shape (subject, regions, d-measures)
    train_features = train_features_pca.reshape(train_features.shape)
    test_features = test_features_pca.reshape(test_features.shape)

    return train_features, test_features


def paper_plots_derek(train_features, train_labels, keyword_dict):
    # for each tract region, fit linear regression and calculate its R2 score
    for tract in range(len(keyword_dict['ROI'])):
        # PC1 value on a specific tract
        train_features_ROI = train_features[:, tract, 0]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x=train_labels, y=train_features_ROI)
  
        lr = LinearRegression()
        lr.fit(train_labels, train_features_ROI)
        preds = lr.predict(train_labels)
        r2 = r2_score(train_features_ROI, preds)
        ax.text(20, 0, f'{r2}', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})

        # draw LR
        x = np.array([[20], [70]])
        ax.plot(x, lr.predict(x))
        plt.savefig(f"derek_paper_plots_dir/{keyword_dict['ROI'][tract]}.jpg")


def training_tractwise_baseline_model(train_features, test_features, train_labels, test_labels, keyword_dict):
    # for each tract region, build a baseline model using GridSearch
    res_dict = dict()
    for tract in range(len(keyword_dict['ROI']))[:2]:
        logger.info(f"Current tract region: {keyword_dict['ROI'][tract]}")
        res_dict[f"{keyword_dict['ROI'][tract]}"] = dict()

        # input shape (subject, num_features) Pc1, Pc2 ...
        train_features_ROI = train_features[:, tract, :]
        test_features_ROI = test_features[:, tract, :]

        # models
        svr = SVR()
        xgb = XGBRegressor()
        gbr = GradientBoostingRegressor()
        rfr = RandomForestRegressor()
        estimators = [('svr', SVR()), ('xgb', XGBRegressor()), ('gbr', GradientBoostingRegressor())]
        stack_reg = StackingRegressor(estimators=estimators,
                                      final_estimator=RandomForestRegressor(n_estimators=100))
        # reg_configs = list(zip((svr, xgb, gbr, rfr, stack_reg), 
        #                        ('svr', 'xgb', 'gbr', 'rfr', 'stack_reg')))
        reg_configs = list(zip((xgb, stack_reg), ('xgb','stack_reg')))
        
        # fit
        scoring = "neg_root_mean_squared_error"
        # params dict
        params = dict()
        params['rfr'] = {'n_estimators': list(range(100,1000,100))}
        params['svr'] = {'C': [1.0, 3.0, 5.0]}
        params['xgb'] = {'max_depth':range(3, 10, 2), 'min_child_weight':range(1, 6, 2), 
                         'gamma':[i/10.0 for i in range(0,5)], 'subsample':[i/10.0 for i in range(6,10)]}
        params['stack_reg'] = {'svr__C': [1.0, 3.0, 5.0], 'xgb__max_depth': range(3, 10, 2), 
                               'xgb__min_child_weight': range(1, 6, 2), 'xgb__gamma': [i/10.0 for i in range(0,5)], 
                               'final_estimator__n_estimators': list(range(100,1000,100)), 
                               'final_estimator__max_depth': list(range(2, 10, 2))
                            }

        for reg, reg_name in reg_configs:
            # logger.info(reg.get_params().keys())
            grid = RandomizedSearchCV(estimator=reg, param_distributions=params[reg_name], cv=5, scoring=scoring, refit=True, n_iter=100)
            grid.fit(train_features_ROI, train_labels.ravel())

            # test best estimator
            logger.info(f'Mean CV score for {reg_name}: {grid.best_score_}')
            logger.info(f'Best params for {reg_name}: {grid.best_params_}')
            test_rmse = mean_squared_error(test_labels, grid.best_estimator_.predict(test_features_ROI), squared=False)
            logger.info(f"Test set RMSE for {reg_name}: {test_rmse}")

            res_dict[f"{keyword_dict['ROI'][tract]}"][reg_name] = test_rmse

    with open('baseline_model_performance.json', 'w') as f:
        json.dump(res_dict, f)