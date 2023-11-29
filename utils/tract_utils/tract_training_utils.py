import logging, json, os, joblib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, KernelPCA
import umap
from sklearn.metrics import r2_score , mean_squared_error

import smogn
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

results_folder = 'model_ckpt_results/tracts'
logger = logging.getLogger(__name__)


def update_args(args):
    """update arguments"""
    args.keyword_dict = build_keyword_dict()
    args.result_dir = results_folder

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
    plt.savefig(f'{args.result_dir}/train_val_test_dist.png')
    plt.close()

    # remove training subjects whose age is larger than 60
    if args.outlier_removal:
        remaining_idx = np.where(train_labels < 60)
        train_features = train_features[remaining_idx]
        train_labels = train_labels[remaining_idx]
        logger.info('Perform outlier removal on Age')
        logger.info(f"Training feature shape: {train_features.shape}, training label shape: {train_labels.shape}.")
    
    return train_features, val_features, test_features, train_labels, val_labels, test_labels


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


def scale_data(train_features, val_features, test_features):
    """Scale features using statistics that are robust to outliers based on diffusion measures"""
    scaler = RobustScaler()
    train_features = scaler.fit_transform(train_features.reshape(-1, train_features.shape[-1])).reshape(train_features.shape)
    val_features = scaler.transform(val_features.reshape(-1, val_features.shape[-1])).reshape(val_features.shape)
    test_features = scaler.transform(test_features.reshape(-1, test_features.shape[-1])).reshape(test_features.shape)

    return train_features, val_features, test_features


def apply_decomposition(args, train_features, val_features, test_features):
    """
    perform decomposition based on a certain axe or both axes.
    Decomposition methods: {PCA, UMAP, Kernel_PCA}. 
    Currently, we only use PCA.
    Output shape: [num_subject, num_tracts or num_d_measures, n_principle_component]
    We perform decomposition individually for each tract or each d_measure by default.
    When decomposing d_measures, we will train model on each tract and select top-performing models.
    When decomposing tracts, we will train model on each d_measure and select top-performing models.
    """
    # Input shape: [subject, tracts, d_measures]
    num_train_sub, num_tract_region, num_d_measures  = train_features.shape
    num_val_sub, _, _ = val_features.shape
    num_test_sub, _, _  = test_features.shape

    # Decomposition on d_measures
    if args.decomposition_feature_names == 'd_measures':
        # PCA
        if args.decomposition_method == 'pca':
            train_features_list = []
            val_features_list = []
            test_features_list = []
            # iterate each tract
            for idx in range(train_features.shape[1]):
                train_features_ROI = train_features[:, idx, :]
                val_features_ROI = val_features[:, idx, :]
                test_features_ROI = test_features[:, idx, :]

                pca = PCA(n_components=args.pca_component)
                pca.fit(train_features_ROI)
                train_features_ROI = pca.transform(train_features_ROI)
                val_features_ROI = pca.transform(val_features_ROI)
                test_features_ROI = pca.transform(test_features_ROI)

                train_features_list.append(train_features_ROI)
                val_features_list.append(val_features_ROI)
                test_features_list.append(test_features_ROI)
            
            train_features = np.stack(train_features_list, axis=1)
            val_features = np.stack(val_features_list, axis=1)
            test_features = np.stack(test_features_list, axis=1)
            
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
        assert val_features.shape == (num_val_sub, num_tract_region, args.pca_component)
        assert test_features.shape == (num_test_sub, num_tract_region, args.pca_component)
    
    # tracts decomposition
    elif args.decomposition_feature_names == 'tracts':
        # reshape to (subject, d-measures, tracts), since decomposition is based on tracts.
        train_features = np.transpose(train_features, (0, 2, 1))
        val_features = np.transpose(val_features, (0, 2, 1))
        test_features = np.transpose(test_features, (0, 2, 1))

        # PCA
        if args.decomposition_method == 'pca':
            train_features_list = []
            val_features_list = []
            test_features_list = []
            # iterate each d_measure
            for idx in range(train_features.shape[1]):
                train_features_ROI = train_features[:, idx, :]
                val_features_ROI = val_features[:, idx, :]
                test_features_ROI = test_features[:, idx, :]

                pca = PCA(n_components=args.pca_component)
                pca.fit(train_features_ROI)
                train_features_ROI = pca.transform(train_features_ROI)
                val_features_ROI = pca.transform(val_features_ROI)
                test_features_ROI = pca.transform(test_features_ROI)

                train_features_list.append(train_features_ROI)
                val_features_list.append(val_features_ROI)
                test_features_list.append(test_features_ROI)
            
            train_features = np.stack(train_features_list, axis=1)
            val_features = np.stack(val_features_list, axis=1)
            test_features = np.stack(test_features_list, axis=1)

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
        assert val_features.shape == (num_val_sub, num_d_measures, args.pca_component)
        assert test_features.shape == (num_test_sub, num_d_measures, args.pca_component)

    # d_measures and tracts decomposition
    elif args.decomposition_feature_names == 'both':
        # since we observe that different tracts give different performance, 
        # we perform pca firstly by tract then by d_measures.
        train_features = np.transpose(train_features, (0, 2, 1))
        val_features = np.transpose(val_features, (0, 2, 1))
        test_features = np.transpose(test_features, (0, 2, 1))

        # PCA
        if args.decomposition_method == 'pca':
            train_features_list = []
            val_features_list = []
            test_features_list = []
            # iterate each d_measure
            for idx in range(train_features.shape[1]):
                train_features_ROI = train_features[:, idx, :]
                val_features_ROI = val_features[:, idx, :]
                test_features_ROI = test_features[:, idx, :]

                pca = PCA(n_components=args.pca_component)
                pca.fit(train_features_ROI)
                train_features_ROI = pca.transform(train_features_ROI)
                val_features_ROI = pca.transform(val_features_ROI)
                test_features_ROI = pca.transform(test_features_ROI)

                train_features_list.append(train_features_ROI)
                val_features_list.append(val_features_ROI)
                test_features_list.append(test_features_ROI)
            
            train_features = np.stack(train_features_list, axis=1)
            val_features = np.stack(val_features_list, axis=1)
            test_features = np.stack(test_features_list, axis=1)

        # reshape back to (num_sub, n_pc_tract, num_d_measures)
        train_features = np.transpose(train_features, (0, 2, 1))
        val_features = np.transpose(val_features, (0, 2, 1))
        test_features = np.transpose(test_features, (0, 2, 1))
        assert train_features.shape == (num_train_sub, args.pca_component, num_d_measures)
        assert val_features.shape == (num_val_sub, args.pca_component, num_d_measures)
        assert test_features.shape == (num_test_sub, args.pca_component, num_d_measures)
        
        # second pca
        if args.decomposition_method == 'pca':
            train_features_list = []
            val_features_list = []
            test_features_list = []
            # iterate each d_measure
            for idx in range(train_features.shape[1]):
                train_features_ROI = train_features[:, idx, :]
                val_features_ROI = val_features[:, idx, :]
                test_features_ROI = test_features[:, idx, :]

                pca = PCA(n_components=args.pca_component)
                pca.fit(train_features_ROI)
                train_features_ROI = pca.transform(train_features_ROI)
                val_features_ROI = pca.transform(val_features_ROI)
                test_features_ROI = pca.transform(test_features_ROI)

                train_features_list.append(train_features_ROI)
                val_features_list.append(val_features_ROI)
                test_features_list.append(test_features_ROI)
            
            train_features = np.stack(train_features_list, axis=1)
            val_features = np.stack(val_features_list, axis=1)
            test_features = np.stack(test_features_list, axis=1)

        assert train_features.shape == (num_train_sub, args.pca_component, args.pca_component)
        assert val_features.shape == (num_val_sub, args.pca_component, args.pca_component)
        assert test_features.shape == (num_test_sub, args.pca_component, args.pca_component)

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


def apply_smote(args, train_features, train_labels, idx):
    """apply smogn to create synthetic examples"""
    df = pd.DataFrame(data=train_features, columns=[f'feature_{i}' for i in range(train_features.shape[-1])])
    df['label'] = train_labels

    # SMOGN
    n_tries = 0
    done = False
    while not done:
        try:
            data_smogn = smogn.smoter(
                data = df, y = 'label', k = 5, samp_method = args.smote_method,
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

    # no subject should have age larger than 70
    data_smogn = data_smogn[data_smogn['label'] <= 70]

    # visualize modified distribution
    if args.decomposition_feature_names == 'd_measures':
        name = args.keyword_dict['ROI'][idx]
    elif args.decomposition_feature_names == 'both':
        name = f'Tract PC {idx}'
    elif args.decomposition_feature_names == 'tracts':
        name = args.keyword_dict['d_measures'][idx]
    plt.figure()
    sns.kdeplot(df['label'], label = "Original")
    sns.kdeplot(data_smogn['label'], label = "Modified")
    plt.legend()
    plt.savefig(f"{args.result_dir}/smote_comparison/{name}_dist.png")
    plt.close()

    return data_smogn.iloc[:, :-1].values, data_smogn.iloc[:, -1].values


def training_baseline_model(args, train_features, val_features, train_labels, val_labels):
    """
    Build a baseline tract-wise or measure-wise model using RandomizedSearch
    After the baseline training, averaging top-performing models.
    Train_features shape: (subject, tracts or d_measures, n_PCs)
    
    When decomposition method is 'd_measures' and 'both', tract-wise baseline is trained.
    When decomposition method is 'tracts', d_measure-wise baseline is trained.
    """
    res_dict = dict()
    for idx in range(train_features.shape[1]):
        if args.decomposition_feature_names == 'd_measures':
            logger.info(f"Current tract region: {args.keyword_dict['ROI'][idx]}")
            res_dict[f"{args.keyword_dict['ROI'][idx]}"] = dict()
        elif args.decomposition_feature_names == 'both':
            logger.info(f"Current tract region PC: {idx}")
            res_dict[f"Tract PC {idx}"] = dict()
        elif args.decomposition_feature_names == 'tracts':
            logger.info(f"Current d_measure: {args.keyword_dict['d_measures'][idx]}")
            res_dict[f"{args.keyword_dict['d_measures'][idx]}"] = dict()

        # input shape (subject, num_features) Pc1, Pc2 ...
        train_features_ROI = train_features[:, idx, :]
        val_features_ROI = val_features[:, idx, :]

        # apply SMOTE 
        if args.smote:
            train_features_ROI, train_labels_ROI = apply_smote(args=args, train_features=train_features_ROI, 
                                                               train_labels=train_labels, idx=idx)
        # models
        svr = SVR()
        xgb = XGBRegressor()
        # estimators = [('svr', SVR())]
        # stack_reg = StackingRegressor(estimators=estimators,
        #                               final_estimator=XGBRegressor())
        # reg_configs = list(zip((svr, xgb, stack_reg), ('svr', 'xgb', 'stack_reg')))
        reg_configs = list(zip((svr, ), ('svr', )))
        args.model_list = []
        
        scoring = "neg_root_mean_squared_error"
        # params dict
        params = dict()
        params['svr'] = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
                         'gamma': ['auto', 'scale'], 
                         'C': [i/10.0 for i in range(5, 50, 2)]}
        params['rfr'] = {'n_estimators': range(50, 150, 10), 
                         'max_features': ['sqrt', 'log2', 1.0], 
                         'max_depth': range(2, 20, 2)}
        params['xgb'] = {'max_depth':range(3, 20, 2), 
                         'min_child_weight':range(1, 9, 2), 
                         'gamma':[i/10.0 for i in range(0, 5)], 
                         'subsample':[i/10.0 for i in range(6, 10)]}
        params['stack_reg'] = {'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
                               'svr__gamma': ['auto', 'scale'], 
                               'svr__C': [i/10.0 for i in range(5, 50, 2)], 
                               'final_estimator__max_depth':range(3, 20, 2), 
                               'final_estimator__min_child_weight':range(1, 9, 2),  
                               'final_estimator__gamma':[i/10.0 for i in range(0, 5)], 
                               'final_estimator__subsample':[i/10.0 for i in range(6, 10)]}

        # fit
        for reg, reg_name in reg_configs:
            args.model_list.append(reg_name)
            grid = RandomizedSearchCV(estimator=reg, param_distributions=params[reg_name], cv=10, 
                                      scoring=scoring, refit=True, n_iter=300, n_jobs=-1)
            grid.fit(train_features_ROI, train_labels_ROI)

            # test best estimator
            logger.info(f'Mean CV score for {reg_name}: {-grid.best_score_}')
            val_rmse = mean_squared_error(val_labels, grid.best_estimator_.predict(val_features_ROI), squared=False)
            logger.info(f"Validation set RMSE for {reg_name}: {val_rmse}")

            # save results and trained model
            baseline_model_dir = f'{args.result_dir}/baseline_models/{args.decomposition_feature_names}_{args.random_state}'
            os.makedirs(baseline_model_dir, exist_ok=True)

            if args.decomposition_feature_names == 'd_measures':
                res_dict[f"{args.keyword_dict['ROI'][idx]}"][reg_name] = str(val_rmse)
                trained_model_name = f"trained_{reg_name}_{args.keyword_dict['ROI'][idx]}.sav"
            elif args.decomposition_feature_names == 'both':
                res_dict[f"Tract PC {idx}"][reg_name] = str(val_rmse)
                trained_model_name = f"trained_{reg_name}_Tract_PC_{idx}.sav"
            elif args.decomposition_feature_names == 'tracts':
                res_dict[f"{args.keyword_dict['d_measures'][idx]}"][reg_name] = str(val_rmse)
                trained_model_name = f"trained_{reg_name}_{args.keyword_dict['d_measures'][idx]}.sav"                
            joblib.dump(grid.best_estimator_, os.path.join(baseline_model_dir, trained_model_name))
    
    with open(os.path.join(baseline_model_dir, 'baseline_model_performance.json'), 'w') as f:
        json.dump(res_dict, f)

    args.model_list = list(set(args.model_list))
    return args


def load_trained_model_ensemble(args, train_features, val_features, test_features, val_labels, test_labels):
    """load trained models and create ensemble models"""
    val_res_dict = dict()
    test_res_dict = dict()
    baseline_model_dir = f'{args.result_dir}/baseline_models/{args.decomposition_feature_names}_{args.random_state}'
    
    # load each trained model again and give predictions on test set again.
    for reg_name in args.model_list:
        val_res_dict[reg_name] = []
        test_res_dict[reg_name] = []
        for idx in range(train_features.shape[1]):
            val_features_ROI = val_features[:, idx, :] 
            test_features_ROI = test_features[:, idx, :] 
            if args.decomposition_feature_names == 'd_measures':
                data_name = args.keyword_dict['ROI'][idx]
            elif args.decomposition_feature_names == 'both':
                data_name = f"Tract_PC_{idx}"
            elif args.decomposition_feature_names == 'tracts':
                data_name = args.keyword_dict['d_measures'][idx]
        
            # fetch the right model
            trained_model_name = f'trained_{reg_name}_{data_name}.sav'
            loaded_model = joblib.load(os.path.join(baseline_model_dir, trained_model_name))
            val_preds = loaded_model.predict(val_features_ROI)
            test_preds = loaded_model.predict(test_features_ROI)
            val_res_dict[reg_name].append(val_preds)
            test_res_dict[reg_name].append(test_preds)

        # select idx with best test loss based on val results
        smallest_idx = np.argpartition(np.array([mean_squared_error(val_labels, i, squared=False) for i in val_res_dict[reg_name]]), 3)[:3]
        
        # ensemble model performance
        average_res_val = np.stack(val_res_dict[reg_name], axis=-1).mean(axis=-1)
        top3_res_val = np.stack(val_res_dict[reg_name], axis=-1)[:, smallest_idx].mean(axis=-1)
        average_res_test = np.stack(test_res_dict[reg_name], axis=-1).mean(axis=-1)
        top3_res_test = np.stack(test_res_dict[reg_name], axis=-1)[:, smallest_idx].mean(axis=-1)

        # load previous result json and add ensemble performance
        with open(os.path.join(baseline_model_dir, 'baseline_model_performance.json'), 'r+') as f:
            res_dict = json.load(f)
            res_dict[f'val_averaged_all_{reg_name}'] = str(mean_squared_error(val_labels, average_res_val, squared=False))
            res_dict[f'val_averaged_top3_{reg_name}'] = str(mean_squared_error(val_labels, top3_res_val, squared=False))
            res_dict[f'test_averaged_all_{reg_name}'] = str(mean_squared_error(test_labels, average_res_test, squared=False))
            res_dict[f'test_averaged_top3_{reg_name}'] = str(mean_squared_error(test_labels, top3_res_test, squared=False))
        with open(os.path.join(baseline_model_dir, 'baseline_model_performance.json'), 'w') as f:
            json.dump(res_dict, f)

        # add scatter plot for test set
        if args.scatter_prediction_plot:
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.plot([10, 70], [10, 70])
            ax.scatter(test_labels, top3_res_test, s=200.0, c='r')
            ax.scatter(val_labels, top3_res_val, s=200.0, c='g')
            ax.set_title(f"Prediction scatter plot", fontsize=40)
            ax.set_xlabel('Age', fontsize=40)
            ax.set_ylabel('Predictions',fontsize=40)
            ax.tick_params(axis='both', which='major', labelsize=25)
            plt.savefig(os.path.join(baseline_model_dir, f'scatter_test_performance_{reg_name}.png'), bbox_inches='tight')

    # aggregate all model predictions
    # average_res_val_all_model_list = []
    # top3_res_val_all_model_list = []
    # average_res_test_all_model_list = []
    # top3_res_test_all_model_list = []
    # for reg_name in args.model_list:
    #     average_res_val = np.stack(val_res_dict[reg_name], axis=-1).mean(axis=-1)
    #     top3_res_val = np.stack(val_res_dict[reg_name], axis=-1)[:, smallest_idx].mean(axis=-1)
    #     average_res_test = np.stack(test_res_dict[reg_name], axis=-1).mean(axis=-1)
    #     top3_res_test = np.stack(test_res_dict[reg_name], axis=-1)[:, smallest_idx].mean(axis=-1)

    #     average_res_val_all_model_list.append(average_res_val)
    #     top3_res_val_all_model_list.append(top3_res_val)
    #     average_res_test_all_model_list.append(average_res_test)
    #     top3_res_test_all_model_list.append(top3_res_test)

    # average_res_val_all_model = np.stack(average_res_val_all_model_list, axis=-1).mean(axis=-1)
    # average_res_test_all_model = np.stack(test_res_dict[reg_name], axis=-1).mean(axis=-1)

    