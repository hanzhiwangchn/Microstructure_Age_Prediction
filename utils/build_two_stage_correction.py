
# def apply_two_stage_correction(args, validation_loader, device, model_config, results_folder, input_shape):
#     """apply two-stage correction on validation set"""
#     # initialize best model
#     if args.model == 'resnet':
#         net_test = ResNetModelHF().to(device)
#     elif args.model == 'resnet_head':
#         net_test = ResNetModelWithHeadHF().to(device)

#     # reload weights
#     net_test.load_state_dict(state_dict=torch.load(os.path.join(results_folder, f"{model_config}_Best_Model.pt")))
#     net_test.eval()

#     df_test = pd.read_csv(os.path.join(results_folder, f"{model_config}_performance_summary.csv"))

#     # save validation results to DataFrame
#     val_preds_list = []
#     val_labels_list = []

#     with torch.no_grad():
#         for images, labels in validation_loader:
#             images, labels = images.to(device), labels.to(device)
#             preds = net_test(images)

#             assert preds.shape == labels.shape

#             val_preds_list.append(preds)
#             val_labels_list.append(labels)

#         # preds and labels will have shape (*, 1)
#         val_preds_tensor = torch.cat(val_preds_list, 0)
#         val_labels_tensor = torch.cat(val_labels_list, 0)

#         assert preds.shape == labels.shape
#         assert preds.shape[1] == 1

#     df_validation = pd.DataFrame()
#     df_validation['predicted_value'] = val_preds_tensor.squeeze().cpu().numpy()
#     df_validation['ground_truth'] = val_labels_tensor.squeeze().cpu().numpy()

#     validation_slope, validation_bias = two_stage_linear_fit(df_val=df_validation)
#     two_steps_bias_correction(validation_slope, validation_bias, df_test, model_config)


# def two_stage_linear_fit(df_val):
#     """two-stage approach: linear fit on validation set"""
#     predicted_value = df_val['predicted_value'].values
#     ground_truth = df_val['ground_truth'].values

#     # use linear regression
#     lr = LinearRegression()
#     lr.fit(ground_truth.reshape(-1, 1), predicted_value.reshape(-1, 1))

#     test_val = np.array([0, 1])
#     test_pred = lr.predict(test_val.reshape(-1, 1))

#     slope = test_pred[1] - test_pred[0]
#     bias = test_pred[0]
#     return slope[0], bias[0]


# def two_steps_bias_correction(slope, bias, df_test, model_config):
    # """two-stage approach: correction on predicted value to get unbiased prediction"""
    # df_test['predicted_value'] = df_test['predicted_value'].apply(lambda x: (x-bias) / slope)
    # df_test.to_csv(os.path.join(results_folder, f"{model_config}_corrected_performance_summary.csv"))
