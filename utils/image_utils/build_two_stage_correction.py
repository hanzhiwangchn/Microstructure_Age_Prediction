import torch, os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def apply_two_stage_correction(args, val_loader, model):
    """apply two-stage correction on validation set"""
    model.eval()
    df_test = pd.read_csv(os.path.join(args.out_dir, "performance_summary.csv"))

    # save validation results to DataFrame
    val_preds_list = []
    val_labels_list = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(batch['image'])

            assert outputs.shape == batch['label'].shape
            assert len(outputs.shape) == 2

            val_preds_list.append(outputs)
            val_labels_list.append(batch['label'])

        # preds and labels will have shape (*, 1)
        val_preds_tensor = torch.cat(val_preds_list, 0)
        val_labels_tensor = torch.cat(val_labels_list, 0)

    df_val = pd.DataFrame()
    df_val['predicted_value'] = val_preds_tensor.squeeze().cpu().numpy()
    df_val['ground_truth'] = val_labels_tensor.squeeze().cpu().numpy()

    val_slope, val_bias = two_stage_linear_fit(df_val=df_val)
    two_steps_bias_correction(val_slope, val_bias, df_test, args)


def two_stage_linear_fit(df_val):
    """two-stage approach: linear fit on validation set"""
    predicted_value = df_val['predicted_value'].values
    ground_truth = df_val['ground_truth'].values

    # use linear regression
    lr = LinearRegression()
    lr.fit(ground_truth.reshape(-1, 1), predicted_value.reshape(-1, 1))

    test_val = np.array([0, 1])
    test_pred = lr.predict(test_val.reshape(-1, 1))

    slope = test_pred[1] - test_pred[0]
    bias = test_pred[0]
    return slope[0], bias[0]


def two_steps_bias_correction(slope, bias, df_test, args):
    """two-stage approach: correction on predicted value to get unbiased prediction"""
    df_test['predicted_value'] = df_test['predicted_value'].apply(lambda x: (x - bias) / slope)
    df_test.to_csv(os.path.join(args.out_dir, "corrected_performance_summary.csv"))
