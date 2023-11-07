import torch, math, logging, os
from transformers import get_scheduler
import evaluate
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.common_utils import calculate_correlation
from utils.build_loss_function import build_loss_function
from utils.build_model import build_model_test
from utils.build_two_stage_correction import apply_two_stage_correction

logger = logging.getLogger(__name__)


# ------------------- build data loader ---------------------
def build_loader(args, dataset_train, dataset_val, dataset_test):
    """main function for dataloader building"""
    # build data-loader configurations
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    validation_kwargs = {'batch_size': args.batch_size, 'shuffle': False}
    test_kwargs = {'batch_size': args.batch_size, 'shuffle': False}
    if torch.cuda.is_available():
        cuda_kwargs = {'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # initialize loader
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(dataset_val, **validation_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    return train_loader, val_loader, test_loader


# ------------------- build optimizer ---------------------
def build_optimizer(model, train_loader, args):
    """build optimizer and learning rate scheduler"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    return optimizer, lr_scheduler


# ------------------- build main training loop ---------------------
def train_val_test(args, train_loader, val_loader, test_loader, model, optimizer, 
        lr_scheduler, m, loss_fn_train):
    """train/val/test for pytorch loop"""
    # variable for compact dynamic lambda
    lambda_correlation_list = []

    if not args.test:
        best_loss = 100

        # train and evaluate
        # We move RunManager track loss to evaluate package using evaluate.load('mae')
        for epoch in range(args.num_train_epochs):
            m.begin_epoch()
            train(args, model, m, train_loader, optimizer, lr_scheduler, loss_fn_train)
            val(args, model, m, val_loader)

            # calculate correlation on train/validation/test set
            calculate_correlation(args, model, m, train_loader, val_loader, test_loader)

            # dynamic lambda algorithm
            if epoch in range(args.update_lambda_start_epoch, args.num_train_epochs+1, args.compact_update_interval) \
                    and args.compact_dynamic:
                args, lambda_correlation_list = update_lamda_max(args, m, epoch, lambda_correlation_list)
                # initialize new skewed loss function based on new lamda_max
                loss_fn_train, _, _ = build_loss_function(args)

            # save the model with the best validation loss
            if args.save_best and epoch >= args.save_best_start_epoch:
                if args.skewed_loss:
                    # adding correlation threshold as another metric when selecting the best model
                    if (m.epoch_stats['val_mae'] < best_loss) & \
                            (abs(m.epoch_stats['val_correlation']) <= args.acceptable_correlation_threshold):
                        logger.info(f'Acceptable and lower validation loss found at epoch {m.epoch_num_count}')
                        best_loss = m.epoch_stats['val_mae']
                        torch.save(model.state_dict(), os.path.join(args.out_dir, "Best_Model.pt"))

                else:
                    if m.epoch_stats['val_mae'] < best_loss:
                        logger.info(f'Lower validation loss found at epoch {m.epoch_num_count}')
                        best_loss = m.epoch_stats['val_mae']
                        torch.save(model.state_dict(), os.path.join(args.out_dir, "Best_Model.pt"))
            
            m.end_epoch()
            m.display_epoch_results()
    
    # testing
    model = evaluate_test_set_performance(args, test_loader, m)

    m.end_run()
    # save stats
    m.save(os.path.join(args.out_dir, f'runtime_stats'))

    # apply two-stage correction approach when using normal loss
    if args.two_stage_correction:
        apply_two_stage_correction(args, val_loader, model)


def train(args, model, m, train_loader, optimizer, lr_scheduler, loss_fn_train):
    """training part"""
    all_metrics, all_metrics_results = dict(), dict()
    all_metric_type = ["mae"]
    for metric in all_metric_type:
        all_metrics[metric] = evaluate.load(metric)

    model.train()
    for batch in train_loader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        outputs = model(batch['image'])

        assert outputs.shape == batch['label'].shape
        assert len(outputs.shape) == 2

        loss = loss_fn_train(outputs, batch['label'])
        # calculate loss again using validation loss function. It is used to detect over-fitting
        for metric in all_metric_type:
            all_metrics[metric].add_batch(predictions=outputs, references=batch["label"])
        # loss should be a tensor of a single value
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # track train loss
        m.track_train_loss(loss=loss)
    
    for metric in all_metric_type:
        if metric in ["recall", "precision", "f1"]:
            all_metrics_results.update(all_metrics[metric].compute(average='weighted'))
        else:
            all_metrics_results.update(all_metrics[metric].compute())

    # track val metrics
    m.collect_train_metrics(metric_results=all_metrics_results)


def val(args, model, m, val_loader):
    """validation part"""
    all_metrics, all_metrics_results = dict(), dict()
    all_metric_type = ["mae"]
    for metric in all_metric_type:
        all_metrics[metric] = evaluate.load(metric)

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(batch['image'])

            assert outputs.shape == batch['label'].shape
            assert len(outputs.shape) == 2

            for metric in all_metric_type:
                all_metrics[metric].add_batch(predictions=outputs, references=batch["label"])

    for metric in all_metric_type:
        if metric in ["recall", "precision", "f1"]:
            all_metrics_results.update(all_metrics[metric].compute(average='weighted'))
        else:
            all_metrics_results.update(all_metrics[metric].compute())
    
    # track val metrics
    m.collect_val_metrics(metric_results=all_metrics_results)


def evaluate_test_set_performance(args, test_loader, m):
    """evaluate performance here, we need to load the best model"""
    all_metrics, all_metrics_results = dict(), dict()
    all_metric_type = ['mae']
    for metric in all_metric_type:
        all_metrics[metric] = evaluate.load(metric)

    # load best model
    model = build_model_test(args=args)
    model.eval()

    # save results to csv file
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(batch['image'])

            assert outputs.shape == batch['label'].shape
            assert len(outputs.shape) == 2

            for metric in all_metric_type:
                all_metrics[metric].add_batch(predictions=outputs, references=batch["label"])

            preds_list.append(outputs)
            labels_list.append(batch['label'])
        
        # preds and labels will have shape (*, 1)
        preds_tensor = torch.cat(preds_list, 0)
        labels_tensor = torch.cat(labels_list, 0)

    for metric in all_metric_type:
        if metric in ["recall", "precision", "f1"]:
            all_metrics_results.update(all_metrics[metric].compute(average='weighted'))
        else:
            all_metrics_results.update(all_metrics[metric].compute())
    
    # track test metrics
    m.collect_test_metrics(metric_results=all_metrics_results)

    df_save = pd.DataFrame()
    df_save['predicted_value'] = preds_tensor.squeeze().cpu().numpy()
    df_save['ground_truth'] = labels_tensor.squeeze().cpu().numpy()
    df_save.to_csv(os.path.join(args.out_dir, "performance_summary.csv"))

    return model


def moving_average(a, n=3):
    """moving average function"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def update_lamda_max(args, m, epoch, lambda_correlation_list):
    """update lambda value based on correlations"""
    # A moving average function is applied because correlation has wild oscillation.
    # We further select the median value to represent the trends of the correlation
    corr_median = np.median(moving_average(m.run_correlation_val[-1 * args.compact_update_interval:], n=3))
    logger.info(f'median averaged correlation is {corr_median}')
    temp_lambda_corr_pair = [args.init_lambda, corr_median]

    lambda_correlation_list.append(temp_lambda_corr_pair)
    # lambda_correlation_list only keeps the last 10 pairs of results for updating lambda
    if len(lambda_correlation_list) > 10:
        lambda_correlation_list.pop(0)

    # At the 1st phase of dynamic lambda, we use a naive approach to tune lambda to get correlations
    # for different lambda values.
    # At the 2nd phase, we apply linear regression to find the optimal lambda value
    if epoch >= args.update_lambda_second_phase_start_epoch:
        args.init_lambda = find_optimal_lambda(lambda_correlation_list)
    else:
        # Instead of using a single multiplier, we assign small changes on it to improve stability
        if corr_median < -0.1:
            args.init_lambda = args.init_lambda * torch.normal(mean=torch.tensor([args.compact_init_multiplier]),
                                                               std=torch.tensor([0.05])).item()
        elif corr_median > 0.1:
            args.init_lambda = args.init_lambda / torch.normal(mean=torch.tensor([args.compact_init_multiplier]),
                                                               std=torch.tensor([0.05])).item()

    logger.info(f'updated lambda at epoch:{epoch} is {args.init_lambda}')
    return args, lambda_correlation_list


def find_optimal_lambda(lambda_correlation_list):
    """find the best lambda value to make correlation move towards zero using LR"""
    lambda_correlation_array = np.array(lambda_correlation_list)
    lambda_val = lambda_correlation_array[:, 0]
    correlation = lambda_correlation_array[:, 1]

    # use linear regression as a start
    lr = LinearRegression()
    lr.fit(lambda_val.reshape(-1, 1), correlation.reshape(-1, 1))

    test_lambda_val = np.array([0, 1])
    test_correlation_pred = lr.predict(test_lambda_val.reshape(-1, 1))
    # get the optimal value
    slope = test_correlation_pred[1] - test_correlation_pred[0]
    bias = test_correlation_pred[0]
    # if slope becomes zero, it means the dots are on a horizontal line,
    # which will result in a much larger lambda.
    if abs(slope[0]) < 1e-2:
        opt_lambda = np.mean(lambda_val)
    else:
        opt_lambda = -1 * bias[0] / slope[0]

    # lambda should not be a negative value
    # stability improvement
    if opt_lambda < 0:
        opt_lambda = 0.0
    if opt_lambda > 20:
        opt_lambda = 20.0

    logger.info(f'slope of lr is {slope}; bias of lr is {bias}')
    logger.info(f'optimal lambda is {opt_lambda}')
    return opt_lambda
