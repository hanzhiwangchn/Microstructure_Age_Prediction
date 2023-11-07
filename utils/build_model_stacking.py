import torch
import numpy as np


def prepare_stacking_training_data(args, model, train_loader, val_loader, test_loader):
    """
    prepare stacking training data. 
    We need to load pre-trained models first. Then, based on the stacking method, 
    feed to training images into the models to get predictions, which will be used as the training data for stacking.
    """
    train_preds_list = []
    train_labels_list = []
    validation_preds_list = []
    validation_labels_list = []
    test_preds_list = []
    test_labels_list = []

    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(batch['image'])

            assert outputs.shape == batch['label'].shape
            assert len(outputs.shape) == 2

            train_preds_list.append(outputs)
            train_labels_list.append(batch['label'])

        # preds and labels will have shape (*, 1)
        train_preds = torch.cat(train_preds_list, 0)
        train_labels = torch.cat(train_labels_list, 0)
        assert train_preds.shape == train_labels.shape
        assert train_preds.shape[1] == 1
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(batch['image'])

            assert outputs.shape == batch['label'].shape
            assert len(outputs.shape) == 2

            validation_preds_list.append(outputs)
            validation_labels_list.append(batch['label'])

        # preds and labels will have shape (*, 1)
        val_preds = torch.cat(validation_preds_list, 0)
        val_labels = torch.cat(validation_labels_list, 0)
        assert val_preds.shape == val_labels.shape
        assert val_preds.shape[1] == 1

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(batch['image'])

            assert outputs.shape == batch['label'].shape
            assert len(outputs.shape) == 2

            test_preds_list.append(outputs)
            test_labels_list.append(batch['label'])

        # preds and labels will have shape (*, 1)
        test_preds = torch.cat(test_preds_list, 0)
        test_labels = torch.cat(test_labels_list, 0)
        assert test_preds.shape == test_labels.shape
        assert test_preds.shape[1] == 1

    np.save(f'{args.image_modality}_train_feature.npy', train_preds.cpu().numpy())
    np.save(f'{args.image_modality}_train_label.npy', train_labels.cpu().numpy())
    np.save(f'{args.image_modality}_val_feature.npy', val_preds.cpu().numpy())
    np.save(f'{args.image_modality}_val_label.npy', val_labels.cpu().numpy())
    np.save(f'{args.image_modality}_test_feature.npy', test_preds.cpu().numpy())
    np.save(f'{args.image_modality}_test_label.npy', test_labels.cpu().numpy())
