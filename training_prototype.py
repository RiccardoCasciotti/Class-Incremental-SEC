import copy
import time
import argparse

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, average_precision_score, f1_score
import numpy as np
from collections import Counter

from cnn14_pann_lin import Cnn14
from cl_dataset_class import CL_dataset

# Code modified from Pytorch's quickstart tutorial
def train(dataloader, model, loss_fn, optimizer, scheduler):
    size = len(dataloader.dataset)
    running_time = 0
    iterations = 0
    model.train()
    
    for batch, (mel, label, fname) in enumerate(dataloader):
        start_time = time.time()
        mel, label = mel.to(device), label.to(device)
        
        # Compute prediction error
        # Try to find out what the tuple's second member is supposed to be
        pred, _ = model(mel)
        loss = loss_fn(pred, label)
        #print(f"Predictions: {pred}")
        #print(f"Actual: {label}")

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        end_time = time.time()
        running_time += end_time - start_time
        iterations += 1
        
        if batch % 2 == 0:
            print(f"Average batch training time: {running_time / iterations}")
            loss, current = loss.item(), (batch + 1) * len(mel)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print(f"Learning rate before scheduler: {before_lr}")
    print(f"Learning rate after scheduler: {after_lr}")

def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for mel, label, fname in dataloader:
            mel, label = mel.to(device), label.to(device)
            pred, _ = model(mel)
            val_loss += loss_fn(pred, label).item()
    val_loss /= num_batches
    print(f"Avg loss through validation: {val_loss:>8f} \n")
    return val_loss

# Following CIL-ML-AUDIO's example
def evaluate(model, eval_loader, model_state_dict=None):
    if model_state_dict != None:
        model.load_state_dict(model_state_dict)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for mel, label, fname in eval_loader:
            out, _ = model(mel.float())
            preds = torch.gt(torch.sigmoid(out), 0.5)
            all_preds.extend(
                preds.cpu().numpy())
            all_targets.extend(np.asarray(label))

        Y_predicted = np.asarray(all_preds)
        Y_ref = np.asarray(all_targets)

        print(f"Predicted array: {Y_predicted} and its type {type(Y_predicted)}")
        print(f"Ground truth array: {Y_ref} and its type {type(Y_ref)}")


        print('Reference polyphony:', Counter(Y_ref.sum(axis=1)))
        print('Predicted polyphony:', Counter(Y_predicted.sum(axis=1)))
        print(classification_report(Y_ref, Y_predicted))

        average_precision = average_precision_score(Y_ref, Y_predicted, average=None)
        mAp = np.mean(average_precision)
        print('mAP', mAp)

        f1_macro = f1_score(Y_ref, Y_predicted, average='macro', zero_division=0.0)
        f1_micro = f1_score(Y_ref, Y_predicted, average='micro', zero_division=0.0)
        print('macro', f1_macro)
        print('micro', f1_micro)

        return mAp, f1_macro, f1_macro




if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('-nr', '--nr_of_classes', type=int, default=30, help='Number of classes to use from data')
    parser.add_argument('-d', '--dataset', type=str, default='audioset', help='Choice of dataset. Either audioset or fsd50k')
    parser.add_argument('-sp', '--split', type=str, default='train', help='Dataset split. Either train or eval')
    parser.add_argument('-pd', '--path_to_data', type=str, help='The path to the HDF5 datafile.')
    parser.add_argument('-nr_w', '--nr_of_workers', type=int, default=1, help='Number of workers for dataloading')
    args = vars(parser.parse_args())

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Dataloading
    nr_of_classes = 30
    dataset = 'audioset'
    PATH_TO_HDF5_DATA = r'C:\Users\mp431591\Documents\work_code\cl_30\continual_learning\data_cl.hdf5'

    data_train = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                            dataset=dataset,
                            split='train',
                            nr_of_classes=30)

    data_eval = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                                    dataset='audioset',
                                    split='eval',
                                    nr_of_classes=30)
    print(len(data_train))
    print(len(data_eval))

    train_data, val_data = torch.utils.data.random_split(data_train, [0.9, 0.1])
    smaller_train_data, _, smaller_val_data  = torch.utils.data.random_split(val_data, [0.01, 0.98, 0.01])
    print(len(train_data))
    print(len(val_data))

    small_eval_data, _ = torch.utils.data.random_split(data_eval,
                                                    [0.001, 0.999])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8)
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=8)
    evaluation_loader = torch.utils.data.DataLoader(data_eval, batch_size=8)

    small_train_loader = torch.utils.data.DataLoader(smaller_train_data, batch_size=8)
    small_val_loader = torch.utils.data.DataLoader(smaller_val_data, batch_size=8)
    small_eval_loader = torch.utils.data.DataLoader(small_eval_data, batch_size=8)

    test_mel, test_label, _ = next(iter(evaluation_loader))
    print(test_mel.shape)
    print(test_label.shape)

    # From CIL-ML-AUDIO
    lr = 0.1
    lr_min = 0.0001 
    epochs = 10
    momentum = 0.9
    weight_decay = 5e-4

    model = Cnn14(nr_of_classes)
    model = model.to(device)

    pos_weight = data_train.get_pos_weight()
    print(f"Pos weight type: {type(pos_weight)}")

    loss_fn = nn.BCEWithLogitsLoss() # Use of pos_weight?
    loss_fn_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Use of weight decay copied from Manju's script
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=weight_decay) 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)

    val_loss = 0
    best_val_loss = float('inf')
    old_val_loss = 0 # Using val_loss as early stopping condition
    val_loss_thr = 0.0001
    patience_counter = 0
    patience_thr = 5

    best_model_state = {}


    for epoch in range(epochs):

        # Training
        train(dataloader=small_train_loader, model=model,
        loss_fn=loss_fn_weighted, optimizer=optimizer,
        scheduler=scheduler)

        # Validation
        val_loss = validate(dataloader=small_val_loader, model=model,loss_fn=loss_fn_weighted)

        # Check for patience and early stopping
        if abs(old_val_loss - val_loss) < val_loss_thr:
            patience_counter += 1
            if patience_counter >= patience_thr:
                print(f"Validation loss had a change smaller than {val_loss_thr} {patience_thr} times. Stopping early.")
                break
        old_val_loss = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model
            best_model_state = copy.deepcopy(model.state_dict())

        # Save the scheduler, optimizer, and model state periodically
        # Inspired partly by https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
        if epoch % 5 == 0:
            print("Saving training state from epoch {epoch}.")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, 'latest_chkp_dict.pth')

    
    evaluate(model=model, eval_loader=small_eval_loader, model_state_dict=best_model_state)

    # Save the model for inference
    model_name = 'trained_model_' + dataset + '_' + str(nr_of_classes) + '.pt'
    torch.save(best_model_state, model_name)