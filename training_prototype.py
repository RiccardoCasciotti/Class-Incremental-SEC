import copy
import time

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim

import numpy as np

from cnn14_pann_lin import Cnn14
from cl_dataset_class import CL_dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

nr_of_classes = 30

PATH_TO_HDF5_DATA = r'C:\Users\mp431591\Documents\work_code\cl_30\continual_learning\data_cl.hdf5'

audioset_data_train = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                           dataset='audioset',
                           split='train',
                           nr_of_classes=30)

audioset_data_eval = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                                dataset='audioset',
                                split='eval',
                                nr_of_classes=30)
print(len(audioset_data_train))
print(len(audioset_data_eval))

train_data, val_data = torch.utils.data.random_split(audioset_data_train, [0.9, 0.1])
smaller_train_data, _, smaller_val_data  = torch.utils.data.random_split(val_data, [0.01, 0.98, 0.01])
print(len(train_data))
print(len(val_data))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=8)
validation_loader = torch.utils.data.DataLoader(val_data, batch_size=8)
evaluation_loader = torch.utils.data.DataLoader(audioset_data_eval, batch_size=8)

small_train_loader = torch.utils.data.DataLoader(smaller_train_data, batch_size=8)
small_val_loader = torch.utils.data.DataLoader(smaller_val_data, batch_size=8)

test_mel, test_label, _ = next(iter(evaluation_loader))
print(test_mel.shape)
print(test_label.shape)

# From CIL-ML-AUDIO
lr = 0.1
lr_min = 0.0001 
epochs = 120
momentum = 0.9
weight_decay = 5e-4

model = Cnn14(nr_of_classes)
model = model.to(device)



loss_fn = nn.BCEWithLogitsLoss() # Use of pos_weight?

# Use of weight decay copied from Manju's script
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                      weight_decay=weight_decay) 

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)

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

        
        if batch % 5 == 0:
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


if __name__ == '__main__':

    old_val_loss = 0 # Using val_loss as early stopping condition
    val_loss_thr = 0.0001

    for i in range(5):
        train(dataloader=small_train_loader, model=model,
        loss_fn=loss_fn, optimizer=optimizer,
        scheduler=scheduler)
        val_loss = validate(dataloader=small_val_loader, model=model,loss_fn=loss_fn)

        if abs(old_val_loss - val_loss) < val_loss_thr:
            print(f"Validation loss had a change smaller than 0.{val_loss_thr}. Stopping early.")
            break