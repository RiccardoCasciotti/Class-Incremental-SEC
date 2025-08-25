import copy
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score

from cnn14_pann_lin import Cnn14
from cl_dataset_class import CL_dataset

# Function to calculate the weighting of the class losses based on a given ratio of class impact.
# If the KLD's impact needs to be increased, a negative value will accomplish this. This was implemented this way to maintain backwards compatability with old scripts.
def class_weight(class_impact: int):
    weight = 0
    if class_impact == 0:
        weight = 1
    elif class_impact < 0:
        weight = round(1 / ((-1*class_impact) + 1), 3)
    else:
        weight = round(class_impact / (class_impact + 1), 3)
    return weight

# Code modified from Pytorch's quickstart tutorial
def train(dataloader,
          model,
          old_model,
          loss_fn,
          optimizer,
          scheduler,
          log_interval,
          device_str,
          scaler,
          use_amp,
          use_kld,
          cil_nr_of_classes,
          T,
          class_impact,
          use_all_logits,
          use_cosine_kd):
    size = len(dataloader.dataset)
    nr_of_batches = len(dataloader)
    print(f"Number of batches: {nr_of_batches}")
    running_time = 0
    iterations = 0
    model.train()

    cls_w = class_weight(class_impact=class_impact)
    kld_w = 1 - cls_w
        
    kl_loss = 0
    if use_kld:
        kl_loss = nn.KLDivLoss(reduction='batchmean') # Math definition
        print(f"Ratio of class impact vs KLD impact in loss function: {cls_w} * (prediction loss) + {kld_w} * (KLD loss)")

    epoch_BCE_loss = 0
    epoch_KLD_loss = 0
    epoch_training_loss = 0
    
    for batch, (mel, label, fname) in enumerate(dataloader):
        start_time = time.time()

        with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
        
            mel, label = mel.to(device), label.to(device)
            
            # Compute prediction error, and for continuous learning use just the new labels.
            pred, conv_feats = model(mel)

            cil_pred = pred[:, -cil_nr_of_classes:]
            cil_label = label[:, -cil_nr_of_classes:]
            if use_all_logits:
                loss = cls_w * loss_fn(pred, label)
            else:
                loss = cls_w * loss_fn(cil_pred, cil_label)
            epoch_BCE_loss += loss.item()
            print(f"cil BCE loss: {loss}")
            print(f"Predictions: {cil_pred}")
            print(f"Prediction shape: {cil_pred.shape}")
            print(f"Actual: {cil_label}")
            print(f"Label shape: {cil_label.shape}")

            # Use the knowledgeable model's preds to help alleviate forgetfulness
            if use_kld:
                with torch.no_grad():
                    old_preds, _ = old_model(mel) # Target
                new_preds = pred[:, 0:old_model.get_output_dim()]
                kld_loss = kl_loss(F.log_softmax(new_preds/T, dim=1), F.softmax(old_preds/T, dim=1)) * (T**2)
                print(f"KLD loss: {kld_loss}")
                loss += kld_w * kld_loss
                epoch_KLD_loss += kld_loss.item()
            
            # Cosine similarity for feature maps. From CIL-ML-AUDIO
            if use_cosine_kd:
                with torch.no_grad():
                    old_preds_cos, old_conv_feats = old_model(mel)
                #logits_dist, cnnfeat_new = model(mel)
                cossim = nn.CosineSimilarity(dim=conv_feats.view(-1).dim() - 1)
                feat_loss = 1 - cossim(F.normalize(old_conv_feats.view(-1), p=2, dim=old_conv_feats.view(-1).dim() - 1),
                                        F.normalize(conv_feats.view(-1), p=2, dim=conv_feats.view(-1).dim() - 1))
                print(f"Feature loss: {feat_loss}")
                #sum_feat_loss += feat_loss.item()
                loss += feat_loss

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #loss.backward()
        #optimizer.step()
        optimizer.zero_grad()

        end_time = time.time()
        running_time += end_time - start_time
        iterations += 1
        
        if batch % log_interval == 0:
            print(f"Average batch training time: {running_time / iterations}", flush=True)
            training_loss, current = loss.item(), (batch + 1) * len(mel)
            epoch_training_loss += loss.item()
            print(f"Training loss: {training_loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)

    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]

    epoch_BCE_loss /= iterations
    epoch_KLD_loss /= iterations
    epoch_training_loss /= iterations

    print(f"Epoch BCE loss: {epoch_BCE_loss}")
    if kl_loss != 0:
        print(f"Epoch KLD loss for epoch: {epoch_KLD_loss}")
    print(f"Epoch training loss: {epoch_training_loss}")

    print(f"Learning rate before scheduler: {before_lr}", flush=True)
    print(f"Learning rate after scheduler: {after_lr}", flush=True)

def validate(dataloader,
             model,
             loss_fn,
             device,
             device_str,
             use_amp,
             cil_nr_of_classes):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    val_loss = 0
    cil_loss = 0
    memory_loss = 0

    with torch.no_grad():
        for mel, label, fname in dataloader:
            with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
                mel, label = mel.to(device), label.to(device)
                pred, _ = model(mel)

            val_loss += loss_fn(pred, label)
            cil_loss += loss_fn(pred[:, -cil_nr_of_classes:],
                                label[:, -cil_nr_of_classes:])
            memory_loss += loss_fn(pred[:, 0:-cil_nr_of_classes],
                                   label[:, 0:-cil_nr_of_classes])
    val_loss /= num_batches
    cil_loss /= num_batches
    memory_loss /= num_batches

    print(f"Avg cil loss through validation: {cil_loss}")
    print(f"Avg memory loss through validation: {memory_loss}")
    print(f"Avg loss through validation: {val_loss:>8f} \n", flush=True)
    return val_loss

def val_map(dataloader, 
           model, 
           device, 
           device_str,
           use_amp):
    
    model.eval()
    val_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch, (mel, label, fname) in enumerate(dataloader):
            
            with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
                mel, label = mel.to(device), label
                out, _ = model(mel.float())
                preds = torch.sigmoid(out)
            all_preds.extend(
                preds.cpu().numpy())
            all_targets.extend(np.asarray(label))

        Y_predicted = np.asarray(all_preds)
        Y_ref = np.asarray(all_targets)

        average_precision = average_precision_score(Y_ref, Y_predicted, average=None)
        print(f"Average precision: {average_precision}")

        mAp = np.mean(average_precision)
        cil_mAp = np.mean(average_precision[-cil_nr_of_classes:])
        init_30_mAp = np.mean(average_precision[0:-cil_nr_of_classes])

        # Higher mAp is good, but lower val_loss is also good
        val_loss = 1 - mAp
        print(f"Validation loss with 1 - mAp: {val_loss}", flush=True)

        # Diagnostics to see if the model's actually learning the classes at any point
        print(f"Whole mAp: {mAp}")
        print(f"Cil mAp: {cil_mAp}")
        print(f"Init 30 mAp: {init_30_mAp}")

    return val_loss

if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--nr_of_classes', type=int, choices=[30, 35, 40, 45, 50], help='Number of classes to use from data')
    parser.add_argument('--cil_nr_of_classes', type=int, default=0, help='Number of classes for class incremental learning. If not 0, the dataloader returns only new files.')
    parser.add_argument('--dataset', type=str, choices=['audioset', 'fsd50k'], help='Choice of dataset.')
    parser.add_argument('--path_to_data', type=str, help='The path to the HDF5 datafile.')
    parser.add_argument('--nr_of_workers', type=int, default=0, help='Number of workers for dataloading')
    parser.add_argument('--resume', action='store_true', help="Whether to resume from the latest saved checkpoint.")
    parser.add_argument('--batch_size', type=int, default=0, help='Size of the loaded data batch. A tensor of [batch_size, data_tensor.shape] is loaded.')
    parser.add_argument('--lr_start', type=float, default=0.1, help='Starting learning rate.')
    parser.add_argument('--lr_min', type=float, default=0.0001, help='End point of the learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum value for the SGD optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay for the SGD optimizer')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help="A value for how often a model's state is  saved in terms of epochs. I.e., for a value 2, the model's state is saved every 2 epochs.")
    parser.add_argument('--path_to_model_state', type=str, help='Location of the model state dict from which to initialize the cnn14 model.')
    parser.add_argument('--path_to_comparison_model_state', type=str, help='Location of the model which represents previously learned information.')
    parser.add_argument('--log_interval', type=int, default=1, help='How often to show some batch information e.g., average time taken, loss etc.')
    parser.add_argument('--use_amp', action='store_true', help='Whether to use Pytorch enabled automatic mixed precision.')
    parser.add_argument('--finetune_classifier', action='store_true', help='If set, only the final classifier layer of the model will be tuned.')
    parser.add_argument('--model_name', type=str, default='default')
    parser.add_argument('--use_kld', action='store_true', help='Whether to add KLD of current and comparison model to the loss in an effort to control forgetting.')
    parser.add_argument('--save_latest_epoch_model', action='store_true', help='If this flag is present, save the final epoch model state regardless of validation loss value.')
    parser.add_argument('--T', type=int, default=1, help='Temperature value for softmax in KLD.')
    parser.add_argument('--class_impact', type=int, default=1, help="Determines the impact of the class loss when counting loss during training. Anything above 1 raises the class loss's impact and diminishes KLD loss.")
    parser.add_argument('--validate_w_map', action='store_true', help='If used, the validation loss will look at the mean average precision score for when validating the model instead of the loss all classes.')
    parser.add_argument('--skip_training', action='store_true', help='If used, the training part of the train/validation loop is skipped. This was implemented for diagnostic purposes.')
    parser.add_argument('--use_all_logits', action='store_true', help="If used, don't constrain the logits to just the new classes during training. Very against the principles of class incremental learning, but useful in diagnosing performance.")
    parser.add_argument('--no_pos_weight', action='store_true', help='If present, BCEloss is used without compensating for class imbalance via pos_weight.')
    parser.add_argument('--no_cil_file_separation', action='store_true', help='If set, the dataloader wont load just the cil files but all files corresponding to nr_of_classes.')
    parser.add_argument('--use_cosine_kd', action='store_true', help='If set, the cosine similarity score of the feature maps between the old and new models will used in the loss computation.')

    args = vars(parser.parse_args())

    # Device selection
    if torch.cuda.is_available():
        device_str = 'cuda'
        device = torch.device('cuda')
    else:
        device_str = 'cpu'
        device = torch.device('cpu')
    print(f"Using device: {device}", flush=True)
    print(f"Using torch version: {torch.__version__}")

    setup_start_time = time.time()

    # Args
    epochs = args['epochs']
    nr_of_classes = args['nr_of_classes']
    cil_nr_of_classes = args['cil_nr_of_classes']
    dataset = args['dataset']
    PATH_TO_HDF5_DATA = args['path_to_data']
    nr_of_workers = args['nr_of_workers']
    resume = args['resume']
    batch_size = args['batch_size']
    lr_start = args['lr_start']
    lr_min = args['lr_min'] 
    momentum = args['momentum']
    weight_decay = args['weight_decay']
    checkpoint_interval = args['checkpoint_interval']
    log_interval = args['log_interval']
    PATH_TO_MODEL_STATE = args['path_to_model_state']
    PATH_TO_COMPARISON_MODEL_STATE = args['path_to_comparison_model_state']
    use_amp = args['use_amp']
    finetune_classifier = args['finetune_classifier']
    model_name = args['model_name']
    use_kld = args['use_kld']
    save_latest_epoch_model = args['save_latest_epoch_model']
    T = args['T']
    class_impact = args['class_impact']
    validate_w_map = args['validate_w_map']
    skip_training = args['skip_training']
    use_all_logits = args['use_all_logits']
    no_pos_weight = args['no_pos_weight']
    no_cil_file_separation = args['no_cil_file_separation']
    use_cosine_kd = args['use_cosine_kd']

    print(f"Starting model class incremental learning training with the following parameters:")
    print(args)

    # Data loading
    print(f"Fetching dataset.", flush=True)
    if no_cil_file_separation:
        data_train = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                                dataset=dataset,
                                split='train',
                                nr_of_classes=nr_of_classes)
    else:
        data_train = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                                dataset=dataset,
                                split='train',
                                nr_of_classes=nr_of_classes,
                                cil_classes=cil_nr_of_classes)

    print(f"There are {len(data_train)} training files in total. 1/10 will be used for validation.", flush=True)

    train_data, val_data = torch.utils.data.random_split(data_train, [0.9, 0.1])

    # For local testing
    smaller_train, smaller_val, _ = torch.utils.data.random_split(val_data, [0.09, 0.01, 0.9])
    small_train_loader = torch.utils.data.DataLoader(smaller_train, batch_size=batch_size, num_workers=nr_of_workers, shuffle=True)
    smaller_val_loader = torch.utils.data.DataLoader(smaller_val, batch_size=batch_size, num_workers=nr_of_workers)

    print(f"{len(train_data)} files will be used for training and {len(val_data)} will be used for validation.", flush=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=nr_of_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=nr_of_workers)

    data_setup_time_end = time.time()
    print(f"Data setup took {round(data_setup_time_end - setup_start_time, 2)} seconds")
    
    model = Cnn14(nr_of_classes)
    # Hacky and very specific, i.e works with nr_of_classes=35 and cil_classes=5, but will break with many other configurations.
    if no_cil_file_separation:
        model = Cnn14(nr_of_classes - cil_nr_of_classes)
    # Initiliaze from a base to ensure uniformity
    model.load_state_dict(torch.load(PATH_TO_MODEL_STATE, weights_only=True))
    if 'trained' in PATH_TO_MODEL_STATE:
        print(f"Initialized weights from an already trained model.")
    # Extend the model's classifier layer to match the additional classes
    model.change_output_dim(model.get_output_dim() + cil_nr_of_classes)
    print(f"Trainable model's classifier output dimension changed to: {model.get_output_dim()}")

    # If using kld, the old model is needed as well but only its inference
    if use_kld or use_cosine_kd:
        old_model = Cnn14(nr_of_classes)
        old_model.load_state_dict(torch.load(PATH_TO_COMPARISON_MODEL_STATE, 
                                             weights_only=True))
        old_model = old_model.to(device)
        old_model.eval()
        print(f"Initialized old model and set it to device:", flush=True)
    else:
        old_model = None

    # If finetuning just the final layer
    if finetune_classifier:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print(f"Training only the final classifier layer.")

    model = model.to(device)
    print(f"Created and initialize model and moved it to device.", flush=True)

    pos_weight = data_train.get_pos_weight()

    # TODO: if there's time compare performance without pos_weight
    if no_pos_weight:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn.to(device)

    # AdamW an option?
    # Use of weight decay copied from Manju's script
    optimizer = optim.SGD(model.parameters(), lr=lr_start, momentum=momentum,
                        weight_decay=weight_decay) 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)

    scaler = torch.amp.GradScaler(device=device_str, enabled=use_amp)

    print(f"Setup loss function, optimizer, scheduler, and scaler.", flush=True)

    # Resuming from latest checkpoint if the script did not go through successfully
    if resume == True:
        checkpoint_dict = torch.load('latest_chkp_dict.pth')

        model_dict = checkpoint_dict['model_state_dict']
        optim_dict = checkpoint_dict['optimizer_state_dict']
        sched_dict = checkpoint_dict['scheduler_state_dict']
        scale_dict = checkpoint_dict['scaler_state_dict']

        epoch = checkpoint_dict['epoch']
        epochs -= epoch
        classes = checkpoint_dict['nr_of_classes']
        data = checkpoint_dict['dataset']
        model.load_state_dict(model_dict)
        optimizer.load_state_dict(optim_dict)
        scheduler.load_state_dict(sched_dict)
        scaler.load_state_dict(scale_dict)

        print(f"Loaded the latest saved model checkpoint. The model was saved with the dataset: {data}, with a class number of: {classes}, and at epoch: {epoch}", flush=True)

    val_loss = 0
    best_val_loss = float('inf')
    old_val_loss = 0 # early stopping condition
    val_loss_thr = 0.0001
    patience_counter = 0
    patience_thr = 5

    best_model_state = {}
    final_model_state = {}

    setup_end_time = time.time()
    print(f"Time taken for setup: {round(setup_end_time - setup_start_time, 2)} seconds.", flush=True)

    # Actual training/validation loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"Entering epoch {epoch}/{epochs-1}.", flush=True)

        # Training
        if not skip_training:
            train(dataloader=train_loader, 
                model=model,
                old_model=old_model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                log_interval=log_interval,
                device_str=device_str,
                scaler=scaler,
                use_amp=use_amp,
                use_kld=use_kld,
                cil_nr_of_classes=cil_nr_of_classes,
                T=T,
                class_impact=class_impact,
                use_all_logits=use_all_logits,
                use_cosine_kd=use_cosine_kd)

        epoch_train_time = time.time()
        print(f"This epoch's training took {round(epoch_train_time-epoch_start_time, 2)}", flush=True)

        # Validation
        if validate_w_map:
            val_loss = val_map(dataloader=val_loader,
                               model=model,
                               device=device,
                               device_str=device_str,
                               use_amp=use_amp)
        else:
            val_loss = validate(dataloader=val_loader,
                            model=model,
                            loss_fn=loss_fn,
                            device=device,
                            device_str=device_str,
                            use_amp=use_amp,
                            cil_nr_of_classes=cil_nr_of_classes)

        epoch_val_time = time.time()
        print(f"This epoch's validation took {round(epoch_val_time-epoch_train_time, 2)}", flush=True)

        # Check for patience and early stopping
        if abs(old_val_loss - val_loss) < val_loss_thr:
            patience_counter += 1
            if patience_counter >= patience_thr:
                print(f"Validation loss had a change smaller than {val_loss_thr} {patience_thr} times. Stopping early.", flush=True)
                break
        old_val_loss = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model, deepcopy suggested by torch docs
            print(f"Saving the model state from epoch {epoch} as the best model state so far.", flush=True)
            best_model_state = copy.deepcopy(model.state_dict())
        
        if save_latest_epoch_model and (epoch+1) == epochs:
            print(f"Saving last epoch model state.")
            final_model_state = copy.deepcopy(model.state_dict())
            final_model_name = model_name.rstrip('.pt') + '_final.pt'
            torch.save(final_model_state, final_model_name)

        # Save the scheduler, optimizer, and model state periodically
        # Inspired partly by https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
        if epoch % checkpoint_interval == 0:
            print(f"Saving training state from epoch {epoch}.")
            torch.save({
                'epoch': epoch,
                'dataset': dataset,
                'nr_of_classes': nr_of_classes,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()                
            }, 'latest_chkp_dict.pth')

    # Save the best model for inference
    if model_name == 'default':
        model_name = 'trained_model_' + dataset + '_' + str(nr_of_classes) + '.pt'
    torch.save(best_model_state, model_name)
    print(f"Finished training for {epochs} epochs. Saved the model: {model_name}.")