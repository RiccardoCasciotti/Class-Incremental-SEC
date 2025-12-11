import copy
import datetime
import random
import sys
import time
import argparse
import os
import uuid

from rank_PANNs_CNN14_filters import calculate_filter_scores
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score
import json

from cnn14_pann_lin import Cnn14
from cl_dataset_class import Audioset, Fsd50k, MID2ID_Audioset, MID2ID_Fsd50k
from extra.partial_freezing import freeze_conv2d_params
from log_class import Log
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
          device,
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
    # print(f"Number of batches: {nr_of_batches}")
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
    
    for batch, (mel, label) in enumerate(dataloader):
        start_time = time.time()

        with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
        
            mel, label = mel.to(device), label.to(device)
            
            # Compute prediction error, and for continuous learning use just the new labels.
            pred, conv_feats = model(mel)
        

            # print(f"Predictions: {cil_pred}")
            # print(f"Prediction shape: {pred.shape}")
            # print(f"Actual: {cil_label}")
            # print(f"Label shape: {label.shape}")

            cil_pred = pred[:, -cil_nr_of_classes:]
            # cil_label = label[:, -cil_nr_of_classes:]
            cil_label = label
            if use_all_logits:
                loss = cls_w * loss_fn(pred, label)
            else:
                loss = cls_w * loss_fn(cil_pred, cil_label)
            epoch_BCE_loss += loss.item()
            # print(f"cil BCE loss: {loss}")
            # print(f"Predictions: {cil_pred}")
            # print(f"Prediction shape: {cil_pred.shape}")
            # print(f"Actual: {cil_label}")
            # print(f"Label shape: {cil_label.shape}")

            # Use the knowledgeable model's preds to help alleviate forgetfulness
            # if use_kld:
            #     with torch.no_grad():
            #         old_preds, _ = old_model(mel) # Target
            #     new_preds = pred[:, 0:old_model.get_output_dim()]
            #     kld_loss = kl_loss(F.log_softmax(new_preds/T, dim=1), F.softmax(old_preds/T, dim=1)) * (T**2)
            #     print(f"KLD loss: {kld_loss}")
            #     loss += kld_w * kld_loss
            #     epoch_KLD_loss += kld_loss.item()
            
            # Cosine similarity for feature maps. From CIL-ML-AUDIO
            # if use_cosine_kd:
            #     with torch.no_grad():
            #         old_preds_cos, old_conv_feats = old_model(mel)
            #     #logits_dist, cnnfeat_new = model(mel)
            #     cossim = nn.CosineSimilarity(dim=conv_feats.view(-1).dim() - 1)
            #     feat_loss = 1 - cossim(F.normalize(old_conv_feats.view(-1), p=2, dim=old_conv_feats.view(-1).dim() - 1),
            #                             F.normalize(conv_feats.view(-1), p=2, dim=conv_feats.view(-1).dim() - 1))
            #     print(f"Feature loss: {feat_loss}")
            #     #sum_feat_loss += feat_loss.item()
            #     loss += feat_loss

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
            # print(f"Average batch training time: {running_time / iterations}", flush=True)
            training_loss, current = loss.item(), (batch + 1) * len(mel)
            epoch_training_loss += loss.item()
            # print(f"Training loss: {training_loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)

    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]

    epoch_BCE_loss /= iterations
    epoch_KLD_loss /= iterations
    epoch_training_loss /= iterations

    # print(f"Epoch BCE loss: {epoch_BCE_loss}")
    # if kl_loss != 0:
    #     print(f"Epoch KLD loss for epoch: {epoch_KLD_loss}")
    # print(f"Epoch training loss: {epoch_training_loss}")

    # print(f"Learning rate before scheduler: {before_lr}", flush=True)
    # print(f"Learning rate after scheduler: {after_lr}", flush=True)
    return epoch_BCE_loss
def validate(dataloader,
             model,
             loss_fn,
             device_str,
             device,
             use_amp,
             cil_nr_of_classes):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    val_loss = 0
    cil_loss = 0
    memory_loss = 0

    with torch.no_grad():
        

        for data in iter(dataloader):
            
            with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
                mel, label = data
                mel, label = mel.to(device), label.to(device)
                pred, _ = model(mel)

            val_loss += loss_fn(pred[:, -cil_nr_of_classes:], label)
            cil_loss += loss_fn(pred[:, -cil_nr_of_classes:],
                                label[:, -cil_nr_of_classes:])
            # memory_loss += loss_fn(pred[:, 0:-cil_nr_of_classes],
            #                        label[:, 0:-cil_nr_of_classes])
    val_loss /= num_batches
    cil_loss /= num_batches
    # memory_loss /= num_batches

    print(f"Avg cil loss through validation: {cil_loss}")
    # print(f"Avg memory loss through validation: {memory_loss}")
    print(f"Avg loss through validation: {val_loss:>8f} \n", flush=True)
    return val_loss

def val_map(dataloader, 
           model, 
           device, 
           device_str,
           use_amp,
           cil_nr_of_classes):
    
    model.eval()
    val_loss = 0

    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch, (mel, label) in enumerate(dataloader):
            
            with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
                mel, label = mel.to(device), label
                out, _ = model(mel.float())
                preds = torch.sigmoid(out)[:, -cil_nr_of_classes:]
            all_preds.extend(
                preds.cpu().numpy())
            all_targets.extend(np.asarray(label))
        
        Y_predicted = np.asarray(all_preds)
        Y_ref = np.asarray(all_targets)
        
        average_precision = average_precision_score(Y_ref, Y_predicted, average=None)
        # print(f"Average precision: {average_precision}")
        print("SHAPES: ", preds.cpu().shape, Y_predicted.shape, Y_ref.shape, average_precision.shape, average_precision[-cil_nr_of_classes:].shape)
        mAp = np.mean(average_precision)
        cil_mAp = np.mean(average_precision[-cil_nr_of_classes:])

        # Higher mAp is good, but lower val_loss is also good
        val_loss = 1 - mAp
        # print(f"Validation loss with 1 - mAp: {val_loss}", flush=True)

        # Diagnostics to see if the model's actually learning the classes at any point
        # print(f"Whole mAp: {mAp}")
        # print(f"Cil mAp: {cil_mAp}")

    return mAp, val_loss

def model_setup(task, args, device):
    
    model_params = args["model_params"]
    exp_params = args["experiment_params"]
    model = Cnn14(exp_params["classes_per_task"][task])
    # Initiliaze from a base to ensure uniformity
    if task > 0:
        old_dim = np.asarray(exp_params["classes_per_task"][:task]).sum()
        model = Cnn14(old_dim)
        sd = torch.load(model_params['model_state_dest'] + f"/model_T{task-1}/model_T{task-1}.pt")
        model.load_state_dict(sd)
        # Extend the model's classifier layer to match the additional classes
        model.change_output_dim(old_dim + exp_params["classes_per_task"][task])
        print(f"Trainable model's classifier output dimension changed to: {model.get_output_dim()}")

    old_model = None
    if model_params["use_kld"] or model_params["use_cosine_kd"]:
        old_dim = np.asarray(exp_params["classes_per_task"][:task]).sum() # ASK MATIAS
        old_model = Cnn14(old_dim)
        old_model.load_state_dict(torch.load(model_params['model_state_dest']+ f"/model_T{task-1}/model_T{task-1}.pt", 
                                             weights_only=True))
        old_model = old_model.to(device)
        old_model.eval()
        print(f"Initialized old model and set it to device:", flush=True)
    

    # If finetuning just the final layer
    if model_params['finetune_classifier']:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print(f"Training only the final classifier layer.")

    model = model.to(device)
    print(f"Created and initialize model and moved it to device.", flush=True)

    # Read in the filter indices per layer and take the desired ones
    if task > 0:
        with os.scandir(model_params['model_state_dest'] + f"/model_T{task-1}/filters_scores/") as it:
            print("path: ", model_params['model_state_dest'] + f"/model_T{task-1}/filters_scores/")
            layer_idx_to_filters = {}
            if it == []:
                print(f"WARNING: filters from task {task-1} are missing, aborting the training.")
                return
            for tmp_file in it:
                layer_index = int(tmp_file.name.lstrip('sim_index').rstrip('.npy'))
                # Creating the indices uses np.argsort which by default returns indices in ascending order -> descending desired
                filter_indices = np.load(tmp_file.path)[::-1]
                # Partial freezing needs a list of indices
                desired_filters = list(filter_indices[np.arange(0, int((model_params["filter_nr"]/8)*len(filter_indices)))])

                layer_idx_to_filters[layer_index] = desired_filters
        # Freeze the first 6 convolutional layers
        model.conv_block1.conv1.weight.requires_grad = False
        model.conv_block1.conv2.weight.requires_grad = False
        model.conv_block2.conv1.weight.requires_grad = False
        model.conv_block2.conv2.weight.requires_grad = False
        model.conv_block3.conv1.weight.requires_grad = False
        model.conv_block3.conv2.weight.requires_grad = False
        # file number connects to the layer number
        conv_layer_idx = 1
        layer_idx_to_module = {}
        for name, module in model.named_modules():
            if 'conv1' in name or 'conv2' in name:
                layer_idx_to_module[conv_layer_idx] = module
                conv_layer_idx += 1
                print("conv_layer_idx: ", conv_layer_idx)
        print("Counted the convolution layers: ", list(layer_idx_to_module.keys()))

        # Freeze the desired filters of the final 6 (for now) layers
        if task > 0:
            for idx in range(7, 13):
                indices = layer_idx_to_filters[idx]
                module = layer_idx_to_module[idx]
                freeze_conv2d_params(layer=module, weight_indices=indices)
            print(f"Chosen layer filters from chosen layers should be effectively frozen now.")

                # print("FOR: ", layer_index, filter_indices, desired_filters)

            # print(f"Read in the filter ranks: ", list(layer_idx_to_filters.keys()))

    
    
    

    return model, old_model

def train_setup(args, task, dataset, device_str, device, setup_start_time, logger=None):
    # Args
    model_params = args["model_params"]
    dataset_params = args["dataset_params"]
    train_params = args["train_params"]

    # training parameters

    epochs = train_params['epochs']
    nr_of_workers = train_params['nr_of_workers']
    resume = train_params['resume']
    batch_size = train_params['batch_size']
    checkpoint_interval = train_params['checkpoint_interval']
    log_interval = train_params['log_interval']
    save_latest_epoch_model = train_params['save_latest_epoch_model']
    skip_training = train_params['skip_training']

    # dataset parameters

    dataset_name = dataset_params['dataset']

    # model parameters

    lr_start = model_params['lr_start']
    lr_min = model_params['lr_min'] 
    momentum = model_params['momentum']
    weight_decay = model_params['weight_decay']
    PATH_TO_MODEL_STATE = model_params['model_state_dest']
    use_amp = model_params['use_amp']
    model_name = model_params['model_name']
    use_kld = model_params['use_kld']
    T = model_params['T']
    class_impact = model_params['class_impact']
    validate_w_map = model_params['validate_w_map']
    use_all_logits = model_params['use_all_logits']
    no_pos_weight = model_params['no_pos_weight']
    use_cosine_kd = model_params['use_cosine_kd']
    use_cls_specific_pos_weight = model_params['use_cls_specific_pos_weight']
    use_cls_specific_pos_weight_input_data_only = model_params['use_cls_specific_pos_weight_input_data_only']
    

    print(f"Starting model class incremental learning training with the following parameters:")
    print(args)

    # Data loading
    print(dataset.__getitem__(0)[0])
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_val])
    # train_data, val_data = torch.utils.data.random_split(dataset, [0.9, 0.1])
    print(train_data.__getitem__(0)[0])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=nr_of_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=nr_of_workers)
    
    
    

    print("Setting up the model... ", end="")

    model, old_model = model_setup(task, args, device_str)    

    print("Done")

    print("Setup loss function, optimizer, scheduler, and scaler... ", end="")
    
    if no_pos_weight:
        loss_fn = nn.BCEWithLogitsLoss()
    # cls_specific pos weights should only be used with validate_w_map flag since the complementary logic of using partial loss during training and full loss during validation hasn't been implemented.
    elif use_cls_specific_pos_weight:
        cls_pos_weight = dataset.pos_weights
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)
    elif use_cls_specific_pos_weight_input_data_only:
        cls_pos_weight_input_only = dataset.pos_weights
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight_input_only)
    else:
        pos_weight = dataset.pos_weights
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn=loss_fn.to(device_str)

    # AdamW an option?
    # Use of weight decay copied from Manju's script
    # Weight decay may not be used with partial layer freezing since the authors report it possibly making the freezing partly ineffective
    # Gradient vanishing/exploding happens regardless of if weight_decay is used. Leaving this as is here, but the rank_filt scripts themselves specify weight_decay=0
    optimizer = optim.SGD(model.parameters(), lr=lr_start, momentum=momentum,
                          weight_decay=weight_decay) 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)

    scaler = torch.amp.GradScaler(device=device_str, enabled=use_amp)
    print("Done")

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
    if logger is not None:
        logger.set_current_task(task)

    setup_end_time = time.time()
    print(f"Time taken for setup: {round(setup_end_time - setup_start_time, 2)} seconds.", flush=True)

    cil_nr_of_classes = args["experiment_params"]["classes_per_task"][task]
    epoch_start_time = time.time()
    # Actual training/validation loop

    for epoch in range(epochs):
        epoch_print = ""
        
        # print(f"Entering epoch {epoch}/{epochs-1}.", flush=True)
        
        # Training
        if not skip_training:
            train_loss = train(dataloader=train_loader, 
                model=model,
                old_model=old_model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                log_interval=log_interval,
                device_str=device_str,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                use_kld=use_kld,
                cil_nr_of_classes=cil_nr_of_classes,
                T=T,
                class_impact=class_impact,
                use_all_logits=use_all_logits,
                use_cosine_kd=use_cosine_kd)

        epoch_train_time = time.time()
        # print(f"This epoch's training took {round(epoch_train_time-epoch_start_time, 2)}", flush=True)
        
        # Validation
        if validate_w_map:
            validation_mAp, val_loss = val_map(dataloader=val_loader,
                               model=model,
                               device=device,
                               device_str=device_str,
                               use_amp=use_amp,
                               cil_nr_of_classes=cil_nr_of_classes
                               )
            
        else:
            val_loss = validate(dataloader=val_loader,
                            model=model,
                            loss_fn=loss_fn,
                            device=device,
                            device_str=device_str,
                            use_amp=use_amp,
                            cil_nr_of_classes=cil_nr_of_classes)

        # print(f"This epoch's validation took {round(epoch_val_time-epoch_train_time, 2)}", flush=True)
        
        # Check for patience and early stopping
        if epoch%log_interval == 0:
            
            epoch_print += f"Epoch {epoch}/{epochs} - "
            epoch_print += f"Time {round(epoch_train_time-epoch_start_time, 2)} s - "
            epoch_start_time = time.time()
            epoch_print += f"Train loss: {train_loss} - "
            epoch_print += f"Val mAp: {validation_mAp}, Val Loss: {val_loss} - "
            print(epoch_print)
            if logger is not None:
                logger.set_mAp(validation_mAp, res_type="val")
                logger.set_loss(train_loss, res_type="train")
                logger.set_loss(val_loss, res_type="val")

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
            
            print(f"Saving last epoch model state to custom destination.")
            final_model_state = copy.deepcopy(model.state_dict())
            final_model_name = model_name.rstrip('.pt') + '_final.pt'
            final_model_name = f"{PATH_TO_MODEL_STATE}/modelT_{task}_final.pt"
            torch.save(final_model_state, final_model_name)

        # Save the scheduler, optimizer, and model state periodically
        # Inspired partly by https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
        if epoch % checkpoint_interval == 0:
            print(f"Saving training state from epoch {epoch}.")
            torch.save({
                'epoch': epoch,
                'dataset': "dataset_name",
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()                
            }, 'latest_chkp_dict.pth')
    # Default name if model name isn't specified
    if model_name == '':
        model_name = f'trained_model_{dataset }_T{task}.pt'
    # Save the best model for inference
    
    model_name = f"{PATH_TO_MODEL_STATE}/model_T{task}/model_T{task}.pt"
    torch.save(best_model_state, model_name)
    nr_of_classes = np.asarray(args["experiment_params"]["classes_per_task"][:task+1]).sum().item()
    calculate_filter_scores(model_name , nr_of_classes, f"{args['model_params']['model_state_dest']}/model_T{task}/filters_scores")

    print(f"Saved the best model to custom destination.")
    print(f"Finished training modelT_{task} for {epochs} epochs.")

    if logger is not None: 
        logger.log_task_on_file(task)


def path_setup(args):
    exp_params = args["experiment_params"]
    if not os.path.exists(f"{exp_params['experiment_save_path']}/experiments"):
        os.makedirs(f"{exp_params['experiment_save_path']}/experiments", exist_ok=True)
        print(f"Created directory: {exp_params['experiment_save_path']}/experiments")
    exp_params['experiment_save_path'] = f"{exp_params['experiment_save_path']}/experiments"

    model_params = args["model_params"]
    #### TO BE FIXED
    model_name =  f"model_{model_params['model_name']}"

    if not os.path.exists(f"{exp_params['experiment_save_path']}/{model_name}"):
        os.makedirs(f"{exp_params['experiment_save_path']}/{model_name}", exist_ok=True)
        print(f"Created directory: {exp_params['experiment_save_path']}/{model_name}")

    exp_params['experiment_save_path'] = f"{exp_params['experiment_save_path']}/{model_name}"

    if not os.path.exists(f"{exp_params['experiment_save_path']}/config.json"):
        with open(f"{exp_params['experiment_save_path']}/config.json", "w") as f:
            json.dump(args, f, indent=4)
        print(f"Created config file: {exp_params['experiment_save_path']}/config.json") 

    if not os.path.exists(f"{exp_params['experiment_save_path']}/runs"):
        os.makedirs(f"{exp_params['experiment_save_path']}/runs", exist_ok=True)
        print(f"Created directory: {exp_params['experiment_save_path']}/runs")
    exp_params['experiment_save_path'] = f"{exp_params['experiment_save_path']}/runs"

    exp_id = os.getenv('SLURM_ARRAY_TASK_ID')
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    # random suffix
    rnd = uuid.uuid4().hex[:8]
    random_id = str(ts)+str(rnd)

    if exp_id is None:
        print("Not running as part of a job array; no SLURM_ARRAY_TASK_ID available, setting a random UUID.")
        exp_id=random_id
    
    if not os.path.exists(f"{exp_params['experiment_save_path']}/{exp_id}"):
        os.makedirs(f"{exp_params['experiment_save_path']}/{exp_id}")
    else:
        print(f"WARNING: experiment with ID: {exp_id} already exists! Creating a new one with random ID!\n")
        exp_id = exp_id + "_" + random_id
        os.makedirs(f"{exp_params['experiment_save_path']}/{exp_id}")
        print(f"Created directory: {exp_params['experiment_save_path']}/{exp_id}")
    exp_params['experiment_save_path'] = f"{exp_params['experiment_save_path']}/{exp_id}"

    if not os.path.exists(f"{exp_params['experiment_save_path']}/models"):
        os.makedirs(f"{exp_params['experiment_save_path']}/models")
        print(f"Created directory: {exp_params['experiment_save_path']}/models")
    model_params['model_state_dest'] = f"{exp_params['experiment_save_path']}/models"
    args["model_params"] = model_params
    args["experiment_params"] = exp_params
    return args

def fetch_datasets(args, dataset_class, dataset, ID2LABEL, classes_per_task, debug_test):
    selected_classes = []
    selected_classes_tot = random.sample(list(ID2LABEL.keys()), np.array(classes_per_task).sum().item())
    for i in range(len(classes_per_task)): 
        prev_slice = i if i == 0 else prev_slice
        slice = classes_per_task[i] + prev_slice
        selected_classes.append(selected_classes_tot[prev_slice:slice])
        prev_slice += classes_per_task[i]


    datasets = []
    data_path = f'{args["dataset_params"]["base_path"]}/{dataset}'
    print(selected_classes)
    for i, elem in enumerate(selected_classes):
        datasets.append(dataset_class(data_path=data_path, selected_classes=selected_classes[i], test=False, debug=debug_test))

    return datasets

def main(input):
    setup_start_time = time.time()
    with open(input["config"], 'r') as f:
        args = json.load(f)
    print("Starting experiment...", flush=True)
    args["model_params"]["model_name"] = input["model_name"]
    debug_test = args["experiment_params"]["debug_test"]
    if debug_test:
        print("!!!   WARNING !!! debug_test flag set to TRUE.")
        args["train_params"]["epochs"] = 2
        args["train_params"]["log_interval"] = args["train_params"]["epochs"]
        args["experiment_params"]["n_tasks"] = 2
        args["experiment_params"]["classes_per_task"] = [30, 5]

    # Path setup for the experiment

    args = path_setup(args)
    print(args)

    # Logger
    logger = Log(args["experiment_params"]["experiment_save_path"])

    # Experiment parameters setup
    
    classes_per_task = args["experiment_params"]["classes_per_task"]
    n_tasks = args["experiment_params"]["n_tasks"]
    dataset = args["dataset_params"]["dataset"]

    print(f"Fetching dataset... ", flush=True)
    selected_classes = []

    ordered = args["dataset_params"]["ordered"]

    if "audioset" in dataset:
        data_path = f'{args["dataset_params"]["base_path"]}/{dataset}'
        if ordered:
            with open(f'{data_path}/metadata/ordered_keys.json', 'r') as f:
                class_order = json.load(f)
            
            selected_classes_tot = class_order["ordered_indexes"][:np.array(classes_per_task).sum().item()]
        else:
            selected_classes_tot = random.sample(list(range(len(MID2ID_Audioset))), np.array(classes_per_task).sum().item())
        for i in range(len(classes_per_task)): 
            prev_slice = i if i == 0 else prev_slice
            slice = classes_per_task[i] + prev_slice
            selected_classes.append(selected_classes_tot[prev_slice:slice])
            prev_slice += classes_per_task[i]

    
        datasets = []
        

        print(selected_classes)
        for i, elem in enumerate(selected_classes):
            datasets.append(Audioset(data_path=data_path, selected_classes=selected_classes[i], test=False, debug=debug_test))


    elif "fsd50k" in dataset:
        data_path = f'{args["dataset_params"]["base_path"]}/{dataset}'
        if ordered:
            with open(f'{data_path}/metadata_OG/ordered_keys.json', 'r') as f:
                class_order = json.load(f)
            selected_classes_tot = class_order["ordered_indexes"][:np.array(classes_per_task).sum().item()]
        else:
            selected_classes_tot = random.sample(list(range(len(MID2ID_Fsd50k))), np.array(classes_per_task).sum().item())
        for i in range(len(classes_per_task)): 
            prev_slice = i if i == 0 else prev_slice
            slice = classes_per_task[i] + prev_slice
            selected_classes.append(selected_classes_tot[prev_slice:slice])
            prev_slice += classes_per_task[i]

    
        datasets = []
        print(selected_classes)
        for i, elem in enumerate(selected_classes):
            datasets.append(Fsd50k(data_path=data_path, selected_classes=selected_classes[i], test=False, debug=debug_test))

    print("Done")


    # Device selection
    if torch.cuda.is_available():
        device_str = 'cuda'
        device = torch.device('cuda')
    else:
        device_str = 'cpu'
        device = torch.device('cpu')
    print(f"Using device: {device}", flush=True)
    print(f"Using torch version: {torch.__version__}", flush=True)

    
    
    if args["experiment_params"]["complete"]: 
        for task in range(n_tasks):
            if not os.path.exists(f"{args['model_params']['model_state_dest']}/model_T{task}/filters_scores"):
                os.makedirs(f"{args['model_params']['model_state_dest']}/model_T{task}/filters_scores")
                print(f"Created directory: {args['model_params']['model_state_dest']}/model_T{task}/filters_scores")
            train_setup(args, task, datasets[task], device=device, device_str=device_str, setup_start_time=setup_start_time, logger=logger)
    
    # Clean-up
    for d in datasets:
        d.__del__()
    
    print(logger.get_full_logs())
    logger.log_full_on_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='')
    parser.add_argument('--model_name', type=str, help='')

    input = vars(parser.parse_args())

    main(input)


