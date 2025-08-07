import argparse
import time

import torch
from sklearn.metrics import classification_report, average_precision_score, f1_score
import numpy as np
from collections import Counter

from cnn14_pann_lin import Cnn14
from cl_dataset_class import CL_dataset

# Heavily copied and inspired by CIL-ML-AUDIO

def evaluate(model, 
             eval_loader, 
             device,
             device_str, 
             use_amp, 
             log_interval):
    
    model.to(device)
    model.eval()
    print(f'Set model to device: {device} and to evaluation mode.', flush=True)
    
    size = len(eval_loader.dataset)
    running_time = 0
    iterations = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch, (mel, label, fname) in enumerate(eval_loader):
            
            batch_start_time = time.time()

            with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
                mel, label = mel.to(device), label
                out, _ = model(mel.float())
                preds = torch.gt(torch.sigmoid(out), 0.5)
            all_preds.extend(
                preds.cpu().numpy())
            all_targets.extend(np.asarray(label))

            batch_end_time = time.time()
            running_time += (batch_end_time - batch_start_time)
            iterations += 1

            if batch % log_interval == 0:
                print(f"Average batch training time: {running_time / iterations}", flush=True)
                current = (batch + 1) * len(mel)
                print(f"Eval progression: [{current:>5d}/{size:>5d}]", flush=True)

        Y_predicted = np.asarray(all_preds)
        Y_ref = np.asarray(all_targets)

        initial_preds = Y_predicted[:, 0:30] # 30 classes originally
        initial_labels = Y_ref[:, 0:30]

        #print(f"Predicted array: {Y_predicted} and its type {type(Y_predicted)}")
        #print(f"Ground truth array: {Y_ref} and its type {type(Y_ref)}")

        report = classification_report(Y_ref, Y_predicted, output_dict=True)
        print(classification_report(Y_ref, Y_predicted))

        average_precision = average_precision_score(Y_ref, Y_predicted, average=None)
        mAp = np.mean(average_precision)
        print('mAP', mAp)

        f1_macro = f1_score(Y_ref, Y_predicted, average='macro', zero_division=0.0)
        f1_micro = f1_score(Y_ref, Y_predicted, average='micro', zero_division=0.0)
        print('macro', f1_macro)
        print('micro', f1_micro)

        # Values for the original 30 classes
        
        report_ini = classification_report(initial_labels,
                                           initial_preds, 
                                           output_dict=True)
        print("For the initial 30 classes", flush=True)
        print(classification_report(initial_labels, initial_preds))

        avg_prec_init = average_precision_score(initial_labels,
                                                initial_preds,
                                                average=None)

        mAp_init = np.mean(avg_prec_init)
        print(f'Initial 30 classes mAP: {mAp_init}')

        f1_macro_init = f1_score(initial_labels,
                                 initial_preds,
                                 average='macro',
                                 zero_division=0.0)
        f1_micro_init = f1_score(initial_labels,
                                 initial_preds,
                                 average='micro',
                                 zero_division=0.0)
        print(f'Initial 30 classes f1-macro {f1_macro_init}')
        print(f'Initial 30 classes f1-micro {f1_micro_init}', flush=True)

        return mAp, f1_macro, f1_macro, report
    
# Note the order of predictions and true values with sklearn metrics
def print_eval_metrics(preds, gt, print_id):

    print(classification_report(gt, preds))
    average_precision = average_precision_score(gt, preds, average=None)
    mAp = np.mean(average_precision)
    f1_macro = f1_score(gt, preds, average='macro', zero_division=0.0)
    f1_micro = f1_score(gt, preds, average='micro', zero_division=0.0)

    metrics = vars([mAp, f1_macro, f1_micro])
    for metric in metrics:
        print(f"{print_id} {metric}: {metrics[metric]}")
    

    

if __name__ == '__main__':

    start_time = time.time()

    # Command line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--nr_of_classes', type=int, choices=[30, 35, 40, 45, 50], help='Number of classes to use from data for model. Choices [30, 35, 40, 45, 50]')
    parser.add_argument('--dataset', type=str, help='Choice of dataset. Either audioset or fsd50k')
    parser.add_argument('--path_to_data', type=str, help='The path to the HDF5 datafile.')
    parser.add_argument('--nr_of_workers', type=int, default=0, help='Number of workers for dataloading')
    parser.add_argument('--batch_size', type=int, default=8, help='Size of the loaded data batch. A tensor of [batch_size, data_tensor.shape] is loaded.')
    parser.add_argument('--path_to_model_state', type=str, help='Path to the pickle saved pytorch model state dict.')
    parser.add_argument('--use_amp', action='store_true', help='Whether to use torch enabled automatic mixed precision.')
    parser.add_argument('--log_interval', type=int, help='How often to display mini batch information.')

    args = vars(parser.parse_args())

    print(f"This model evaluation is run with the following commandline arguments: {args}", flush=True)

    # Device selection
    if torch.cuda.is_available():
        device_str = 'cuda'
        device = torch.device('cuda')
    else:
        device_str = 'cpu'
        device = torch.device('cpu')
    print(f"Using device: {device}", flush=True)
    print(f"Using torch version: {torch.__version__}")

    # Params
    nr_of_classes = args['nr_of_classes']
    dataset = args['dataset']
    batch_size = args['batch_size']
    nr_of_workers = args['nr_of_workers']
    use_amp = args['use_amp']
    log_interval = args['log_interval']
    PATH_TO_HDF5_DATA = args['path_to_data'] #r'C:\Users\mp431591\Documents\work_code\cl_30\continual_learning\data_cl.hdf5'
    PATH_TO_MODEL_STATE = args['path_to_model_state']

    # Data setup
    print(f"Fetching data...", flush=True)
    data_eval = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                                    dataset=dataset,
                                    split='eval',
                                    nr_of_classes=nr_of_classes)
    print(f"There are {len(data_eval)} evaluation files in total.", flush=True)
    
    evaluation_loader = torch.utils.data.DataLoader(data_eval, batch_size=batch_size, num_workers=nr_of_workers)
    print("Finished data setup.", flush=True)

    model = Cnn14(nr_of_classes)
    model.load_state_dict(torch.load(PATH_TO_MODEL_STATE,
                                         weights_only=True))
    print(f"Finished loading model state from: {PATH_TO_MODEL_STATE}", flush=True)

    evaluate(model=model, 
             eval_loader=evaluation_loader, 
             device=device,
             device_str=device_str, 
             use_amp=use_amp, 
             log_interval=log_interval)
    
    end_time = time.time()
    total_time = round(end_time-start_time, 2)

    print(f"Finished evaluating the model. The evaluation script ran for {total_time} seconds in total without synchronizations with GPU.", flush=True)