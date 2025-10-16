import argparse
import time

import torch
from sklearn.metrics import classification_report, average_precision_score, f1_score
import numpy as np
from collections import Counter

from cnn14_pann_lin import Cnn14
from cl_dataset_class import CL_dataset

# Heavily copied from and inspired by CIL-ML-AUDIO

def evaluate(model, 
             eval_loader, 
             device,
             device_str, 
             use_amp, 
             log_interval,
             cil_classes,
             all_episodes,
             nr_of_classes):
    
    model.to(device)
    model.eval()
    print(f'Set model to device: {device} and to evaluation mode.', flush=True)
    
    size = len(eval_loader.dataset)
    running_time = 0
    iterations = 0

    og_classes = 30

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch, (mel, label, fname) in enumerate(eval_loader):
            
            batch_start_time = time.time()

            with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=use_amp):
                mel, label = mel.to(device), label
                out, _ = model(mel.float())
                print(f"Out: {out}")
                tmp_sig = torch.sigmoid(out)
                print(f"Sigmoid values: {tmp_sig}")
                #preds = torch.gt(torch.sigmoid(out), 0.5)
            all_preds.extend(
                tmp_sig.cpu().numpy())
            all_targets.extend(np.asarray(label))

            batch_end_time = time.time()
            running_time += (batch_end_time - batch_start_time)
            iterations += 1

            if batch % log_interval == 0:
                print(f"Average batch training time: {running_time / iterations}", flush=True)
                current = (batch + 1) * len(mel)
                print(f"Eval progression: [{current:>5d}/{size:>5d}]", flush=True)

        Y_predicted = np.asarray(all_preds)
        Y_predicted_bin = torch.gt(torch.tensor(Y_predicted), 0.5)
        Y_ref = np.asarray(all_targets)

        # One classification report provides necessary information. No need to flood the output with several classification results.
        # Other metrics are useful to have as separate prints to inspect plasticity and stability
        print(classification_report(Y_ref, Y_predicted_bin))
        print("Full run results\n")
        print_eval_metrics(gt=Y_ref, 
                           preds=Y_predicted, 
                           print_id="Full run")

        initial_preds = Y_predicted[:, 0:og_classes] # 30 classes originally
        initial_labels = Y_ref[:, 0:og_classes]
        print(f"Initial {og_classes} classes results\n")
        print_eval_metrics(gt=initial_labels, 
                           preds=initial_preds, 
                           print_id=f"Initial {og_classes} classes")

        if cil_classes != 0 and not all_episodes:
            # Latest cil classes
            cil_preds = Y_predicted[:, -cil_classes:]
            cil_labels = Y_ref[:, -cil_classes:]
            print(f"Latest cil classes({cil_classes})\n")
            print_eval_metrics(gt=cil_labels,
                               preds=cil_preds,
                               print_id=f"Latest {cil_classes} classes")
        elif cil_classes != 0 and all_episodes:

            episodes = (nr_of_classes - og_classes) // cil_classes
            start_idx = og_classes

            for iter in range(1, episodes+1):
                end_idx = iter * 5 + og_classes
                cil_preds = Y_predicted[:, start_idx:end_idx]
                cil_labels = Y_ref[:, start_idx:end_idx]
                print(f"Episode {iter} cil classes({start_idx}:{end_idx})\n")
                print_eval_metrics(gt=cil_labels,
                                preds=cil_preds,
                                print_id=f"Episode {iter} ({start_idx}:{end_idx}) classes")
                start_idx = end_idx
            
    
# Note the order of predictions and true values with sklearn metrics
def print_eval_metrics(gt, preds, print_id):

    average_precision = average_precision_score(gt, preds, average=None)
    mAp = np.mean(average_precision)

    preds = torch.gt(torch.tensor(preds), 0.5)
    f1_macro = f1_score(gt, preds, average='macro', zero_division=0.0)
    f1_micro = f1_score(gt, preds, average='micro', zero_division=0.0)

    metrics = {'mAp': mAp, 'F1-macro': f1_macro, 'F1-micro': f1_micro}
    for metric in metrics:
        print(f"{print_id} {metric}: {metrics[metric]}")
    print("\n")


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
    parser.add_argument('--cil_classes', type=int, default=0, help="If this is not zero, print the metrics for the newly learned cil classes as well.")
    parser.add_argument('--all_episodes', action='store_true', help='Whether to evaluate every episode separately. The assumption is that each episode is done in increments of 5 classes beyond the original 30 classes.')

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
    PATH_TO_HDF5_DATA = args['path_to_data'] 
    PATH_TO_MODEL_STATE = args['path_to_model_state']
    cil_classes = args['cil_classes']
    all_episodes = args['all_episodes']

    # Data setup
    print(f"Fetching data...", flush=True)
    data_eval = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                                    dataset=dataset,
                                    split='eval',
                                    nr_of_classes=nr_of_classes)
    print(f"There are {len(data_eval)} evaluation files in total.", flush=True)
    
    evaluation_loader = torch.utils.data.DataLoader(data_eval, batch_size=batch_size, num_workers=nr_of_workers)
    print("Finished data setup.", flush=True)

    # For quick local tests
    smaller_eval, _ = torch.utils.data.random_split(data_eval, [0.005, 0.995])
    smaller_eval_loader = torch.utils.data.DataLoader(smaller_eval, batch_size=batch_size, num_workers=nr_of_workers)

    model = Cnn14(nr_of_classes)
    model.load_state_dict(torch.load(PATH_TO_MODEL_STATE,
                                         weights_only=True))
    print(f"Finished loading model state from: {PATH_TO_MODEL_STATE}", flush=True)

    evaluate(model=model, 
             eval_loader=evaluation_loader, 
             device=device,
             device_str=device_str, 
             use_amp=use_amp, 
             log_interval=log_interval,
             cil_classes=cil_classes,
             all_episodes=all_episodes,
             nr_of_classes=nr_of_classes)
    
    end_time = time.time()
    total_time = round(end_time-start_time, 2)

    print(f"Finished evaluating the model. The evaluation script ran for {total_time} seconds in total without synchronizations with GPU.", flush=True)