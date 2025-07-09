import random
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from cka_lib import cka
from cnn14_pann_lin import Cnn14
from cl_dataset_class import CL_dataset

# Taken from https://docs.pytorch.org/docs/stable/notes/randomness.html#reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Assumes the use of PANN-CNN14 architecture
# Copied from CIL-ML-AUDIO
def do_cka(model1, model2, device, device_str, dataloader, model_1_name, model_2_name, plot_title, plot_save_path, cka_diag_vals_dest):
    model1.eval()
    model2.eval()
    model1.to(device)
    model2.to(device)
    cka_alg = cka.CKA(model1, model2,
                      model1_name=model_1_name,  # good idea to provide names to avoid confusion
                      model2_name=model_2_name,
                      model1_layers=['conv_block1.conv1', 'conv_block1.conv2', 'conv_block2.conv1', 'conv_block2.conv2',
                                     'conv_block3.conv1', 'conv_block3.conv2', 'conv_block4.conv1', 'conv_block4.conv2',
                                     'conv_block5.conv1', 'conv_block5.conv2', 'conv_block6.conv1', 'conv_block6.conv2',
                                     'conv_block1.bn1', 'conv_block1.bn2', 'conv_block2.bn1', 'conv_block2.bn2',
                                     'conv_block3.bn1', 'conv_block3.bn2', 'conv_block4.bn1', 'conv_block4.bn2',
                                     'conv_block5.bn1', 'conv_block5.bn2', 'conv_block6.bn1', 'conv_block6.bn2', 'fc'],
                      # List of layers to extract features from
                      model2_layers=['conv_block1.conv1', 'conv_block1.conv2', 'conv_block2.conv1', 'conv_block2.conv2',
                                     'conv_block3.conv1', 'conv_block3.conv2', 'conv_block4.conv1', 'conv_block4.conv2',
                                     'conv_block5.conv1', 'conv_block5.conv2', 'conv_block6.conv1', 'conv_block6.conv2',
                                     'conv_block1.bn1', 'conv_block1.bn2', 'conv_block2.bn1', 'conv_block2.bn2',
                                     'conv_block3.bn1', 'conv_block3.bn2', 'conv_block4.bn1', 'conv_block4.bn2',
                                     'conv_block5.bn1', 'conv_block5.bn2', 'conv_block6.bn1', 'conv_block6.bn2', 'fc'],
                      device=device_str)

    cka_alg.compare(dataloader)  # secondary dataloader is optional

    cka_alg.plot_results(save_path=plot_save_path, title=plot_title, display_plot=False)

    plt.close()

    results = cka_alg.export()
    diag_sim = np.array(results['CKA'])
    diag_res = np.diag(diag_sim)
    print('CKA scores:', diag_res)
    indices = []
    cka_vals = []
    for idx, val in enumerate(diag_res):
        indices.append(idx)
        cka_vals.append(round(val, 5))
    plt.plot(np.array(indices), np.array(cka_vals))
    plt.xlabel('Layer index')
    plt.ylabel('CKA value')
    plt.title('Diagonals: ' + plot_title, fontsize=15)
    plt.ylim(0.0, 1.05)
    plt.savefig((cka_diag_vals_dest + '.svg'), dpi=300)

if __name__ == '__main__':

    start_time = time.time()

    # Command line args
    parser = argparse.ArgumentParser()

    parser.add_argument('--nr_of_classes', type=int, choices=[30, 50], help='Number of classes to use from data')
    parser.add_argument('--dataset', type=str, choices=['audioset', 'fsd50k'], help='Choice of dataset.')
    parser.add_argument('--dataset_split', type=str, choices=['train', 'eval'], help='Which dataset split to use for dataloading and with CKA.')
    parser.add_argument('--path_to_data', type=str, help='The path to the HDF5 datafile.')
    parser.add_argument('--nr_of_workers', type=int, default=0, help='Number of workers for dataloading')
    parser.add_argument('--batch_size', type=int, default=8, help='Size of the loaded data batch. A tensor of [batch_size, data_tensor.shape] is loaded.')
    parser.add_argument('--path_to_model_state_1', type=str, help='Location of the model state dict from which to initialize the 1st cnn14 model.')
    parser.add_argument('--path_to_model_state_2', type=str, help='Location of the model state dict from which to initialize the 2nd cnn14 model.')
    parser.add_argument('--model_name_1', type=str, help='Name of the 1st model, for CKA')
    parser.add_argument('--model_name_2', type=str, help='Name of the 2nd model, for CKA')
    parser.add_argument('--cka_plot_save_path', type=str, help='The desired location of the CKA plot.')

    args = vars(parser.parse_args())

    print(f"CKA comparison was done with the following commandline arguments: {args}", flush=True)

    nr_of_classes = args['nr_of_classes']
    dataset = args['dataset']
    dataset_split = args['dataset_split']
    PATH_TO_HDF5_DATA = args['path_to_data']
    batch_size = args['batch_size']
    nr_of_workers = args['nr_of_workers']
    PATH_TO_MODEL_STATE_1 = args['path_to_model_state_1']
    PATH_TO_MODEL_STATE_2 = args['path_to_model_state_2']
    model_name_1 = args['model_name_1']
    model_name_2 = args['model_name_2']
    cka_plot_save_path = args['cka_plot_save_path']

    cka_plot_title = f"{model_name_1} VS {model_name_2} on {dataset}-{dataset_split}"
    cka_diag_vals_dest = os.path.join(cka_plot_save_path, (('diags ' + cka_plot_title)).replace(' ', '_'))
    cka_plot_save_dest = os.path.join(cka_plot_save_path, cka_plot_title.replace(' ', '_'))

    # Initialize (hopefully reproducible) randomness for data loading
    torch_generator = torch.Generator()
    torch_generator.manual_seed(0)

    # Device selection
    if torch.cuda.is_available():
        device_str = 'cuda'
        device = torch.device('cuda')
    else:
        device_str = 'cpu'
        device = torch.device('cpu')
    print(f"Using device: {device}", flush=True)

    model_1 = Cnn14(nr_of_classes)
    model_1.load_state_dict(torch.load(PATH_TO_MODEL_STATE_1, weights_only=True))
    model_2 = Cnn14(nr_of_classes)
    model_2.load_state_dict(torch.load(PATH_TO_MODEL_STATE_2, weights_only=True))
    print(f"Initialized the models successfully.", flush=True)

    data_eval = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                                dataset=dataset,
                                split=dataset_split,
                                nr_of_classes=nr_of_classes)

    small_eval_data, _ = torch.utils.data.random_split(data_eval, [0.0025, 0.9975])

    # 'Drop last' suggested by CKA library author for avoiding dimension
    # mismatches
    eval_loader = torch.utils.data.DataLoader(dataset=small_eval_data, batch_size=batch_size, num_workers=nr_of_workers, worker_init_fn=seed_worker, generator=torch_generator, drop_last=True)

    print(f"Set up data.", flush=True)

    setup_time = time.time()
    print(f"Time taken before doing CKA: {round(setup_time-start_time, 2)} seconds.")

    do_cka(model1=model_1, model2=model_2, model_1_name=model_name_1, model_2_name=model_name_2, dataloader=eval_loader, device=device, device_str=device_str, plot_title=cka_plot_title, plot_save_path=cka_plot_save_dest, cka_diag_vals_dest=cka_diag_vals_dest)

    end_time = time.time()
    print(f"The script took a total of {round(end_time-start_time, 2)} seconds.", flush=True)
