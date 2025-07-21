import argparse

import torch
from sklearn.metrics import classification_report, average_precision_score, f1_score
import numpy as np
from collections import Counter

from cnn14_pann_lin import Cnn14
from cl_dataset_class import CL_dataset

def evaluate(model, eval_loader):

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for mel, label, fname in eval_loader:
            mel, label = mel.to(device), label.to(device)
            print(label.sum())
            out, _ = model(mel.float())
            preds = torch.gt(torch.sigmoid(out), 0.5)
            all_preds.extend(
                preds.numpy())
            all_targets.extend(np.asarray(label))

        Y_predicted = np.asarray(all_preds)
        Y_ref = np.asarray(all_targets)

        #print(f"Predicted array: {Y_predicted} and its type {type(Y_predicted)}")
        #print(f"Ground truth array: {Y_ref} and its type {type(Y_ref)}")


        print('Reference polyphony:', Counter(Y_ref.sum(axis=1)))
        print('Predicted polyphony:', Counter(Y_predicted.sum(axis=1)))
        report = classification_report(Y_ref, Y_predicted, output_dict=True)
        print(classification_report(Y_ref, Y_predicted))

        average_precision = average_precision_score(Y_ref, Y_predicted, average=None)
        mAp = np.mean(average_precision)
        print('mAP', mAp)

        f1_macro = f1_score(Y_ref, Y_predicted, average='macro', zero_division=0.0)
        f1_micro = f1_score(Y_ref, Y_predicted, average='micro', zero_division=0.0)
        print('macro', f1_macro)
        print('micro', f1_micro)

        return mAp, f1_macro, f1_macro, report
    
def comparative_printing(class_report1: dict, class_report2: dict):
    """
    The function takes in 2 sklearn's classification reports. If the reports share the same key, then they are compared.
    params:
        class_report1: sklearn classification report dict
        class_report2: sklearn classification report dict
    return:
        Nothing
    """
    for key1 in class_report1.keys():
        for key2 in class_report2.keys():
            if key1 == key2:
                print(f"F1-score for class {key1}: \n new: {class_report1[key1]['f1-score']} \n old: {class_report2[key1]['f1-score']}")

if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--nr_of_classes_new', type=int, choices=[30, 35, 40, 45, 50], help='Number of classes to use from data for new model. Choices [30, 35, 40, 45, 50]')
    parser.add_argument('--nr_of_classes_old', type=int, choices=[30, 35, 40, 45, 50], help='Number of classes to use from data for old model. Choices [30, 35, 40, 45, 50]')
    parser.add_argument('--dataset', type=str, help='Choice of dataset. Either audioset or fsd50k')
    parser.add_argument('--path_to_data', type=str, help='The path to the HDF5 datafile.')
    parser.add_argument('--nr_of_workers', type=int, default=0, help='Number of workers for dataloading')
    parser.add_argument('--batch_size', type=int, default=8, help='Size of the loaded data batch. A tensor of [batch_size, data_tensor.shape] is loaded.')
    parser.add_argument('--path_to_model_state', type=str, help='Path to the pickle saved pytorch model state dict.')
    parser.add_argument('--path_to_old_model_state', type=str, help='Path to the pickle saved pytorch model state dict of the older model.')

    args = vars(parser.parse_args())

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Params
    nr_of_classes_new = args['nr_of_classes_new']
    nr_of_classes_old = args['nr_of_classes_old']
    dataset = args['dataset']
    batch_size = args['batch_size']
    nr_of_workers = args['nr_of_workers']
    PATH_TO_HDF5_DATA = args['path_to_data'] #r'C:\Users\mp431591\Documents\work_code\cl_30\continual_learning\data_cl.hdf5'
    PATH_TO_MODEL_STATE = args['path_to_model_state']
    PATH_TO_OLD_MODEL_STATE = args['path_to_old_model_state']

    # Data setup
    data_eval_new = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                                    dataset=dataset,
                                    split='eval',
                                    nr_of_classes=nr_of_classes_new)
    data_eval_old = CL_dataset(path_to_data_hdf5=PATH_TO_HDF5_DATA,
                                    dataset=dataset,
                                    split='eval',
                                    nr_of_classes=nr_of_classes_old)
    
    small_eval_data_new, _ = torch.utils.data.random_split(data_eval_new,
                                                    [0.001, 0.999])
    small_eval_data_old, _ = torch.utils.data.random_split(data_eval_old,
                                                    [0.001, 0.999])
    
    evaluation_loader_new = torch.utils.data.DataLoader(small_eval_data_new, batch_size=batch_size, num_workers=nr_of_workers)
    evaluation_loader_old = torch.utils.data.DataLoader(small_eval_data_old, batch_size=batch_size, num_workers=nr_of_workers)


    model_new = Cnn14(nr_of_classes_new)
    model_new.load_state_dict(torch.load(PATH_TO_MODEL_STATE,
                                         weights_only=True))
    model_old = Cnn14(nr_of_classes_old)
    model_old.load_state_dict(torch.load(PATH_TO_OLD_MODEL_STATE,
                                         weights_only=True))


    mAP_new, f1_macro_new, f1_micro_new, report_new = evaluate(model=model_new,
                                               eval_loader=evaluation_loader_new)
    mAP_old, f1_macro_old, f1_micro_old, report_old = evaluate(model=model_old,
                                                               eval_loader=evaluation_loader_old)
    print(f"New metrics:\n mAP {mAP_new}, f1_M {f1_macro_new}, f1_m {f1_micro_new}")
    print(f"Old metrics:\n mAP {mAP_old}, f1_M {f1_macro_old}, f1_m {f1_micro_old}")
    comparative_printing(report_new, report_old)



    