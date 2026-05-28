import argparse

import numpy as np    
from scipy.stats.mstats import gmean
from scipy.spatial import distance
import torch

from cnn14_pann_lin import Cnn14

# Copied from https://github.com/Arshdeep-Singh-Boparai/Efficient_CNNs_passive_filter_pruning
def operator_norm_pruning(W):
	C_M=[]	
	mean_vec=[]
	for i in range(np.shape(W)[1]):
		A=W[:,i,:].T
		A_mean=np.mean(A,0)
		e=np.tile(A_mean,(np.shape(A)[0],1))
		A_centred=A-e
		mean_vec.append(A_mean)
		u,q,v=np.linalg.svd(A_centred)
		u1=np.reshape(u[:,0],(np.shape(A)[0],1))
		v1=np.reshape(v[0,:],(np.shape(A)[1],1))
		c_1=np.matmul(u1,v1.T)
		c_1_norm=c_1[0,:]/np.linalg.norm(c_1[0,:])
		C_M.append(c_1_norm)
	Score=[]
	for i in range(np.shape(W)[2]):
		Score.append(np.trace((np.matmul((W[:,:,i]-np.array(mean_vec).T).T,np.array(C_M).T))))
	Mse_score=(np.array(Score))**2
	Mse_score_norm=Mse_score/np.max(Mse_score)
	return Mse_score_norm

def calculate_filter_scores(path_to_model_state, nr_of_classes, dest_of_filter_scores):	
    
	
    MODEL_PATH = path_to_model_state
    NR_OF_CLASSES = nr_of_classes
    DEST_OF_FILTER_SCORES = dest_of_filter_scores

    model = Cnn14(NR_OF_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    tmp_W = model.state_dict()
    indices_and_keys = {}
    tmp_L = []
    counter = 1
    for idx, key in enumerate(tmp_W):
        if 'conv1' in key or 'conv2' in key:
            print(f"{key} is the {idx} member of the state dict")
            print(f"It's shape: {tmp_W[key].shape}", flush=True)
            indices_and_keys[idx] = key
            tmp_L.append(counter)
            counter += 1
    print(indices_and_keys)
    print(tmp_L)

    for idx, key in enumerate(indices_and_keys):
        conv_layer_key = indices_and_keys[key]
        print(conv_layer_key)
        W_2D=tmp_W[conv_layer_key].numpy()
        W=np.reshape(W_2D,
                    (9, np.shape(W_2D)[1], np.shape(W_2D)[0]))
        print(np.shape(W),'layer  :','  ',tmp_L[idx])
        print(np.shape(W),'shape of weights', flush=True)
        print(type(W))
        score_norm_m1 = operator_norm_pruning(W)
        print(f"Length of score norm (nr of filters): {len(score_norm_m1)}")
        print(f"Saving layer {str(tmp_L[idx])}'s scores", flush=True)
        file_name = f"{DEST_OF_FILTER_SCORES}/sim_index{str(tmp_L[idx])}.npy"
        np.save(file_name, np.argsort(score_norm_m1))