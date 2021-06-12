### code to generate combinedresults 
### since each program generates a single file for each run, this is a wrapper code to combine data for all runs into a single file


import numpy as np 
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch
import os


def main():
    outputs =  pickle.load(open('methane_storage.pkl', 'rb'))['outputs'].values

    rf_data = {}
    temp = []
    temp_idxs = []
    for i in range(10):
        temp.append(np.array(pickle.load(open('mbo_rf_methane_data_tsp99_'+str(i)+'.pkl', 'rb'))['all_best_vals']))
        temp_idxs.append(np.array(pickle.load(open('mbo_rf_methane_data_tsp99_'+str(i)+'.pkl', 'rb'))['all_best_idxs']))
    rf_data['outputs_normalized'] = np.array(temp)
    rf_data['outputs'] = outputs[np.array(temp_idxs)]
    torch.save(rf_data, "rf_single_acq_results.pkl")
    rf_data = {}
    temp = []
    temp_idxs = []
    for i in range(10):
        temp.append(np.array(pickle.load(open('mbo_50percent_rf_methane_data_tsp50_'+str(i)+'.pkl', 'rb'))['all_best_vals']))
        temp_idxs.append(np.array(pickle.load(open('mbo_50percent_rf_methane_data_tsp50_'+str(i)+'.pkl', 'rb'))['all_best_idxs']))
    rf_data['outputs_normalized'] = np.array(temp)
    rf_data['outputs'] = outputs[np.array(temp_idxs)]
    torch.save(rf_data, "rf_50percent_acq_results.pkl")
    diverse_rf_data = {}
    temp = []
    temp_idxs = []
    for i in range(10):
        temp.append(np.array(pickle.load(open('mbo_diverse_rf_data_num_training_points_99_'+str(i)+'.pkl', 'rb'))['all_best_vals']))
        temp_idxs.append(np.array(pickle.load(open('mbo_diverse_rf_data_num_training_points_99_'+str(i)+'.pkl', 'rb'))['all_best_idxs']))
    diverse_rf_data['outputs_normalized'] = np.array(temp)
    diverse_rf_data['outputs'] = outputs[np.array(temp_idxs)]
    torch.save(diverse_rf_data, "diverse_rf_single_acq_results.pkl")


    diverse_rf_data = {}
    temp = []
    temp_idxs = []
    for i in range(10):
        temp.append(np.array(pickle.load(open('mbo_50percent_diverse_rf_data_num_training_points_50_'+str(i)+'.pkl', 'rb'))['all_best_vals']))
        temp_idxs.append(np.array(pickle.load(open('mbo_50percent_diverse_rf_data_num_training_points_50_'+str(i)+'.pkl', 'rb'))['all_best_idxs']))
    diverse_rf_data['outputs_normalized'] = np.array(temp)
    diverse_rf_data['outputs'] = outputs[np.array(temp_idxs)]
    torch.save(diverse_rf_data, "diverse_rf_50percent_acq_results.pkl")


    temp = []
    for i in range(10):
        temp.append(torch.load('bo_data_run'+str(i)+'.pkl')['outputs'])
    bo_results = {}
    bo_results['outputs_normalized'] = np.array(temp)
    torch.save('bo_results.pkl', bo_results)


    normalized_es_data = []
    max_len = 0
    sigma = 0.2
    popsize = 20
    for i in range(10):
        normalized_es_data.append((-1*np.matrix.flatten(np.array(pickle.load(open('es_run_'+str(i)+'_' + 'sigma_' + str(sigma) + '_popsize_'+ str(popsize) +'.pkl', 'rb'))['final_outputs']))))
    normalized_es_data = np.array(normalized_es_data)
    torch.save({"outputs_normalized":normalized_es_data[:, :500]}, "es_results.pkl")

if __name__ == '__main__':
    main()
