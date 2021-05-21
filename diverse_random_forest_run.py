import numpy as np 
import matplotlib.pyplot as plt
import pickle
import torch
import time

from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score, mean_squared_error 
from sklearn.model_selection import train_test_split
import autosklearn.regression
from sklearn.ensemble import RandomForestRegressor


def main():
    inputs = pickle.load(open('methane_storage.pkl', 'rb'))['inputs']
    # normalizing input features
    for i in range(len(inputs[0])):
        inputs[:, i] = (inputs[:, i] - np.min(inputs[:, i]))/(np.max(inputs[:, i]) - np.min(inputs[:, i]))
    outputs = pickle.load(open('methane_storage.pkl', 'rb'))['outputs'].values
    # normalizing outpuuts
    outputs = ((outputs - np.min(outputs))/(np.max(outputs)-np.min(outputs)))
    rlist, maelist, mselist = [], [], []

    for num_training_points in [50, 100, 150, 200, 300, 400, 500]:
        print(f"num_training_points : {num_training_points}")
        #num_training_points = int(test_size_percentage * len(inputs))
        all_best_vals = []
        all_best_idxs = []
        for i in range(10): # num of runs
            train_idxs = [np.random.randint(0, len(inputs))] # initialize with one random point; pick others in a max diverse fashion
            for j in range(num_training_points):
                #print(np.sum((inputs - inputs[train_idxs][:, None, :])**2, axis=-1).T.shape)
                #print(np.sort(np.sum((inputs - inputs[train_idxs][:, None, :])**2, axis=-1).T, axis=1))
                if j >= 1:
                    distances = np.sort(np.sum((inputs - inputs[train_idxs][:, None, :])**2, axis=-1).T, axis=1)[:, 1]
                else:
                    distances = np.sort(np.sum((inputs - inputs[train_idxs][:, None, :])**2, axis=-1).T, axis=1)
                train_idxs.append(np.argmax(distances))
            #print(train_idxs)
            X_train = inputs[train_idxs]
            y_train = outputs[train_idxs]

            X_test = inputs[np.setdiff1d(np.arange(len(inputs)), train_idxs)]
            y_test = outputs[np.setdiff1d(np.arange(len(inputs)), train_idxs)]
                
            start_time = time.time()
            regr = RandomForestRegressor()
            regr.fit(X_train, y_train)
            #best_idx = np.argmax(regr.predict(X_test))
            best_indices = np.argsort(-regr.predict(X_test))[:100]
            best_val = y_test[best_indices] 
            all_best_vals.append(best_val)
            all_best_idxs.append(best_indices)
        with open('diverse_rf_data_num_training_points_'+str(num_training_points)+'.pkl', 'wb') as f:
            pickle.dump({'all_best_vals': all_best_vals, 'all_best_idxs': all_best_idxs}, f)


if __name__ == '__main__':
    main()
