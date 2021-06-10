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


    for i in range(10):
        all_best_vals = []
        all_best_idxs = []
        for num_training_points in [50, 100, 150, 200, 300, 400, 500]:
            print(f"num_training_points : {num_training_points}")
            num_training_points = num_training_points - len(all_best_vals)
            num_training_points = num_training_points - 1 
            print(f"num_training_points : {num_training_points}")
            train_idxs = [np.random.randint(0, len(inputs))] # initialize with one random point; pick others in a max diverse fashion
            for j in range(num_training_points - 1):
                if j >= 1:
                    distances = np.sort(np.sum((inputs - inputs[train_idxs][:, None, :])**2, axis=-1).T, axis=1)[:, 1]
                else:
                    distances = np.sort(np.sum((inputs - inputs[train_idxs][:, None, :])**2, axis=-1).T, axis=1)
                train_idxs.append(np.argmax(distances))
            print(f'train_idxs {len(train_idxs)}')
            X_train = inputs[train_idxs]
            y_train = outputs[train_idxs]

            all_best_vals.extend(y_train)
            all_best_idxs.extend(train_idxs)

            test_idxs = np.setdiff1d(np.arange(len(inputs)), train_idxs)

            X_test = inputs[test_idxs]
            y_test = outputs[test_idxs]
                
            start_time = time.time()
            regr = RandomForestRegressor()
            regr.fit(X_train, y_train) 
            #best_idx = np.argmax(regr.predict(X_test))
            best_indices = np.argsort(-regr.predict(X_test))[:1]
            #best_val = y_test[best_idx] 
            best_vals = y_test[best_indices]
            all_best_vals.extend(best_vals)
            all_best_idxs.extend(test_idxs[best_indices])
            #all_best_vals.append(best_val)
            #all_best_idxs.append(test_idxs[best_idx])
            #print(f"Best value {best_val} found at {test_idxs[best_idx]}")

        with open('mbo_diverse_rf_data_num_training_points_'+str(num_training_points)+'_'+str(i)+'.pkl', 'wb') as f:
            pickle.dump({'all_best_vals': all_best_vals, 'all_best_idxs': all_best_idxs}, f)


if __name__ == '__main__':
    main()
