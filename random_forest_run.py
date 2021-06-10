import numpy as np 
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
    outputs = ((outputs - np.min(outputs))/(np.max(outputs)-np.min(outputs)))
    rlist, maelist, mselist = [], [], []

    for i in range(10):
        all_best_vals = []
        all_best_idxs = []
        for test_size in [50, 100, 150, 200, 300, 400, 500]:
            test_size = test_size - len(all_best_vals)
            test_size = test_size // 2
            X_train, X_test, y_train, y_test, train_idxs, test_idxs = train_test_split(inputs, outputs, np.arange(len(outputs)), test_size=test_size/len(outputs), random_state=i)

            all_best_vals.extend(y_test)
            all_best_idxs.extend(test_idxs)

            start_time = time.time()
            regr = RandomForestRegressor()
            regr.fit(X_test, y_test)
            best_indices =  np.argsort(-regr.predict(X_train))[:test_size]
            #best_idx = np.argmax(regr.predict(X_train))
            #best_val = y_train[best_idx]
            best_vals = y_train[best_indices] 
            #print(f"Best value {best_val} found at {train_idxs[best_idx]}")
            #all_best_vals.append(best_val)
            #all_best_idxs.append(train_idxs[best_idx])
            all_best_vals.extend(best_vals)
            all_best_idxs.extend(train_idxs[best_indices])

            with open('mbo_50percent_rf_methane_data_tsp'+str(test_size)+'_' + str(i)+'.pkl', 'wb') as f:
                pickle.dump({'all_best_vals': all_best_vals, 'all_best_idxs': all_best_idxs}, f)

if __name__ == '__main__':
    main()
