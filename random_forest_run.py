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

    for test_size in [50, 100, 150, 200, 300, 400, 500]:
        all_best_vals = []
        all_best_idxs = []
        for i in range(25):
            X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_size/len(outputs), random_state=i)
            #print(y_train)
            start_time = time.time()
            regr = RandomForestRegressor()
            regr.fit(X_test, y_test)
            best_idx = np.argmax(regr.predict(X_train))
            best_val = y_train[best_idx] 
            print(f"Best value {best_val} found at {best_idx}")
            all_best_vals.append(best_val)
            all_best_idxs.append(best_idx)
            with open('rf_methane_data_tsp'+str(test_size)+'_' + str(i)+'.pkl', 'wb') as f:
                pickle.dump({'all_best_vals': all_best_vals, 'all_best_idxs': all_best_idxs}, f)

if __name__ == '__main__':
    main()
