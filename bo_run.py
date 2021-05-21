import torch
from botorch.models import FixedNoiseGP, SingleTaskGP
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
import numpy as np
import pickle
import sys
import time

def initialize_model(train_x, train_obj, covar_module=None, state_dict=None):
    # define models for objective and constraint
    if covar_module is not None:
        model = SingleTaskGP(train_x, train_obj, covar_module=covar_module)
    else:
        model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def main():
    n_init = 10
    n_evals = 500
    # Normalizing input features
    inputs = pickle.load(open('methane_storage.pkl', 'rb'))['inputs']
    for i in range(len(inputs[0])):
        inputs[:, i] = (inputs[:, i] - np.min(inputs[:, i]))/(np.max(inputs[:, i]) - np.min(inputs[:, i]))
    outputs = pickle.load(open('methane_storage.pkl', 'rb'))['outputs'].values
    outputs = ((outputs - np.min(outputs))/(np.max(outputs)-np.min(outputs)))
    DATA_SIZE = inputs.shape[0]

    # print(f"Input data size: {inputs.shape}")
    # print(f"Output data size: {outputs.shape}")
    for nrr in range(25): 
        initial_random_idxs = np.random.choice(np.arange((DATA_SIZE)), size=n_init, replace=False)
        train_x = torch.from_numpy(inputs[initial_random_idxs])
        train_y = torch.from_numpy(outputs[initial_random_idxs]).unsqueeze(-1)
        mll_ei, model_ei = initialize_model(train_x, train_y)#, covar_module)
        for num_iters in range(n_init, n_evals):
            print("Fitting GP ....")
            fit_gpytorch_model(mll_ei)
            print("Fitting GP finished.")
            EI = ExpectedImprovement(model_ei, best_f = train_y.max().item())
            EI_vals = EI(torch.from_numpy(inputs).unsqueeze(1)).detach().numpy()
            indices = np.argsort(EI_vals)[::-1]
            #print(indices)
            for idx in indices:
                if not torch.all(torch.from_numpy(inputs[idx]).unsqueeze(0) == train_x, axis=1).any(): # pick topmost already not in the dataset
                    best_next_input = torch.from_numpy(inputs[idx]).unsqueeze(0)
                    break
            train_x = torch.cat([train_x, best_next_input])
            train_y = torch.cat([train_y, torch.tensor(outputs[idx]).reshape(1, 1)])
            print(f"Iteration {num_iters}:")
            print(f"{idx}th point selected ", end='')
            print(f"with value: {train_y[-1].item()}")
            print(f"Best value found till now: {train_y.max().item()}")

            mll_ei, model_ei = initialize_model(train_x, train_y, state_dict = model_ei.state_dict())#  update model
            torch.save({'inputs_selected':train_x, 'outputs':train_y}, 'bo_data_run'+str(nrr)+'.pkl')

if __name__ == '__main__':
    main()
