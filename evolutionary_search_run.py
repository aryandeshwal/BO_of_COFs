import cma
import pickle
import numpy as np
import time
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def main():
    inputs = pickle.load(open('methane_storage.pkl', 'rb'))['inputs']
    for i in range(len(inputs[0])):
        inputs[:, i] = (inputs[:, i] - np.min(inputs[:, i]))/(np.max(inputs[:, i]) - np.min(inputs[:, i]))
    outputs = pickle.load(open('methane_storage.pkl', 'rb'))['outputs'].values
    outputs = ((outputs - np.min(outputs))/(np.max(outputs)-np.min(outputs)))
    # CMA-ES requires two main params: sigma and population size
    sigma_list = [0.2]# [0.1, 0.2, 0.5]
    pop_list = [20]# [20, 50, 100]
    for i in range(25):
        print(f"Iteration {i}**************")
        x_init = np.random.rand(12)
        for popsize in pop_list:
            for sigma in sigma_list:
                cont_bounds = [0, 1]
                start_time = time.time()
                es = cma.CMAEvolutionStrategy(x0=x_init,sigma0=sigma,inopts={'bounds': cont_bounds, "popsize": popsize},)
                final_outputs = []
                final_inputs = []
                #while not es.stop():
                for iter in range(int(20000/popsize)):
                    xs = es.ask()
                    temp_time = time.time()
                    Y = []
                    X = []
                    for x in xs:
                        # find the closest input
                        idx = closest_node(x, inputs)
                        X.append(inputs[idx])
                        Y.append(-1*outputs[idx])
                        #Y.append(-1*outputs[closest_node(x, inputs)])
                    es.tell(X, Y)  # return the result to the optimizer
                    final_outputs.append(Y)
                    final_inputs.append(X)
                    if iter % 50 == 0:
                        print(f"CMA-ES iteration:{iter}")
                        print("current best")
                        print(f"{es.best.f}")
                best_x = es.best.x
                #print(best_x)
                with open('es_run_'+str(i)+'_' + 'sigma_' + str(sigma) + '_popsize_'+ str(popsize) +'.pkl', 'wb') as f:
                    pickle.dump({'final_outputs':final_outputs, 'final_inputs': final_inputs}, f)

if __name__ == '__main__':
    main()
