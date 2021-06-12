# Bayesian optimization of nanoporous materials

#### This repository contains source code for the paper [Bayesian optimization of nanoporous materials](). The details for reproducing the results are given below:


- The main code of Bayesian optimization can be run by ```python bo_run.py```. The core logic for this code is built using [GpyTorch](https://github.com/cornellius-gp/gpytorch) and
[BoTorch](https://github.com/pytorch/botorch) libraries. 
- Code for One shot supervised learning (Random Forest: with and without diverse training set) is provided in ```random_forest_run.py``` ```diverse_random_forest_run.py```.
The core logic for this code is written using [Scikit-learn](https://github.com/scikit-learn/scikit-learn) library.
- To run Evolutionary search (CMA-ES) baseline, run ```python evolutionary_search_run.py```. The core logic for this baseline requires installing [CMA-ES](https://github.com/CMA-ES/pycma) package.
This code iterates over different choices of ```sigma``` and ```population size``` (two key parameters for instantiating CMA-ES search).  As mentioned in our paper, we found ```sigma=0.2``` and ```population size=20``` to be the best parameters. 
- Since each code generates a single file for each different random run of the method, we provide a simple wrapper ```compile_results_in_one_file.py``` to combine all the results into a single file.
- The code for generating figures is shown given in jupyter notebook ```cofs_results.ipynb```.

All the libraries required for the entire repository are given in ```requirements.txt``` file.


