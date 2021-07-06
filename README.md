Python code to reproduce all plots in:

> ❝Bayesian optimization of nanoporous materials❞
> A. Deshwal, C. Simon, J. R. Doppa.
> ChemRxiv. (2021) [DOI](https://chemrxiv.org/engage/chemrxiv/article-details/60d2c7d7e211337735e056e2)

## requirements

the Python 3 libraries required for the project are in `requirements.txt`. use Jupyter Notebook or Jupyter Lab to run Python 3 in the `*.ipynb`.

# search methods

## step 1: prepare the data

our paper relies on data from Mercado et al. [here](https://pubs.acs.org/doi/10.1021/acs.chemmater.8b01425). visit [Materials Cloud](https://archive.materialscloud.org/record/2018.0003/v2) to download and untar `properties.tgz`. place `properties.csv` in the main directory.

run the code in the Jupyter Notebook `prepare_Xy.ipynb` to prepare the data and write `inputs_and_outputs.pkl` to be read in by other Notebooks.

## step 2: run the searches

run the following Jupyter Notebooks, which will write search results to `.pkl` files.
* `random_search.ipynb` for random search
* `evol_search.ipynb` for evolutionary search (CMA-ES)
* `random_forest_run.ipynb` for one-shot supervised machine learning (via random forests). run twice, one with the flag `diversify_training = True`, the other with `diversify_training = False`.
* `BO_run.ipynb` for Bayesian optimization. run three times, with `which_acquisition` set to `"EI"`, `"max y_hat"`, and `max sigma`.

each `.ipynb` can be run on a desktop computer. the BO code takes the longest, at ~10 min per run.

[BoTorch](https://github.com/pytorch/botorch) libraries. 

## step 3: visualize the results

finally, run `viz.ipynb` to read in the `*.pkl` files and visualize the results.


# toy GP illustrations
see `synthetic_example.ipynb` for the toy GP plots in the paper.
