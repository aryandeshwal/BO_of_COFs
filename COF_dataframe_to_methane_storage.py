# code to convert from the original dataframe dataset 'properties.csv' to 'methane_storage.pkl'

import numpy as np 
import pickle
import pandas as pd

df = pd.read_csv('properties.csv')
# we remove Column 48225 since it is an outlier for void fraction feature  # can be seen by df[df[' void fraction [widom]'] > 1]
df = df.drop(48225)

features = [' void fraction [widom]', ' density [kg/m^3]', ' largest included sphere diameter [A]', 
            ' largest free sphere diameter [A]', ' surface area [m^2/g]']
extra_features =  [' num carbon', ' num fluorine', ' num hydrogen', ' num nitrogen', ' num oxygen', ' num sulfur', ' num silicon'] 

inputs = np.concatenate([df[features].values, (df[extra_features].values)/(df[' supercell volume [A^3]'].values.reshape(-1, 1))], axis=1)

outputs = df[' deliverable capacity [v STP/v]']


with open('methane_storage.pkl', 'wb') as file:
    pickle.dump({'inputs': inputs, 'outputs': outputs}, file)
