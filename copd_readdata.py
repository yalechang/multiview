import numpy as np
import csv
from sklearn.preprocessing import scale
from sklearn.cluster import spectral_clustering
from python.multiview.utils.compute_affinity import compute_affinity
from python.multiview.utils.opt_affinity_weight import opt_affinity_weight
from sklearn.metrics import normalized_mutual_info_score as nmi
from time import time
import pickle

file_name = '/Users/changyale/dataset/COPDGene/'+\
        'CG2500UnadjustedForJRandJD_minusNA_Standardized.csv'

file_case = '/Users/changyale/dataset/COPDGene/Training.csv'
file_control = '/Users/changyale/dataset/COPDGene/Test.csv'

file_use = file_control

# Load csv file
file_csv = open(file_use,'rb')
csvreader = csv.reader(file_csv)
lines = [line for line in csvreader]
file_csv.close()

n_instances = len(lines)-1
n_features = len(lines[0])-2

# feature names
features_name = lines[0][1:-1]

case_id = []
code_group = []
data = []

for i in range(n_instances):
    case_id.append(lines[i+1][0])
    code_group.append(lines[i+1][-1])
    data.append(lines[i+1][1:-1])

data = np.double(np.array(data))

# random choose 1000 samples
idx = np.random.choice(n_instances,1000,replace=False)

# Normalization
data = scale(data[idx,:])

# Compute affinity matrix for each feature
affs = np.zeros((n_features,n_instances,n_instances))
flag_sigma = 'global'
for j in range(n_features):
    #sigma,affs[j] = compute_affinity(data[:,j].reshape(n_instances,1),\
    #        flag_sigma=flag_sigma)
    #print j
    pass

# Save data file
file_pkl = open("copd_control.pkl","wb")
pickle.dump([data,features_name],file_pkl)
file_pkl.close()

