import h5py
import pandas as pd

filename = 'data/generated_synthetic_data/valid.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
c = f[u'label']
b = f[u'data']
print(c.shape)