# Datasets

For practice purposes, sklearn provides some datasets.

```
from sklearn import datasets

ds = datasets.load_wine() # sklearn.utils.Bunch

X = ds.data # numpy.darray
y = ds.target # numpy.darray

```