import numpy as np

# types in the np array -> array dtype
# rank -> number of dimensions in the array
# shape -> tuple giving size of the array along each dimension
# ndarray -> used to represent both matrices(two dimensioanl array) and vectors(single dimension array).
# tensor -> 3d or higher dimensional arrays

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a[0])
print(f"Zeroes: {np.zeros(2)}")
print(f"Ones: {np.ones(2)}")
# empty creates array with random inital content
# reason to use empty over zeros is speed.
print(f"Empty: {np.empty(2)}")
print(f"Arange: {np.arange(4)}")
# specify data types
print(f"Ones with specfic data type: {np.ones(2, dtype=np.int64)}")
print(f"Sorted array: {np.sort(np.array([2,1,5,3,7,4,6,8]))}")

a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
print(f"Concatenate: {np.concatenate((a, b))}")
print(f"Concatenate on axis: {np.concatenate((a, b), axis=0)}")

array_example = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                         
                         [[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                         
                         [[0, 1, 2, 3],
                          [4, 5, 6, 7]]])
print(f"Number of axes: {array_example.ndim}")
print(f"Total number of elements: {array_example.size}")
print(f"Number of elements on each dimension: {array_example.shape}") # TODO: understand (3,2,4)

print(f"Reshape: {np.arange(6).reshape(3,2)}")

# np.newaxis -> increase dimensions of array by one
a = np.array([1,2,3,4,5,6])
a2 = a[np.newaxis, :]
print(f"Newaxis shape: {a2.shape}")

# np.expand_dims -> add an axis at index position
b = np.expand_dims(a, axis=1)
print(f"expand_dims: {b}")

print(f"values less than 4: {a[a < 4]}") 

data = np.array([1.0, 2.0])
print(f"array max aggregate: {data.max()}")
print(f"array min aggregate: {data.min()}")
print(f"array sum aggregate: {data.sum()}")

print(f"random numbers: {np.random.default_rng(0).random(3)}")
a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
print(f"unique values: {np.unique(a)}")
print(f"reverse array: {np.flip(a)}")


#################### PANDAS
import pandas as pd

print("FOLLOWING IS PANDAS")
s = pd.Series([1,3,5, np.nan, 6, 8])
print(s)

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
print(df)

print(f"DF head: {df.head()}")
print(f"DF index: {df.index}")
print(f"DF columns: {df.columns}")

import matplotlib.pyplot as plt

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()

