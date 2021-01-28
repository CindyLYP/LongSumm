import logging
import os
import re
import numpy as np
from scipy.special import softmax

a = np.array([[1,2],[3,4]])
b = np.array([3,5,1,2])
c = ['1234', 'asdf', 'gggg', 'hhhhh']
print(a+a.T-np.diag(a.diagonal()))
idx = list(np.argsort(-b))
for i in range(3):
    print(c[idx[i]])
