import logging
import os
import re
import numpy as np
from scipy.special import softmax

a = np.array([[0,1,2], [1,0,4],[2,4,0]])

print(softmax(a, axis=1))
