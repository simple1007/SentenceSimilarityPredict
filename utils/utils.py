from numpy import dot
from numpy.linalg import norm

import random
import numpy as np

def cos_sim(A, B):
    sim = dot(A, B)/(norm(A)*norm(B))
    return sim
