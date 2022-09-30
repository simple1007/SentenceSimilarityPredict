from numpy import dot
from numpy.linalg import norm

import random
import re
import numpy as np

def cos_sim(A, B):
    sim = dot(A, B)/(norm(A)*norm(B))
    sim = sim * 10.0
    return sim

def normalization(l):
    for i in range(ord('ㄱ'),ord('ㅎ')+1):
        # print(chr(i))
        l = re.sub(chr(i)+'+',chr(i)+chr(i),l)
    
    return l

# print(normalization('ㅋㅋㅋㅋㅋㅎㅎㅎㅋㅋㅋㅋㅎㅎㅋㅋㅋㅋㅋ'))