from numpy import dot
from numpy.linalg import norm

import random
import re
import math
import numpy as np

def cos_sim(A, B):
    sim = dot(A, B)/(norm(A)*norm(B))
    sim = sim #* 10.0
    return sim

def cos_sim_per(A, B):
    sim = cos_sim(A,B)
    sim = (sim - (-1)) / (1-(-1))

    return sim * 100.0

def cos_sim_normal(A, B):
    return cos_sim(A,B)
    # sim = dot(A, B)/(norm(A)*norm(B))
    # # sim = sim * 10.0
    # # print('sim1',sim)
    # sim = (sim - (-1)) / (1-(-1))
    # # sim = 1 - math.cos(sim)
    
    # # print('sim2',sim)
    # return sim * 100.0

def normalization(l):
    for i in range(ord('ㄱ'),ord('ㅎ')+1):
        # print(chr(i))
        l = re.sub(chr(i)+'+',chr(i)+chr(i),l)
    
    return l

# print(normalization('ㅋㅋㅋㅋㅋㅎㅎㅎㅋㅋㅋㅋㅎㅎㅋㅋㅋㅋㅋ'))