import numpy as np
import pandas as pd 
import random as rand
import import_ipynb
from arrival_networkx import *

def binary_search(a, b, f):
    if f(a) == a:
        return [a]
    elif f(b) == b:
        return [b]
    else:
        mid = (a + b) // 2
        if f(mid) < mid:
            return binary_search(a, mid, f)
        elif f(mid) > mid:
            return binary_search(mid, b, f)
        else:
            return mid
        
        
def fixed_point(a, b, f):
    if len(a) >= 10:
        print(a,b)    
    d = len(a)
    if d == 1:
        return binary_search(a[0], b[0], lambda x: f([x])[0])
    else:
        mid = [(a[i] + b[i]) // 2 for i in range(d)]
        x_star = fixed_point(a[:-1], b[:-1], lambda x: f(x + [mid[-1]])[:-1])
        if f(x_star + [mid[-1]])[-1] > mid[-1]:
            return fixed_point(x_star + [mid[-1]], b, f)
        elif f(x_star + [mid[-1]])[-1] < mid[-1]:
            return fixed_point(a, x_star + [mid[-1]], f)
        else:
            return x_star + [mid[-1]]


def find_fixed_point(a, b, f):
    assert len(a) == len(b)
    return fixed_point(a, b, f)

instance = Arrival(10,True)
a = len(instance.X)*[0]
b = [2**len(instance.X) for i in range(len(instance.X))]
find_fixed_point(a, b, instance.evaluate)