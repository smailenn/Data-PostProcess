#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 
from numpy import savetxt
from scipy import stats
from scipy import signal
from scipy import integrate
import os

bx = [1, 1, 1, 1, 1, 1]
#time = [1, 2, 3, 4, 5, 6]
time = [0, .1, .2, .3, .4, .5]

int = integrate.simps(bx,time)
print(int)
int2 = np.sum(int)
print(int2)