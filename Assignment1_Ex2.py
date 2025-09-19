from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 

#question 9 
def ReluActivation(FeatureMap):
    relu = np.maximum(0, FeatureMap)
    return relu

def MaxPooling(FeatureMap):

    return 0 