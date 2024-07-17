import preprocessor
import math

data = preprocessor.outputter()
#print('first entry:', data[0])

# kernels defined here. 
kernelOne = [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]] # pls be ok size (maybe 5x5 easier on computer)
kernelTwo = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
kernelThree = [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
kernelFour = [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]
kernelFive = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
kernelArr = [kernelOne, kernelTwo, kernelThree, kernelFour, kernelFive]

# padding, stride, and output size defined here
padding = float((len(kernelOne) - 1)/2) # equal to 1 (defined this way in case we change kernel dimensionality)
stride = 3 # larger stride = faster (reduce size in lower layers)
filteredSize = math.floor((len(data[0]) + 2 * padding - len(kernelOne))/stride + 1)
#print('output size:', filteredSize)

def convolute(): 
    transformer = []
    for i in range(0, len(data[0]), stride): # are we going to have to use thrice nested loops here? :( only looping in first entry for now
        # need to mind padding but getting type error when subtracting from len(data[0])
        entry = 0
        #print(data[0][i])
        # there's probably predefined numpy functions for handling what we need to do with filtering, fact it's in C should help with runtime

convolute()