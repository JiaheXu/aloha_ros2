import numpy as np
def VECTORIZE(skewMat):
    
    # vec = [-skewMat(2,3); skewMat(1,3); -skewMat(1,2)];

    vec = np.array( [-skewMat[1,2], skewMat[0,2], -skewMat[0,1] ] )
    return vec
