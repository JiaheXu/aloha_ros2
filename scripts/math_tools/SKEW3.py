import numpy as np
def SKEW3(x):

    skew = np.zeros(( 3,3) )
    skew[0,1] =  -x[2]
    skew[1,0] =  x[2]
    
    skew[0,2] =  x[1]
    skew[2,0] =  -x[1]

    skew[1,2] =  -x[0]
    skew[2,1] =  x[0]

    # print("skew: ",skew)
    return skew
