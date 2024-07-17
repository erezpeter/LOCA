import numpy as np



####################################################################
# Defining the non-linear function used in experiment 6A,6B and 6C #
####################################################################

# The non linear function that is used in 




# Diffusion maps implementation. The function gets as input a pairwise distance matrix, and estimates its kernel width
# using a scaled version of the maximal 1-nearest neighbor distance of the dataset.
# Then, it estimated and returns the probability transition matrix defined in diffusion maps.
# Input:
#       d_in - n-by-n matrix that includes the distances between any pairs of points with a dataset with n points.
#       fac - The factor that will divide 
def dm_from_dist(d_in,fac=1,alpha=0.):
    sort_dist=np.sort(d_in)
    epsilon = fac*2*np.max(sort_dist[:,1])
    
    pdx = (d_in.T / epsilon).T
    K0 = np.exp(-1 * pdx)
    
    
    Pmat0= 1.
    if alpha> 1e-8:
        d_sum0=(np.sum(K0,axis=1).reshape(-1,1))**alpha
        Pmat0=np.matmul(d_sum0,np.transpose(d_sum0))
    
    K1=K0/Pmat0
    
    d_sum1=(np.sum(K1,axis=1).reshape(-1,1))**(-1)
    d_sum1=np.diag(d_sum1.reshape(-1))
    P=np.matmul(d_sum1,K1)
    return P


# The function gets two sets of points and samples points points on the line that connects them
# Input:
#    arA, arB -  n-by-d arrays. Each includes n samples in dimension d
#    numSamples -  Int. The amount of samples that will be sampled on the line that connects each corresponding points.
#    
# Output:
#    n-by-numSamples-by- d array. The output tensor in index [i,j,k] will include the j-th point on a line that connects the
#                                 i-th sample from arA and arB.
def interpolate(arA, arB, numSamples= 100):
    inter= np.zeros((np.shape(arA)[0],numSamples,np.shape(arA)[-1]))
    for i,alpha in enumerate(np.linspace(0.,1.,numSamples)):
        inter[:,i,:]= arA*alpha+ arB*(1-alpha)
    return inter




# Find an affine transformation of data b (of the form f(x)= R*X+ bias, 
# where R is an orthogonal matrix and bias is a vector) that 
# minimize the least squares problems between the two datasets.
#
# If scaling is True then the parameters that will be returned are of f(x)=a*R*X+bias
# where a is a scale
#
# Input:
#       - data_a, data_b - n-br-d arrays. Two datasets that contain n points in d-dimensional space. Each datapoint in data_a 
#                                         corresponds to the datapoint in data_b that shares the same index.
#       - scaling- Boolean. 
# 
# Output- 
#         R - d-by-d matrix that corresponds to the estimated f.
#         bias - 1-by-d matrix that corresponds to the estimated f.
#         scaling_fac (Only if scaling is True) - a scalar that corresponds to the estimated f
#
def calibrate_data_b(data_a, data_b,scaling=False):
    if len(np.shape(data_a))!=2 and len(np.shape(data_b))!=2:
        raise Exception('Data should be a 2-d tensor of nxd')
    
    data_a_no_bias= data_a - np.mean(data_a, axis=0)
    data_b_no_bias= data_b - np.mean(data_b, axis=0)
    U,_,VT = np.linalg.svd(np.dot(data_a_no_bias.T, data_b_no_bias))
    
    old_bias= np.mean(data_b, axis=0, keepdims=True)
    new_bias= np.mean(data_a, axis=0, keepdims=True)
    
    R= np.dot(VT.T, U.T)    
        
    
    
    if scaling:
        scaling_fac= np.sum(data_a_no_bias* np.matmul(data_b_no_bias,R))/ np.sum(data_b_no_bias**2)
        bias =  new_bias- np.dot(old_bias,R)*scaling_fac
        return R, bias, scaling_fac
    
    bias= new_bias- np.dot(old_bias,R)
    
    return R, bias

# The function gets two sets of pairwise distances, and estimates the scaling factor on the first set of distances that will
# minimize the mean squared difference between the first scaled set and the second set. i.e. find a scalar 'a' 
# that will minimize the functional  \sum_i (a*distsA_i - distsB_i)^2
#
# Input: 
#       distsA, distsB - m-by-1 martix.
#      
# Output:
#        a scalar.
def get_calibrate_factor_for_distsA(distsA, distsB):
    return np.sum(distsB*distsA)/np.sum(distsA**2)



