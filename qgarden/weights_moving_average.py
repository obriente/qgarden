'''
weights_moving average: Implements the adaptive decoder technique described in arxiv:1712.02360

Author: Stephen Spitz
Licensed under the GNU GPL 3.0
'''

import numpy as np
from math import sqrt,log
from scipy.stats import norm

class weights_moving_average(object):
    
    def __init__(self,num_anc,lookback,window,pval):
        
        self.num_anc = num_anc
        self.lookback = lookback
        self.window = window
        self.pval = pval
        
        self.measurement_matrix = np.zeros(shape=(2,num_anc))
        self.syndrome_matrix = np.array([])
        
        self.xor_matrix = np.zeros(shape=(lookback,num_anc,num_anc))
        self.and_matrix = np.zeros(shape=(lookback,num_anc,num_anc))
        #self.freq = np.zeros(shape=(num_anc))
        
        self.var_matrix = np.zeros(shape=(lookback,num_anc,num_anc))
        self.qmat = np.zeros(shape=(lookback,num_anc,num_anc))
        self.boundary_q = np.zeros(shape=(num_anc))
                       
        
            
    def update_syndrome(self,new_measurement):
        
        #Calculate Derivative of Measurements
        new_syndrome = np.logical_xor(new_measurement,self.measurement_matrix[1]).astype(int)
        
        #Pop odd measurement, add new one
        self.measurement_matrix = np.vstack([new_measurement,np.delete(self.measurement_matrix,1,0)])
       
        if len(self.syndrome_matrix) == 0:
            self.syndrome_matrix = np.array([new_syndrome])
        elif len(self.syndrome_matrix) < self.window:
            self.syndrome_matrix = np.vstack([new_syndrome,self.syndrome_matrix])
        else:
            self.syndrome_matrix = np.vstack([new_syndrome,np.delete(self.syndrome_matrix,self.window - 1,0)])
    
    
    def averaging_function(self,dim):
        
                
        averaging_vec_0 = np.array([1]*dim)#np.exp(-10*np.array(range(dim))/dim)
        averaging_vec = averaging_vec_0 / np.sum(averaging_vec_0)
        
        return averaging_vec
        
    
    def update_xor_matrix(self):
        
        dim = len(self.syndrome_matrix)
        
        for t in range(self.lookback):
            for i in range(self.num_anc):
                for j in range(self.num_anc):
                    if t == 0 and i == j:
                        self.xor_matrix[t,i,j] = 0
                    else:
                        self.xor_matrix[t,i,j] = np.dot(self.averaging_function(dim-t),np.logical_xor(self.syndrome_matrix[:dim - t,i],self.syndrome_matrix[t:dim,j]).astype(int)) #/ (dim-t)
              
    
    def update_and_matrix(self):
        
        dim = len(self.syndrome_matrix)
        
        for t in range(self.lookback):
            for i in range(self.num_anc):
                for j in range(self.num_anc):
                    
                    self.and_matrix[t,i,j] = np.dot(self.averaging_function(dim-t),np.logical_and(self.syndrome_matrix[:dim - t,i],self.syndrome_matrix[t:dim,j]).astype(int)) #/ (dim-t)
        
        
    #def update_freq(self):
        
        
     #   for i in range(self.num_anc):
            
      #      self.freq[i] = np.sum(self.syndrome_matrix[:,i]) / self.window
    
    
    def update_varmat(self):
        
        for t in range(self.lookback):
            for i in range(self.num_anc):
                for j in range(self.num_anc):
                    if (1 - 2 * self.xor_matrix[t,i,j]) == 0:
                        self.var_matrix[t][i][j] = 0
                    else:
                        self.var_matrix[t][i][j] = (self.and_matrix[t,i,j] - self.and_matrix[0,i,i] * self.and_matrix[0,j,j]) / (1 - 2 * self.xor_matrix[t,i,j])
   
    def update_qmat(self):
        
        self.update_xor_matrix()
        self.update_and_matrix()
        #self.update_freq()
        self.update_varmat()
        
        for t in range(self.lookback):
            for i in range(self.num_anc):
                for j in range(self.num_anc):
                    Q = 1 - 4 * self.var_matrix[t][i][j]
                    if i == j and t == 0:
                        self.qmat[t][i][j] = 0
                    elif Q < 0:
                        self.qmat[t][i][j] = 0
                    elif 1- sqrt(Q) < 0:
                        self.qmat[t][i][j] = 0
                    else:
                        self.qmat[t][i][j] = self.sig_test(t,i,j)*(1 - sqrt(Q)) / 2
                        
                        
                
        self.update_boundary_vec()
   
    def sig_test(self,t,i,j):
        
        if t==0 and (j==i+1 or j==i-1):
            return 1
        elif t==1 and (j==i or j==i-1):
            return 1
        else:
            return 0
    
    #def sig_test(self,t,i,j):
        
    #    stderr = sqrt(self.and_matrix[t][i][j]*(1-self.and_matrix[t][i][j]) / len(self.syndrome_matrix))
        
    #    if stderr <= 0:
    #        test_stat = 0
    #    else:
    #        test_stat = (self.and_matrix[t][i][j] - self.and_matrix[0,i,i] * self.and_matrix[0,j,j]) / stderr
        
    #    if t < 2:
    #        return 1
    #    elif 1 - norm.cdf(test_stat) > self.pval:
    #        return 0
    #    else:
    #        return 1
        
        
        
            
                

    
    def return_weight_matrix(self):
        
        weight_matrix = np.zeros(shape=(self.num_anc*self.lookback,self.num_anc*self.lookback))
        boundary_weights = np.zeros(shape=(self.num_anc))
        
        self.update_qmat()
        
        for n in range(self.num_anc*self.lookback):
            for m in range(n,self.num_anc*self.lookback):
                t = m // self.num_anc - n // self.num_anc
                i = n - (n // self.num_anc) * self.num_anc
                j = m - (m // self.num_anc) * self.num_anc
        
                if t == 0 and i == j:
                    weight_matrix[n][m] = 0
                else:
                    weight_matrix[n][m] = self.qmat[t][i][j] * self.sig_test(t,i,j)
                    
        weight_matrix = weight_matrix + np.transpose(weight_matrix)
        
        
        weight_matrix = (np.linalg.inv(np.identity(self.num_anc*self.lookback) - weight_matrix))
        
        exact_boundary_q_left = [self.boundary_q[0]] + [0]*(self.num_anc - 1)
        exact_boundary_q_right = [0]*(self.num_anc - 1) + [self.boundary_q[-1]]
        
        #A_bound_vec = np.vstack([np.array(self.boundary_q) * np.array([1,0]),np.array(self.boundary_q) * np.array([0,1])]*self.lookback)
        
        A_bound_vec = np.transpose(np.vstack([np.array(exact_boundary_q_left*self.lookback),np.array(exact_boundary_q_right*self.lookback)]))
        
        boundary_weights = -np.log(np.max(np.dot(weight_matrix,A_bound_vec),axis=1)[:self.num_anc])
        
        for n in range(self.num_anc*self.lookback):
            for m in range(self.num_anc*self.lookback):
                
                if weight_matrix[n][m] <= 0:
                    weight_matrix[n][m] = 10000
                elif weight_matrix[n][m] >= 1:
                    weight_matrix[n][m] = .001
                else:
                    weight_matrix[n][m] = -log(weight_matrix[n][m])
       
        
        
        return weight_matrix, boundary_weights
         
        
      
    def update_boundary_vec(self):
        
        freq = np.array([self.and_matrix[0,i,i] for i in range(self.num_anc)])
        
        self.boundary_q = (1 - (1 - 2*freq) * np.prod(1 - 2*self.qmat[0],axis=1) / np.multiply(np.prod(np.prod(1 - 2*self.qmat,axis=0),axis=0),np.prod(np.prod(1 - 2*self.qmat,axis=0),axis=1)))/2
        
        def sig_test_boundary(i):
            if (i == 0 or i == self.num_anc - 1):
                return 1
            else:
                return 0
    
        boundary_filter = np.array([sig_test_boundary(i) for i in range(self.num_anc)])
        
        self.boundary_q = self.boundary_q * boundary_filter
        

        
            