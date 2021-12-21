# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 06:27:18 2021

@author: Jiang Liu. The author would like to thank Mr. Chao Lei for his help optimizing the codes
"""

import math
import time
import numpy as np
import gurobipy as gp
import itertools 
import scipy.sparse as sp


startime = time.time()

mat_set = np.array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.],
       [1., 1., 0., 0.],
       [1., 0., 1., 0.],
       [1., 0., 0., 1.],
       [0., 1., 1., 0.],
       [0., 1., 0., 1.],
       [0., 0., 1., 1.],
       [1., 1., 1., 0.],
       [1., 1., 0., 1.],
       [1., 0., 1., 1.],
       [0., 1., 1., 1.]])

num_var = 4    # The number of variables 

num_clauses_num_var = 0   # The number of clauses with num_var  variables

for i in range(3):
    num_clauses_num_var = num_clauses_num_var + math.comb(num_var,i+1)  # For num_var many variables, we compute the number of clauses with at most 3  positive literals

for i in range(num_clauses_num_var-1):
    
    num_clauses =i+2         # The number of clauses of a positive 3-CNF
    comb_set_num_clauses = list(itertools.combinations(np.arange(num_clauses_num_var),num_clauses))    # The indices of clauses of positive 3-CNFs, in mat_set
    size_comb_set_num_clauses = math.comb(num_clauses_num_var,num_clauses)    # The number of positive 3-CNFs with num_clauses clauses and num_var variables
    
    print('There are %i many 3-CNFs for %i variables %i clauses' %(size_comb_set_num_clauses, num_var, num_clauses))
           
    LPR_num_SBS_num_clauses   = 0       # Count the number of SBS in 3-CNFs found by LPR, with num_clauses  clauses
    ILPR_num_SBS_num_clauses  = 0       # Count the number of SBS in 3-CNFs found by ILPR, with num_clauses  clauses
    Bopt_num_SBS_num_clauses  = 0       # Count the number of SBS in 3-CNFs found by binary optimization method, with num_clauses  clauses
        
    for i in range(size_comb_set_num_clauses):
        
        ind_temp_num_clauses = comb_set_num_clauses[i]
        temp_CNF = mat_set[ind_temp_num_clauses[0],:]
        
        for j in range(num_clauses-1):
            temp_CNF = np.vstack((temp_CNF, mat_set[ind_temp_num_clauses[j+1],:]))
            
        instance_matrix = temp_CNF  # Generating the instance matrix indexed by i, corresponding to a 3-CNF
        
        # Now, we solve SBS by LPR
        num_row, num_col = np.shape(instance_matrix)          # For a matrix, we compute its dimension information
        mat_prod     = np.dot(np.transpose(instance_matrix),instance_matrix)   # Compute  instance_matrix' * instance_matrix 
        diag_vec    = np.diagonal(mat_prod)          # Compute the diagonal of instance_matrix' * instance_matrix
    
        m = gp.Model()                    #  Build an optimization model named by m, having empty content
    
        x = m.addVars(num_col, num_col, vtype=gp.GRB.CONTINUOUS, name="x")    # Adding variables
        z = m.addVar(vtype=gp.GRB.CONTINUOUS, name="z")                       # Adding variables
        
        m.addConstrs((x[i,j]==x[j,i] for i in range(num_col) for j in range(i+1,num_col)),name="scon_")   # Constraints for symmetry
    
        m.addConstr(z==[0,1])  # Constraint for 0<= z <=1
    
        mask = np.eye(num_col)  # Generating an identity matrix
    
        x_diag_num_col_vec = gp.MVar([x[i, i] for i in range(num_col)])
        
        
        for i in range(num_col):
            x_ii_repeat_num_row_vec  = gp.MVar([x[i, i] for j in range(num_row)])
            x_ii_repeat_num_col_vec  = gp.MVar([x[i, i] for j in range(num_col)])    
            
            x_ith_col   = gp.MVar([x[i, j] for j in range(num_col)])  # Extract the i-th row that is same as the i-th columns of variable matrix x
        
            m.addConstr(x_ii_repeat_num_row_vec - instance_matrix @ x_ith_col == 0)  # Constraints instance_matrix * X = ones(m,1)*diag(X)^T       
            m.addConstr(mask @ x_ith_col >= 0)    # Constraints X[i,j] >=0
            m.addConstr(mask @ x_ith_col <= 1)    # Constraints X[i,j] <=1
            
        ones_m 	   = np.ones(num_row)
        ones_m_mat = sp.diags(ones_m)                
        z_mvar_col_vec  = gp.MVar([z for i in range(num_row)])
        
        m.addConstr(instance_matrix @ x_diag_num_col_vec  == ones_m_mat @ z_mvar_col_vec)  # Constraints instance_matrix * diag(X) = z * ones(m,1)   
          
        m.setObjective(z, gp.GRB.MAXIMIZE)      # Set optimization objective function
        
        def printSolution():                      # Define the optimization outputs
            if m.status == gp.GRB.OPTIMAL:
                print('\nThe LP optimum is : %g' % m.objVal)    
            else:
                print('No solution') 
            
        m.Params.Presolve   = 2  
    
        m.optimize()
        
        if  m.objVal > 1/2 :
            
            LPR_num_SBS_num_clauses = LPR_num_SBS_num_clauses  + 1
        
        # End of solving SBS by LPR
                
        # Now, we test iterated LPRs (ILPRs) method
        
        d = np.zeros(num_col)
        for i in range(num_col):
            d[i] = x[i,i].X        # Extract the diagonal value of x
        
        if  m.objVal > 1/2 :       # When the optimum is greater 0, the instance is satisfiable, we compute a satisfying assigment by ILPR
                                      
            U = set() 
        
            verified_set = set()
            iter_counter = 0
            
            op_order = np.arange(num_col)
            np.random.shuffle(op_order)
            
            for i in op_order:
                
                if x[i,i].X > 1/100 :
                    iter_counter += 1
                    U = U.union({i})
                    m.addConstr( x[i,i] == 1)        # If we found optimum x[i,i] > 1/100, we add constraint x[i,i] ==1         
                    
                    m.optimize()                    
                   
                    x_diag = np.array([round(x[i,i].X) for i in range(num_col)])
                    
                    if(all(instance_matrix @ x_diag)):
                        # notice, we did not assign this round sol to model's sol
                        break
            #print('############# The iteration number of ILPRs is %i $$$$$$$$$$$$$ \n' %i  )
        
            ld = np.zeros(num_col)
            
            for i in range(num_col):
                ld[i] = x[i,i].X                 
        
            cvn = np.dot(instance_matrix,np.round(ld))
        
            if  cvn.all() == np.ones(num_row).all():     # We check whether the 0-1 solution obtained by ILPR is a solution to the original problem
                ILPR_num_SBS_num_clauses = ILPR_num_SBS_num_clauses +1
                                 
        # end of the test of ILPRs
                
        print("\n****************************Binary Optimization********************************\n")
        
        # Solve SBS by Binary optimization
        
        mI = gp.Model()    
    
        mI.Params.Method = 3
    
        y=mI.addVars(num_col,vtype = gp.GRB.BINARY,name="y")
    
        mI.setObjective(sum(y[i]*diag_vec[i] for i in range(num_col)), gp.GRB.MINIMIZE)
    
        for j in range(num_row):
            mI.addConstr(sum(instance_matrix[j,k]*y[k] for k in range(num_col)) == 1)
        
        
        mI.optimize()
    
        if mI.status == gp.GRB.OPTIMAL:
            Bopt_num_SBS_num_clauses = Bopt_num_SBS_num_clauses  + 1   # We compute the number of SBS of 3-CNFs with num_clauses clauses
        
        # End of  solving SBS by Binary optimization
        
    if    LPR_num_SBS_num_clauses  !=  Bopt_num_SBS_num_clauses:    # We check if the LPR method can get the same number of SBS as the correct number for  3-CNFs with num_clauses clauses and num_var variables 
      print('\n LPR has a problem in %i variables with %i clauses' %(num_var, num_clauses))
      exit(0)
    elif  ILPR_num_SBS_num_clauses !=  Bopt_num_SBS_num_clauses:    # We check if the ILPR method can get the same number of SBS as the correct number for  3-CNFs with num_clauses clauses and num_var variables
      print('\n ILPR has a problem in %i variables with %i clauses' %(num_var, num_clauses))
      exit(0)        
    else: 
      print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')     
      print('\n Both LPR and ILPR are verified for 3-CNFs with %i variables and %i clauses' %(num_var, num_clauses))
 
endtime = time.time()

RT = endtime - startime

print('\n Running time is %g' %RT )
                