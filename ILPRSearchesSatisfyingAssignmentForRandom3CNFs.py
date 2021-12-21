# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:26:08 2021

@author: LIU Jiang. The author would like to thank Mr. Chao Lei for his help optimizing the codes
"""


import time
import numpy as np
import scipy.sparse as sp
import colorama
from colorama import Fore, Back, Style
import gurobipy as gp


import math




def ConvertCoords(i, j, num_col):
    """ deal with symmetry of variable matrix

    variable matrix is constrained to symmetry matrix, In order to
    reduce the number of variables and introduce the symmetric property, 
    diagonal matrix is introduced. This function map the variable 
    at (i, j) in variable matrix to position after imposing new 
    storage scheme.

    Args:
        i:
            row     index of wanted variable in original variable matrix
        j:
            column  index of wanted variable in original variable matrix
        num_col: 
            number of columns of variable matrix

    Returns:
        A tuple of converted pos index of variable.

    
    """
    converted_i = 0
    converted_j = 0

    if(i >= j and i < num_col/2):
        converted_i = i
        converted_j = j
    elif(i > j and i >= num_col/2):
        converted_i = num_col-1-i
        converted_j = num_col-1-j
    elif(i == j and j >= math.ceil(num_col/2)):
        converted_i = i-math.ceil(num_col/2)
        converted_j = num_col
    elif(i < j):
        converted_i, converted_j = ConvertCoords(j, i, num_col)

    return converted_i, converted_j


def SolveP(input_path = None, coeff_mat_Param = None):
	""" solve the given coefficient optimized problem

	Args:
		input_path:
			path of matrix file produced by np.save
		coeff_mat_Param:
			path of matrix which has form of numpy array

	Notice:
		If both input_path and coeff_mat_Param are given, input_path will has
		the toppest priority. if neither of them are given, an m*6 random matrix
		are given for a test

	Return:
		A tuple with (result token, opt_value)
		result_token:
			0: solved, relating opt_value is the optimized value of current problem
			1: A counterexample for ILPR searching a SBS Asignment
			
		instance_type:
			1: SBS instance is indicated by instance_SBS=1
			1: ILPR instance is indicated by instance_ILPR =1
	"""

	# priority: input_path > coeff_mat

	startime = time.time()
	
	# ******************************** Declare statistical information ********************************
	instance_SBS	= -1
	instance_ILPR	= -1    
	result_token	= 0

	# ******************************** Initialize the coefficient matrix and its properties. ********************************
	num_col 	= 50 
	num_row     = np.random.randint(np.floor(0.4*num_col), np.ceil(0.80*num_col))  # Randomly choose the number of rows
	coeff_mat   = np.zeros((num_row, num_col))                                     # If there is not input matrix, then we start to generate cofeeficient matrix of (LS)
	if(input_path is not None):
		coeff_mat 			= np.loadtxt(input_path)
		num_row, num_col 	= np.shape(coeff_mat)
	elif(coeff_mat_Param is not None):
		coeff_mat			= coeff_mat_Param
		num_row, num_col	= np.shape(coeff_mat)
	else:
		# Assign the coeff_mat with a random matrix in which each row has 1, 2 or 3 many 1s
		for i in range(num_row):
			rand_nz_per_col_num = np.random.randint(1,4)
			rand_col_idx        = np.random.randint(0,num_col, size=rand_nz_per_col_num)

			for j in range(rand_nz_per_col_num):
				coeff_mat[i, rand_col_idx[j]] = 1

    # The randomly generated coeff_mat is the C matrix in the article

	# ******************************** gurobi ********************************
	model 		= gp.Model()                    # gurobi model
	
	model.Params.OutputFlag = 0                 # shut down gurobi screen output
	model.Params.Method = 1                   	# -1default，0simplex，1dual simplex，2Interior method，3(0,1,2)parallel， 4deterministic(0,1,2)parallel，5(0,1)parallel
	model.Params.Presolve   = 2                 # set preconditioning manner，-1 default，0 close，1 conservative，2 passitive
	
	vars_x = model.addVars(math.ceil(num_col/2), 
						num_col+1, 
						vtype=gp.GRB.CONTINUOUS, name="vars_x")
						# Add variables vars_x, vars_x is variable matrix stored in reduced form.
	
	var_z = model.addVar(vtype=gp.GRB.CONTINUOUS, name="var_z")     # Add variable vars_z

	model.setObjective(var_z, gp.GRB.MAXIMIZE)                      # Set objective function

	# Diagonal vec of variable matrix
	x_diag_Nc_vec = [vars_x[ConvertCoords(i, i, num_col)] 
						for i in range(num_col)]

	# Generate column needed by the following constraint 4
	x_diag_Nc_vec_mvar      = gp.MVar(x_diag_Nc_vec)
	z_mvar_col_vec          = gp.MVar([var_z
									for i in range(num_row)])
	ones_size_m 			= np.ones(num_row)
	ones_diag_m_mat 		= sp.diags(ones_size_m)

	# ******************************** Adding constraints ********************************
	# constraint 1: 0 <= z <= 1
	model.addConstr(var_z >= 0)
	model.addConstr(var_z <= 1)

	for i in range(num_col):   
		x_ith_col           = gp.MVar([vars_x[ConvertCoords(i, j, num_col)] for j in range(num_col)])
		# constraint 2: 0<= X[i,j] <=  1 , that is, x[i, j] \in [0, 1]
		model.addConstr( x_ith_col >= 0) 
		model.addConstr( x_ith_col <= 1)
	
		# constraint 3: C * X  = ones(m,1) * diag(X)^T 
		x_ii_repeat_Nr_vec  = gp.MVar([vars_x[ConvertCoords(i, i, num_col)] for j in range(num_row)])
		model.addConstr(coeff_mat @ x_ith_col           ==  x_ii_repeat_Nr_vec)

		# constraint 4: C * diag(X)= ones(m,1) * z
		model.addConstr(coeff_mat @ x_diag_Nc_vec_mvar  == ones_diag_m_mat @ z_mvar_col_vec)
				

	model.optimize()
	
	# ******************************** ILPR Searches a satisfying assignmemt ********************************
	if  model.objVal > 1/2 :
        
		instance_SBS = 1		
        
		U = set()

		iter_counter = 0
		# optimization logic
		# 1. using random trick to get operating-variable
		# 2. round_sol = round(sol), preverify whether round(sol) is the solution of the original problem: coeff_mat @ dig == 1, if true, end. (pending-implementation)
		op_order = np.arange(num_col)
		np.random.shuffle(op_order)

		ld = np.array([round(vars_x[ConvertCoords(i, i, num_col)].X) for i in range(num_col)])	# round(X_diag)
		cvn = coeff_mat@ld                                                                 		# coeff_mat@diag
		
		for i in op_order:
			# idx = random.randint(0, num_col)
			if vars_x[ConvertCoords(i,i,num_col)].X > 1/100 :
				iter_counter += 1
				U = U.union({i})
				model.addConstr( vars_x[ConvertCoords(i,i,num_col)] == 1)
				
				model.optimize()

				ld = np.array([round(vars_x[ConvertCoords(i, i, num_col)].X) for i in range(num_col)])
				cvn = coeff_mat@ld

				if(all(cvn)):
					# notice, we did not assign this round sol to model's sol
					break
		
		if(all(cvn)):
			instance_ILPR = 1
		else:
			np.savetxt('Acounterexample4ILPRearchingSBS_' + str(startime), coeff_mat)
			result_token = 1   	
	

	else:
		pass
		# indicate this instance is unSBS
        
	if result_token == 0:
		return result_token, model.objVal, instance_SBS, instance_ILPR
	elif result_token == 1:
		print(Fore.RED+'\n !!!!!!!!!!!!!!! A counterexample for ILPR searching SBS Assignments !!!!!!!!!!!!!!!!!!!!!!!')
		print(Style.RESET_ALL +'**********************End of iterated LP**********************')
		return result_token, -1, instance_SBS, instance_ILPR

	# ******************************** End of ILPR Searching ********************************

if __name__ == "__main__":
	num_SBS 	= 0    	     # For counting the number of SBS   instances
	num_ILPR 	= 0    	     # For counting the number of SBS solution found by iterated LPs

	total_num_ins = 500      # Set the number of the testing 3-CNF instances
    
	start_time			= time.time()

	for i in range(total_num_ins):
		print(i)
		# initialize coefficient matrix
		Nc = 500             # Set the number of variables in the testing  3-CNF instances
		Nr = np.random.randint(np.floor(0.4*Nc),np. ceil(0.80*Nc))       # Randomly generate the number of rows in the testing  3-CNF instances
		CM = np.zeros((Nr,Nc))
		
		for i in range(Nr):
			RandN = np.random.randint(1,4)
			RandI = np.random.randint(0,Nc,size=RandN)
			for j in range(RandN):
				CM[i,RandI[j]] = 1   # Generate a random Nr * Nc matrix in which each row has 1, 2 or 3 many 1s

		result_token, opt_val, instance_SBS, instance_ILPR = SolveP(None, CM)    # Computing the result using SolveP()
        
		if(result_token != 0):
			print("error")
		else:
			print("optimal value is: {}\n".format(opt_val))
		if(instance_SBS == 1):
			num_SBS 	+= 1       # Count the number of SBS instances
		if(instance_ILPR == 1):
			num_ILPR 	+= 1       # Count the number of SBS instances found a satisfying assignment by ILPR
	
	end_time = time.time()

	print("total running time is {}\n num of SBS is {} \n num of UnSBS is {}\n num of ILPR is {}".format(
		end_time - start_time,
		num_SBS,
		total_num_ins - num_SBS,
		num_ILPR                   # num_SBS = num_ILPR  implies that ILPR can always find a satisfying assignment for SBS instances in the experiment
	))
	