from pyGPGOMEA import GPGOMEARegressor as GPG
from sklearn.datasets import load_boston
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler as SS
import os, sys

import matplotlib.pyplot as plt
from pandas import DataFrame
from hyperVolume import HyperVolume
from scipy import stats
class Linear:
	def __init__(self):	
		self.max_run = 30;
		self.pop_size = 1000;
		self.iterations = 1;
		self.init_seed = 42;
		self.max_explen = 100;
		self.crossover_proportion = 0.5;
		self.mutation_proportion = 0.5;
		self.prob="multiobj";
		self.multiobj="symbreg_size";
		self.methodtype="alpha_dominance";
		self.alphafunction="linear";
		self.tournament = 7;
		self.algoframework="NSGA2";
		self.abbr_name = "alpha_dominance_linear";
		#string used for outputing information
		self.RunName_best = "linear_best_run = "
		self.RunName_worse = "linear_worse_run = "
		self.RunName_median = "linear_median_run = "
		self.HVList_name = "HV_list_linear = "
		#for recording info after run the algorithm
		self.bestHV = -1
		self.bestid = -1
		self.worseHV = -1
		self.worseid = -1
		self.medianHV = -1
		self.medianid = -1
		self.averageHV = -1
		self.HV_std = -1
		self.HV_list = []
		#for recording info after run the algorithm
		self.bestHV_test = -1
		self.bestid_test = -1
		self.worseHV_test = -1
		self.worseid_test = -1
		self.medianHV_test = -1
		self.medianid_test = -1
		self.averageHV_test = -1
		self.HV_std_test = -1
		self.HV_list_test = []
class Sigmoid:
	def __init__(self):	
		self.max_run = 30;
		self.pop_size = 500;
		self.iterations = 100;
		self.init_seed = 42;
		self.max_explen = 100;
		self.crossover_proportion = 0.5;
		self.mutation_proportion = 0.5;
		self.prob="multiobj";
		self.multiobj="symbreg_size";
		self.methodtype="alpha_dominance";
		self.alphafunction="sigmoid";
		self.tournament = 7;
		self.algoframework="NSGA2";
		self.abbr_name = "alpha_dominance_sigmoid";
		#string used for outputing information
		self.RunName_best = "sigmoid_best_run = "
		self.RunName_worse = "sigmoid_worse_run = "
		self.RunName_median = "sigmoid_median_run = "
		self.HVList_name = "HV_list_sigmoid = "
		#for recording info after run the algorithm
		self.bestHV = -1
		self.bestid = -1
		self.worseHV = -1
		self.worseid = -1
		self.medianHV = -1
		self.medianid = -1
		self.averageHV = -1
		self.HV_std = -1
		self.HV_list = []
		#for recording info after run the algorithm
		self.bestHV_test = -1
		self.bestid_test = -1
		self.worseHV_test = -1
		self.worseid_test = -1
		self.medianHV_test = -1
		self.medianid_test = -1
		self.averageHV_test = -1
		self.HV_std_test = -1
		self.HV_list_test = []
class Cosine:
	def __init__(self):	
		self.max_run = 30;
		self.pop_size = 500;
		self.iterations = 100;
		self.init_seed = 42;
		self.max_explen = 100;
		self.crossover_proportion = 0.5;
		self.mutation_proportion = 0.5;
		self.prob="multiobj";
		self.multiobj="symbreg_size";
		self.methodtype="alpha_dominance";
		self.alphafunction="cosine";
		self.tournament = 7;
		self.algoframework="NSGA2";
		self.abbr_name = "alpha_dominance_cosine";
		#string used for outputing information
		self.RunName_best = "cosine_best_run = "
		self.RunName_worse = "cosine_worse_run = "
		self.RunName_median = "cosine_median_run = "
		self.HVList_name = "HV_list_cosine = "
		#for recording info after run the algorithm
		self.bestHV = -1
		self.bestid = -1
		self.worseHV = -1
		self.worseid = -1
		self.medianHV = -1
		self.medianid = -1
		self.averageHV = -1
		self.HV_std = -1
		self.HV_list = []
		#for recording info after run the algorithm
		self.bestHV_test = -1
		self.bestid_test = -1
		self.worseHV_test = -1
		self.worseid_test = -1
		self.medianHV_test = -1
		self.medianid_test = -1
		self.averageHV_test = -1
		self.HV_std_test = -1
		self.HV_list_test = []
class Duplicate_control:
	def __init__(self):	
		self.max_run = 30;
		self.pop_size = 500;
		self.iterations = 100;
		self.init_seed = 42;
		self.max_explen = 100;
		self.crossover_proportion = 0.5;
		self.mutation_proportion = 0.5;
		self.prob="multiobj";
		self.multiobj="symbreg_size";
		self.methodtype="Duplicate_control";
		self.alphafunction="none";
		self.tournament = 7;
		self.algoframework="NSGA2";
		self.abbr_name = "Duplicate_control";
		#string used for outputing information
		self.RunName_best = "Duplicate_control_best_run = "
		self.RunName_worse = "Duplicate_control_worse_run = "
		self.RunName_median = "Duplicate_control_median_run = "
		self.HVList_name = "HV_list_Duplicate_control = "
		#for recording info after run the algorithm
		self.bestHV = -1
		self.bestid = -1
		self.worseHV = -1
		self.worseid = -1
		self.medianHV = -1
		self.medianid = -1
		self.averageHV = -1
		self.HV_std = -1
		self.HV_list = []
		#for recording info after run the algorithm
		self.bestHV_test = -1
		self.bestid_test = -1
		self.worseHV_test = -1
		self.worseid_test = -1
		self.medianHV_test = -1
		self.medianid_test = -1
		self.averageHV_test = -1
		self.HV_std_test = -1
		self.HV_list_test = []
class Adaptive_alpha:
	def __init__(self):	
		self.max_run = 30;
		self.pop_size = 500;
		self.iterations = 100;
		self.init_seed = 42;
		self.max_explen = 100;
		self.crossover_proportion = 0.5;
		self.mutation_proportion = 0.5;
		self.prob="multiobj";
		self.multiobj="symbreg_size";
		self.methodtype="adaptive_alpha_dominance";
		self.alphafunction="none";
		self.tournament = 7;
		self.algoframework="NSGA2";
		self.abbr_name = "adaptive_alpha";
		#string used for outputing information
		self.RunName_best = "adaptive_alpha_best_run = "
		self.RunName_worse = "adaptive_alpha_worse_run = "
		self.RunName_median = "adaptive_alpha_median_run = "
		self.HVList_name = "HV_list_adaptive_alpha = "
		#for recording info after run the algorithm
		self.bestHV = -1
		self.bestid = -1
		self.worseHV = -1
		self.worseid = -1
		self.medianHV = -1
		self.medianid = -1
		self.averageHV = -1
		self.HV_std = -1
		self.HV_list = []
		#for recording info after run the algorithm
		self.bestHV_test = -1
		self.bestid_test = -1
		self.worseHV_test = -1
		self.worseid_test = -1
		self.medianHV_test = -1
		self.medianid_test = -1
		self.averageHV_test = -1
		self.HV_std_test = -1
		self.HV_list_test = []
class SPEA2:
	def __init__(self):	
		self.max_run = 30;
		self.pop_size = 500;
		self.iterations = 100;
		self.init_seed = 42;
		self.max_explen = 100;
		self.crossover_proportion = 0.5;
		self.mutation_proportion = 0.5;
		self.prob="multiobj";
		self.multiobj="symbreg_size";
		self.methodtype="none";
		self.alphafunction="none";
		self.tournament = 7;
		self.algoframework="SPEA2";
		self.abbr_name = "SPEA2";
		#string used for outputing information
		self.RunName_best = "SPEA2_best_run = "
		self.RunName_worse = "SPEA2_worse_run = "
		self.RunName_median = "SPEA2_median_run = "
		self.HVList_name = "HV_list_SPEA2 = "
		#for recording info after run the algorithm
		self.bestHV = -1
		self.bestid = -1
		self.worseHV = -1
		self.worseid = -1
		self.medianHV = -1
		self.medianid = -1
		self.averageHV = -1
		self.HV_std = -1
		self.HV_list = []
		#for recording info after run the algorithm
		self.bestHV_test = -1
		self.bestid_test = -1
		self.worseHV_test = -1
		self.worseid_test = -1
		self.medianHV_test = -1
		self.medianid_test = -1
		self.averageHV_test = -1
		self.HV_std_test = -1
		self.HV_list_test = []
class NSGA2:
	def __init__(self):	
		self.max_run = 30;
		self.pop_size = 500;
		self.iterations = 100;
		self.init_seed = 42;
		self.max_explen = 100;
		self.crossover_proportion = 0.5;
		self.mutation_proportion = 0.5;
		self.prob="multiobj";
		self.multiobj="symbreg_size";
		self.methodtype="classicNSGA2";
		self.alphafunction="none";
		self.tournament = 7;
		self.algoframework="NSGA2";
		self.abbr_name = "classicNSGA2";
		#string used for outputing information
		self.RunName_best = "NSGA2_best_run = "
		self.RunName_worse = "NSGA2_worse_run = "
		self.RunName_median = "NSGA2_median_run = "
		self.HVList_name = "HV_list_NSGA2 = "
		#for recording info after run the algorithm
		self.bestHV = -1
		self.bestid = -1
		self.worseHV = -1
		self.worseid = -1
		self.medianHV = -1
		self.medianid = -1
		self.averageHV = -1
		self.HV_std = -1
		self.HV_list = []
		#for recording info after run the algorithm
		self.bestHV_test = -1
		self.bestid_test = -1
		self.worseHV_test = -1
		self.worseid_test = -1
		self.medianHV_test = -1
		self.medianid_test = -1
		self.averageHV_test = -1
		self.HV_std_test = -1
		self.HV_list_test = []		
class LengthControl_interpol:
	def __init__(self):	
		self.max_run = 30;
		self.pop_size = 500;
		self.iterations = 100;
		self.init_seed = 42;
		self.max_explen = 100;
		self.crossover_proportion = 0.5;
		self.mutation_proportion = 0.5;
		self.prob="multiobj";
		self.multiobj="symbreg_size";
		self.methodtype="none";
		self.alphafunction="none";
		self.tournament = 7;
		self.algoframework="LengthControlTruncation";
		self.abbr_name = "LC_interpol";
		#string used for outputing information
		self.RunName_best = "LengthControl_interpol_best_run = "
		self.RunName_worse = "LengthControl_interpol_worse_run = "
		self.RunName_median = "LengthControl_interpol_median_run = "
		self.HVList_name = "LengthControl_interpol_list_NSGA2 = "
		#for recording info after run the algorithm
		self.bestHV = -1
		self.bestid = -1
		self.worseHV = -1
		self.worseid = -1
		self.medianHV = -1
		self.medianid = -1
		self.averageHV = -1
		self.HV_std = -1
		self.HV_list = []
		#for recording info after run the algorithm
		self.bestHV_test = -1
		self.bestid_test = -1
		self.worseHV_test = -1
		self.worseid_test = -1
		self.medianHV_test = -1
		self.medianid_test = -1
		self.averageHV_test = -1
		self.HV_std_test = -1
		self.HV_list_test = []		
		
def load_data(DataSetFileName):
	
	if DataSetFileName == "boston":
		X, y = load_boston(return_X_y=True)
		X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
		scaler_X = SS()
		scaler_y = SS()
	else:
		raw = np.loadtxt(DataSetFileName, delimiter=' ') 
		X = raw[:,:-1]
		y = np.expand_dims(raw[:,-1],axis = 1)
		X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=88)
		scaler_X = SS()
		scaler_y = SS()
		
	X_train_scaled = scaler_X.fit_transform(X_train)
	y_train_scaled = scaler_y.fit_transform(y_train.reshape((-1,1)))
	X_test_scaled = scaler_X.transform(X_test)
	y_test_scaled = scaler_y.transform(y_test.reshape((-1,1)))
	
	return scaler_X,scaler_y,X_train_scaled,y_train_scaled,X_test_scaled,y_test_scaled


	
def main():

	
	X, y = load_boston(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
	scaler_X = SS()
	scaler_y = SS()
	X_train_scaled = scaler_X.fit_transform(X_train)
	y_train_scaled = scaler_y.fit_transform(y_train.reshape((-1,1)))
	X_test_scaled = scaler_X.transform(X_test)
	y_test_scaled = scaler_y.transform(y_test.reshape((-1,1)))



	max_run = 30
	pop_size = 500
	iterations = 100
	init_seed = 42
	max_explen = 100
	crossover_proportion = 0.5
	mutation_proportion = 0.5
	multiobj="symbreg_size"
	
	prob="multiobj"
	methodtype="Duplicate_control"
	alphafunction="none"
	tournament = 7
	algoframework="NSGA2"

#===============settings of running all comparison methods====================
#	prob		algoframework		methodtype					alpha_function	
#	multiobj	NSGA2				NA							NA					
#	multiobj	NSGA2				alpha_dominance				cosine				
#	multiobj	NSGA2				alpha_dominance				linear
#	multiobj	NSGA2				alpha_dominance				sigmoid
#	multiobj	NSGA2				adaptive_alpha_dominance	NA
#	multiobj	SPEA2				NA							NA
#	multiobj	evoNSGA2			NA							NA
#	multiobj	NSGA2DP				NA							NA
#=============================================================================


	for i in range(1,max_run+1):
		print('now is the '+str(i)+"th instances")
		ea = GPG(gomea = False, caching = False, ims = False, time=999999, elitism=0, generations= iterations, seed = init_seed+i, silent = False, prob = prob, multiobj=multiobj, popsize=pop_size, linearscaling=True, erc=True, maxsize = max_explen, methodtype=methodtype, alphafunction= alphafunction, numofrun=i, tournament = tournament, algoframework =algoframework, functions="+_-_*_p/_sqrt_plog", subcross=crossover_proportion,submut=mutation_proportion)
		ea.fit(X_train_scaled, y_train_scaled)
		print('Test RMSE:', np.sqrt( scaler_y.var_ * mean_squared_error(y_test_scaled, ea.predict(X_test_scaled)) ))
		print('training RMSE:', np.sqrt( scaler_y.var_ * mean_squared_error(y_train_scaled, ea.predict(X_train_scaled)) ))
		
		archive_on_test_set = ea.get_final_archive(X_test_scaled)
		test_preds = [x[0] for x in archive_on_test_set]
		test_mses = [mean_squared_error(y_test_scaled, p) for p in test_preds]
		test_sizes = [x[2] for x in archive_on_test_set]
		popu_test = [test_mses,test_sizes]
		popu_test = DataFrame(popu_test).T.values.tolist()
		np.savetxt("PF_test//PF__"+methodtype+"_archive_runs"+str(i)+".txt", popu_test, fmt="%f %d", delimiter=" ",newline="\r\n")
		test_sizes = (np.array(test_sizes).astype(float) - 1) / (100-1)
		popu_test = [test_mses,test_sizes]
		ea.reset()
	

if __name__ == '__main__':
    main()
