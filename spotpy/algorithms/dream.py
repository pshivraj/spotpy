# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska and Motjaba Sadegh

This class holds the Differnetial Evolution adaptive Metropolis (DREAM) algorithm.
'''
import spotpy
from . import _algorithm
import numpy as np
import time

class dream(_algorithm):
    '''
    Implements the Differnetial Evolution adaptive Metropolis algorithm.
    Caution: This algorithm is still under developement.
    
    Input
    ----------
    spot_setup: class
        model: function 
            Should be callable with a parameter combination of the parameter-function 
            and return an list of simulation results (as long as evaluation list)
        parameter: function
            When called, it should return a random parameter combination. Which can 
            be e.g. uniform or Gaussian
        objectivefunction: function 
            Should return the objectivefunction for a given list of a model simulation and 
            observation.
        evaluation: function
            Should return the true values as return by the model.
            
    dbname: str
        * Name of the database where parameter, objectivefunction value and simulation results will be saved.
    
    dbformat: str
        * ram: fast suited for short sampling time. no file will be created and results are saved in an array.
        * csv: A csv file will be created, which you can import afterwards.        
        
    save_sim: boolean
        *True:  Simulation results will be saved
        *False: Simulationt results will not be saved
     '''
    def __init__(self, spot_setup, dbname=None, dbformat=None, parallel='seq',save_sim=True):

        _algorithm.__init__(self,spot_setup, dbname=dbname, dbformat=dbformat, parallel=parallel,save_sim=save_sim)


    def find_min_max(self):
        randompar=self.parameter()['random']        
        for i in range(1000):
            randompar=np.column_stack((randompar,self.parameter()['random']))
        return np.amin(randompar,axis=1),np.amax(randompar,axis=1)
    
    def check_par_validity(self,par):
        if len(par) == len(self.min_bound) and len(par) == len(self.max_bound):
            for i in range(len(par)):
                if par[i]<self.min_bound[i]: 
                    #par[i]=self.min_bound[i]#-self.min_bound[i]+par[i] #TODO: Build in reflection
                    par[i]=self.min_bound[i]+self.min_bound[i]-par[i] #TODO: Build in reflection
                if par[i]>self.max_bound[i]:
                    #par[i]=self.max_bound[i]#+self.max_bound[i]-par[i]
                    par[i]=self.max_bound[i]+self.max_bound[i]-par[i]
            return par
        else:
            print('ERROR: Bounds have not the same lenghts as Parameterarray')
            return par    

    def get_other_random_chains(self,cur_chain):
        valid=False        
        while valid == False:         
            random_chain1 = np.random.randint(0,self.nChains)
            random_chain2 = np.random.randint(0,self.nChains)
            if random_chain1!=cur_chain and random_chain2!=cur_chain and random_chain1!=random_chain2:
                valid=True
        return random_chain1, random_chain2

    def get_new_proposal_vector(self,cur_chain,cur_parameter,newN,nrN):
        gamma = self._get_gamma(nrN)
        random_chain1,random_chain2 = self.get_other_random_chains(cur_chain)
        new_parameterset=[]        
        position = self.chain_samples-1#self.nChains*self.chain_samples+self.chain_samples+cur_chain-1
        for i in range(self.N):#Go through parameters
            
            if newN[i] == True:
                new_parameterset.append(cur_parameter[cur_chain][position][i] + gamma*np.array(cur_parameter[random_chain1][position][i]-cur_parameter[random_chain2][position][i]) + np.random.uniform(0,self.eps))
            else:
                new_parameterset.append(cur_parameter[cur_chain][position][i])
                
        new_parameter=self.check_par_validity(new_parameterset)        
        return new_parameter
    
    def get_r_hat(self, cur_parameter): # TODO: Use only the last 50% of each chain (vrugt 2009)
        # Calculates the \hat{R}-convergence diagnostic
        # ----------------------------------------------------
        # For more information please refer to: 
        # Gelman, A. and D.R. Rubin, (1992) Inference from Iterative Simulation 
        #      Using Multiple chain, Statistical Science, Volume 7, Issue 4, 
        #      457-472.
        # Brooks, S.P. and A. Gelman, (1998) General Methods for Monitoring 
        #      Convergence of Iterative Simulations, Journal of Computational and 
        #      Graphical Statistics. Volume 7, 434-455. Note that this function 
        #      returns square-root definiton of R (see Gelman et al., (2003), 
        #      Bayesian Data Analsyis, p. 297).

        # Written by Jasper A. Vrugt
        # Los Alamos, August 2007
        # Translated into Python by Tobias Houska
        

        n = self.chain_samples
        m = self.nChains
        N = 1.0#number of parameters #TODO: Adjust for more than 1 parameter
        #x = cur_parameter
        if N ==1.0:
            # ----------- DREAM Manual -----
            N = m #chains
            T = n #chain samples
            T2 = int(T/2.0) # Analyses just the second half of the chains
            sums2=[]
            cmeans=[]
            for i in range(N):
                c_mean = (2.0/(T-2.0))*np.sum(cur_parameter[i][T2:])
                cmeans.append(c_mean)                
                sums1=[]                
                for j in range(T2):
                    sums1.append((cur_parameter[i][T2+j]-c_mean)**2.0)
                sums2.append(np.sum(sums1))
            W  = 2.0/(N*(T-2.0))*np.sum(sums2)
            sums =[]
            v_mean = 1.0/N * np.sum(cmeans)
            for i in range(N):
                sums.append((cmeans[i]-v_mean)**2.0)
                
            B  = (1.0 / 2.0*(N-1.0))*np.sum(sums)*T
            s2 = ((T-2.0)/T) * W + 2.0/T*B
            R = np.sqrt(((N+1.0)/N)*s2/W-(T-2.0)/(N*T))
            return R

    def update_status(self,parameter,statistic,simulations,rep):
        self.cur_parameter.append(parameter)
        self.abc_statistic[rep].append(statistic)
        self.datawriter.save(statistic,parameter,simulations=simulations,chains=rep)
    
    def _get_gamma(self,N):
        #N = Number of parameters
        p = np.random.uniform(low=0,high=1)
        if p >=0.2:
            gamma = 2.38/np.sqrt(2*int(N))
        else:
            gamma = 1
        return gamma
        
    def sample(self, repetitions, nChains = 10, convergenceCriteria = 0.9, eps = 10e-6, nCr=3):
        
        self.rep           = repetitions
        self.nChains       = nChains
        self.eps           = eps     
        self.r_hats        = []
        self.N             = len(self.parameter()['random'])
        self.objs          = 1
        self.cur_parameter = np.zeros((self.nChains,self.rep,self.N))
        #self.cur_parameter = [[],[],[],[],[]]
        self.abc_statistic = np.zeros((self.nChains,self.rep)) # TODO more objs
        #self.abc_statistic = [[],[],[],[],[]]
        self.simulations   = [[],[],[],[],[],[],[],[],[],[]] #TODO: Unfortunatelly [[]]*self.nChains does not doe the trick. Needs to be manual adapted at the moment to the number of chains
        self.min_bound, self.max_bound = self.find_min_max()
        self.iter          = 0
        self.chain_samples = 0
        print('Initiat chains')
        for rep in range(self.nChains):
            par = self.parameter()['random']
            self.cur_parameter[rep][self.chain_samples] = par
            sim = self.model(par)
            like = self.objectivefunction(sim,self.evaluation)
            self.abc_statistic[rep][self.chain_samples] = like
            self.simulations[rep] = sim
            self.datawriter.save(like,par,simulations=sim,chains=rep)
            self.iter+=1
        self.chain_samples += 1            

        accepted  = 0.0
        starttime=time.time()
        intervaltime=starttime
        self.min_bound, self.max_bound = self.find_min_max()

        print('Beginn Random Walk')
        self.CR = []
        for i in range(nCr):
            self.CR.append((i+1)/nCr)
                        
        self.convergence=False
        while self.iter<repetitions-nChains:
            pCr = np.random.randint(0,nCr)            
            ids=[]         
            for i in range(self.N):
                ids.append(np.random.uniform(low=0,high=1))
            newN = []
            nrN  = 0
            for i in range(len(ids)):
                if ids[i] < self.CR[pCr]:
                    newN.append(True)
                    nrN+=1
                else:
                    newN.append(False)
            if nrN == 0:
                ids=[np.random.randint(0,self.N)]
                nrN=1
            param_generator = ((rep,self.get_new_proposal_vector(rep,self.cur_parameter,newN,nrN)) for rep in xrange(int(self.nChains)))                
            for rep,vector,simulations in self.repeat(param_generator):

                position = self.chain_samples     
                new_like = self.objectivefunction(simulations,self.evaluation)

                # Accept new candidate if better as last one
                if (new_like >= self.abc_statistic[rep][position-1]):
                    #if rep == 0:
                        #print('Accepted',vector,new_like,self.abc_statistic[rep][-1])
                    self.datawriter.save(new_like,vector,simulations=simulations,chains=rep)
                    accepted += 1.0  # monitor acceptance
                    self.cur_parameter[rep][position] = vector
                    self.abc_statistic[rep][position] = new_like
                    self.simulations[rep] = sim
                    self.status(self.iter+nChains,new_like,vector)                      
                
                else:
                    logMetropHastRatio = np.exp(new_like-self.abc_statistic[rep][position-1])
                    print('Metropolis',new_like,self.abc_statistic[rep][-1],logMetropHastRatio)
                    u = np.random.uniform(low=0,high=1)
                    #Accept proposal with Metropolis decision
                    if logMetropHastRatio > u: 
                        self.datawriter.save(new_like,vector,simulations=simulations,chains=rep)               
                        self.status(self.iter,new_like,vector)                      
                        #accepted = accepted + 1.0  # monitor acceptance
                        self.cur_parameter[rep][position] = vector
                        self.abc_statistic[rep][position] = new_like
                        self.simulations[rep] = sim
                    #Reject proposal and save last states
                    else:
                        self.cur_parameter[rep][position] = self.cur_parameter[rep][position-1]
                        self.abc_statistic[rep][position] = self.abc_statistic[rep][position-1]
                        self.datawriter.save(self.abc_statistic[rep][position-1],self.cur_parameter[rep][position-1],simulations=self.simulations[rep],chains=rep)   
                self.iter+=1
            if self.iter > 10*nChains:
                r_hat=[]
                for i in range(self.N):
                    r_hat.append(self.get_r_hat(self.cur_parameter[:,:,i]))
                #print r_hat
                self.r_hats.append(r_hat)

            #Progress bar
            self.chain_samples+=1
            acttime=time.time()
            #Refresh progressbar every second
            if acttime-intervaltime>=2:
                text='%i of %i (best like=%g)' % (self.iter,repetitions,self.status.objectivefunction)
                print(text)
                #print('rhat='+str(r_hat))
                intervaltime=time.time()
        
        try:
            self.datawriter.finalize()
        except AttributeError: #Happens if no database was assigned
            pass
        print('End of sampling')
        text="Acceptance rate = "+str(accepted/repetitions)        
        print(text)
        text='%i of %i (best like=%g)' % (self.status.rep,repetitions,self.status.objectivefunction)
        print(text)
        print('Best parameter set:')
        print(self.status.params)
        text='Duration:'+str(round((acttime-starttime),2))+' s'
        print(text)
        return self.r_hats