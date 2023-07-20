# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:08:55 2019

@author: gameg
"""
import numpy as np
import pandas as pd

import random

"""
Optimized for speed, but not for memory
issues with memory can be solved by using a generator
global variables are mostly unnecessary
will solve when reimplimenting
"""


random.seed(616)
#os.chdir('C:/Users/Frank/Desktop/Record linkage')
#os.chdir('C:/Users/gameg/Desktop/Record linkage')
#Unfortunatly I could not find a native way in numpy to do this for characteres, thus these functions
def CompareArray(ComparisonArrayA,ComparisonArrayB):
    returnArray=np.zeros_like(ComparisonArrayA, dtype=bool)
    
    for j in range(0,ComparisonArrayA[0].size):
        
        for i in range(0,ComparisonArrayB.size):
            if(ComparisonArrayA[i,j]==ComparisonArrayB[i]):
                returnArray[i,j]=True
            else:
                returnArray[i,j]=False
    return(returnArray)

#Dataset_A=genfromtxt('D:/Power prior starting point/Dataset_A.csv', delimiter=",",skip_header=1,dtype='float64')
#Dataset_B=genfromtxt('D:/Power prior starting point/Dataset_B.csv', delimiter=",",skip_header=1,dtype='float64')  

#Full_Dataset=pd.read_csv('E:/2015.csv')
Full_Dataset=pd.read_csv('Full dataset to read!!!')
Full_Dataset_head=Full_Dataset.iloc[0:10,:]
i=1
np.random.seed(i)
amount=500
testing=np.random.choice(range(0,Full_Dataset.shape[0]),amount+50,replace=False)
np.random.seed(i+1)
testingA=np.random.choice(testing,amount,replace=False)
Full_Dataset_A=Full_Dataset.iloc[testingA,:]
Full_Dataset_B=Full_Dataset.iloc[testing,:]
#PercentError=.4
#For now we'll assign error randomly

np.random.seed(616)
probs=np.random.choice(range(5,60),np.unique(Full_Dataset_A.FMONTH).size,replace=False)/100
testing_origin=testing.copy()
testing_A_origin=testingA.copy()
amount_origin=amount

#We need to textmine N_month and N_Day outside the loop:
# Interview Month is 1 through 12
    # Need to textmine
Full_Dataset_A.IMONTH
words = list(Full_Dataset_A.IMONTH)
nums = [word[2:4] for word in words]
length_nums = len(nums)
for i in range(0,length_nums):
    nums[i]=int(nums[i])
#pd.unique(nums)
Full_Dataset_A.insert(0,'N_MONTH',nums)
Full_Dataset_A.N_MONTH
    
words = list(Full_Dataset_B.IMONTH)
nums = [word[2:4] for word in words]
length_nums = len(nums)
for i in range(0,length_nums):
    nums[i]=int(nums[i])
#pd.unique(nums)
Full_Dataset_B.insert(0,'N_MONTH',nums)
Full_Dataset_B.N_MONTH

# Interview Day is 1 through 31
# Need to textmine
Full_Dataset_A.IDAY
words = list(Full_Dataset_A.IDAY)
nums = [word[2:4] for word in words]
length_nums = len(nums)
for i in range(0,length_nums):
    nums[i]=int(nums[i])
#pd.unique(nums)
Full_Dataset_A.insert(0,'N_DAY',nums)
Full_Dataset_A.N_DAY
words = list(Full_Dataset_B.IDAY)
nums = [word[2:4] for word in words]
length_nums = len(nums)
for i in range(0,length_nums):
    nums[i]=int(nums[i])
#pd.unique(nums)
Full_Dataset_B.insert(0,'N_DAY',nums)
Full_Dataset_B.N_DAY


for a in range(0,np.unique(Full_Dataset_A.FMONTH).size):
    PercentError=probs[a]
    testingA=testing_A_origin[Full_Dataset_A.FMONTH==np.unique(Full_Dataset_A.FMONTH)[a]]
    testing=testing_origin[Full_Dataset_B.FMONTH==np.unique(Full_Dataset_B.FMONTH)[a]]
    amount=testingA.size
    # 'State' but has 72 territories
    Full_Dataset_A._STATE
    np.random.seed(i+2)
    error=np.random.choice(testingA,int(amount*PercentError),replace=False)
    np.random.seed(i+3)
    Full_Dataset_A.loc[np.isin(Full_Dataset_A._STATE.index,error),'_State']=np.random.choice(np.unique(Full_Dataset_B._STATE),int(amount*PercentError),replace=True).copy()
    np.random.seed(i+4)
    error=np.random.choice(testing,int(amount*PercentError),replace=False)
    np.random.seed(i+5)
    Full_Dataset_B.loc[np.isin(Full_Dataset_B._STATE.index,error),'_State']=np.random.choice(np.unique(Full_Dataset_B._STATE),int(amount*PercentError),replace=True).copy()
    
    # File Month is 1 through 12
    Full_Dataset_A.FMONTH
    np.random.seed(i+6)
    error=np.random.choice(testingA,int(amount*PercentError),replace=False)
    np.random.seed(i+7)
    Full_Dataset_A.loc[np.isin(Full_Dataset_A._STATE.index,error),'FMONTH']=np.random.choice(np.unique(Full_Dataset_B.FMONTH),int(amount*PercentError),replace=True).copy()
    np.random.seed(i+8)
    error=np.random.choice(testing,int(amount*PercentError),replace=False)
    np.random.seed(i+9)
    Full_Dataset_B.loc[np.isin(Full_Dataset_B._STATE.index,error),'FMONTH']=np.random.choice(np.unique(Full_Dataset_B.FMONTH),int(amount*PercentError),replace=True).copy()
    
    
    
    np.random.seed(i+10)
    error=np.random.choice(testingA,int(amount*PercentError),replace=False)
    np.random.seed(i+11)
    Full_Dataset_A.loc[np.isin(Full_Dataset_A.index,error),'N_MONTH']=np.random.choice(np.unique(Full_Dataset_B.N_MONTH),int(amount*PercentError),replace=True).copy()
    
    #Check
    #Full_Dataset_A.N_DAY[Full_Dataset_A.index==7594]
    #Full_Dataset_A.N_DAY[Full_Dataset_A.index==367980]
    
    
    np.random.seed(i+12)
    error=np.random.choice(testing,int(amount*PercentError),replace=False)
    np.random.seed(i+13)
    Full_Dataset_B.loc[np.isin(Full_Dataset_B.index,error),'N_MONTH']=np.random.choice(np.unique(Full_Dataset_B.N_MONTH),int(amount*PercentError),replace=True).copy()
    
    
    error=np.random.choice(testingA,int(amount*PercentError),replace=False)
    np.random.seed(i+14)
    Full_Dataset_A.loc[np.isin(Full_Dataset_A.index,error),'N_DAY']=np.random.choice(np.unique(Full_Dataset_B.N_DAY),int(amount*PercentError),replace=True).copy()
    
    
    error=np.random.choice(testing,int(amount*PercentError),replace=False)
    np.random.seed(i+15)
    Full_Dataset_B.loc[np.isin(Full_Dataset_B.index,error),'N_DAY']=np.random.choice(np.unique(Full_Dataset_B.N_DAY),int(amount*PercentError),replace=True).copy()
    
    #From here we chose variables that are equally matched between their two notes
    
    # Cellular Telephone
    Full_Dataset.CELLFON3
    np.random.seed(i+16)
    error=np.random.choice(testingA,int(amount*PercentError),replace=False)
    np.random.seed(i+17)
    Full_Dataset_A.loc[np.isin(Full_Dataset_A.index,error),'CELLFON3']=np.random.choice(np.unique(Full_Dataset_B.CELLFON3),int(amount*PercentError),replace=True).copy()
    np.random.seed(i+18)
    error=np.random.choice(testing,int(amount*PercentError),replace=False)
    np.random.seed(i+19)
    Full_Dataset_B.loc[np.isin(Full_Dataset_B.index,error),'CELLFON3']=np.random.choice(np.unique(Full_Dataset_B.CELLFON3),int(amount*PercentError),replace=True).copy()
    
    #Sex
    Full_Dataset.SEX
    error=np.random.choice(testingA,int(amount*PercentError),replace=False)
    np.random.seed(i+20)
    Full_Dataset_A.loc[np.isin(Full_Dataset_A.index,error),'SEX']=np.random.choice(np.unique(Full_Dataset_B.SEX),int(amount*PercentError),replace=True).copy()
    np.random.seed(i+21)
    error=np.random.choice(testing,int(amount*PercentError),replace=False)
    np.random.seed(i+22)
    Full_Dataset_B.loc[np.isin(Full_Dataset_B.index,error),'SEX']=np.random.choice(np.unique(Full_Dataset_B.SEX),int(amount*PercentError),replace=True).copy()
    
    #MARITAL
    Full_Dataset.MARITAL
    error=np.random.choice(testingA,int(amount*PercentError),replace=False)
    np.random.seed(i+23)
    Full_Dataset_A.loc[np.isin(Full_Dataset_A.index,error),'MARITAL']=np.random.choice(np.unique(Full_Dataset_B.MARITAL),int(amount*PercentError),replace=True).copy()
    np.random.seed(i+24)
    error=np.random.choice(testing,int(amount*PercentError),replace=False)
    np.random.seed(i+25)
    Full_Dataset_B.loc[np.isin(Full_Dataset_B.index,error),'MARITAL']=np.random.choice(np.unique(Full_Dataset_B.MARITAL),int(amount*PercentError),replace=True).copy()
    
    # _RACEG21 Choosen race varaible
    Full_Dataset._RACEG21
    error=np.random.choice(testingA,int(amount*PercentError),replace=False)
    np.random.seed(i+26)
    Full_Dataset_A.loc[np.isin(Full_Dataset_A.index,error),'_RACEG21']=np.random.choice(np.unique(Full_Dataset_B._RACEG21),int(amount*PercentError),replace=True).copy()
    np.random.seed(i+27)
    error=np.random.choice(testing,int(amount*PercentError),replace=False)
    np.random.seed(i+27)
    Full_Dataset_B.loc[np.isin(Full_Dataset_B.index,error),'_RACEG21']=np.random.choice(np.unique(Full_Dataset_B._RACEG21),int(amount*PercentError),replace=True).copy()
    
    # _PAINDX1 Physical Activity index
    Full_Dataset._PAINDX1
    error=np.random.choice(testingA,int(amount*PercentError),replace=False)
    np.random.seed(i+28)
    Full_Dataset_A.loc[np.isin(Full_Dataset_A.index,error),'_PAINDX1']=np.random.choice(np.unique(Full_Dataset_B._PAINDX1),int(amount*PercentError),replace=True).copy()
    np.random.seed(i+29)
    error=np.random.choice(testing,int(amount*PercentError),replace=False)
    np.random.seed(i+30)
    Full_Dataset_B.loc[np.isin(Full_Dataset_B.index,error),'_PAINDX1']=np.random.choice(np.unique(Full_Dataset_B._PAINDX1),int(amount*PercentError),replace=True).copy()

testing=testing_origin.copy()
testingA=testing_A_origin.copy()
amount=amount_origin
  
Comparison_A=pd.concat([Full_Dataset_A]*Full_Dataset_B.shape[0])
#View_A=Comparison_A.iloc[0:1000,]
#Works but is too slow:
#Comparison_B=pd.concat([Full_Dataset_B.iloc[0,:]]*Full_Dataset_A.shape[0])
#for i in range(0,Full_Dataset_B.shape[0]):
#    print(i)
#    Comparison_B=Comparison_B.append(pd.concat([Full_Dataset_B.iloc[i,:]]*Full_Dataset_A.shape[0]))

Comparison_B=Full_Dataset_B.loc[Full_Dataset_B.index.repeat(Full_Dataset_A.shape[0])]

#First attempt:
State_gamma=np.apply_along_axis(np.equal,0,Comparison_B._STATE,Comparison_A._STATE)
FMonth_gamma=np.apply_along_axis(np.equal,0,Comparison_B.FMONTH,Comparison_A.FMONTH)
NMonth_gamma=np.apply_along_axis(np.equal,0,Comparison_B.N_MONTH,Comparison_A.N_MONTH)
NDay_gamma=np.apply_along_axis(np.equal,0,Comparison_B.N_DAY,Comparison_A.N_DAY)
#Then Add
Phone_gamma=np.apply_along_axis(np.equal,0,Comparison_B.CELLFON3,Comparison_A.CELLFON3)
Sex_gamma=np.apply_along_axis(np.equal,0,Comparison_B.SEX,Comparison_A.SEX)
Marital_gamma=np.apply_along_axis(np.equal,0,Comparison_B.MARITAL,Comparison_A.MARITAL)
Race_gamma=np.apply_along_axis(np.equal,0,Comparison_B._RACEG21,Comparison_A._RACEG21)
Pain_gamma=np.apply_along_axis(np.equal,0,Comparison_B._PAINDX1,Comparison_A._PAINDX1)


############################################################
###Create Gamma matrices from element-wise comparisons
#Interview Date

#Make each a singular comparison (maybe)

#Note:these will throw errors if not converted into float 32s

#Not sure which order the numbers should be in

#NMonth
Gamma2=NMonth_gamma.astype(np.float32)
Gamma1=-(NMonth_gamma.astype(np.float32))+1
# gamme 1 = opposite of gamma 2 (we think for summing the number of non-links ie. counting elts of gamma 2 = counting 0s in gamma 1 )

#NDay
Gamma4=NDay_gamma.astype(np.float32)
Gamma3=-(NDay_gamma.astype(np.float32))+1

#FMonth
Gamma6=FMonth_gamma.astype(np.float32)
Gamma5=-(FMonth_gamma.astype(np.float32))+1

#State
Gamma8=State_gamma.astype(np.float32)
Gamma7=-(State_gamma.astype(np.float32))+1

#Phone
Gamma10=Phone_gamma.astype(np.float32)
Gamma9=-(Phone_gamma.astype(np.float32))+1

A_Index=Comparison_A.index.astype(np.float32)
B_Index=Comparison_B.index.astype(np.float32)

#pd.DataFrame(Data=[[np.array(Comparison_A.index)],[np.array(Comparison_B.index)],[Gamma5],[Gamma4]])

#Specify number of iterations (including burn-in) and initialize parameter matrices
# to-do: test number of iterations
nsim=1000

#Initialize the linking matrix C
C=np.zeros((Gamma1.shape[0]),dtype=float)

#Start with 2 links known 
Known=np.zeros((Gamma1.shape[0]),dtype=float)

SuperGamma=pd.DataFrame(np.c_[A_Index,B_Index,Gamma1,Gamma2,Gamma3,Gamma4,Gamma5,Gamma6,Gamma7,Gamma8,\
                              Gamma9,Gamma10,C,Known],\
            columns=['A_ID','B_ID','Gamma1','Gamma2','Gamma3','Gamma4','Gamma5','Gamma6','Gamma7','Gamma8',\
                     'Gamma9','Gamma10','C','Known'])

#SuperGamma.Known[SuperGamma.A_ID==431151]=1
#SuperGamma.Known[SuperGamma.B_ID==431151]=1
#SuperGamma.C[np.logical_and(SuperGamma.B_ID==431151,SuperGamma.A_ID==431151)]=1

#SuperGamma.Known[SuperGamma.A_ID==344333]=1
#SuperGamma.Known[SuperGamma.B_ID==344333]=1
#SuperGamma.C[np.logical_and(SuperGamma.B_ID==344333,SuperGamma.A_ID==344333)]=1
amount_known=int(amount*(.3))
np.random.seed(616) #Set seed for reproducibility(was it necessarry?)


fix=np.random.choice(np.unique(SuperGamma.A_ID),amount_known,replace=False)

LinkDesignation=pd.DataFrame(np.zeros((Full_Dataset_A.shape[0], nsim),dtype=float))
Unique_AID=np.unique(A_Index)
Unique_BID=np.unique(B_Index)
LinkDesignation.index=Unique_AID


# loops through gamma matrix; for every pairing selected to be "fixed", sets known and C val to 1
for i in range(0,amount_known):
    SuperGamma.Known[SuperGamma.A_ID==fix[i]]=1
    SuperGamma.Known[SuperGamma.B_ID==fix[i]]=1
    SuperGamma.C[np.logical_and(SuperGamma.B_ID==fix[i],SuperGamma.A_ID==fix[i])]=1
    LinkDesignation.iloc[LinkDesignation.index==fix[i],:]=np.repeat(fix[i],nsim)
    



#Specify values for hyperparameters of prior distributions (these are dirichlet prior parameters)
prior_pi=np.array([1,1])
prior_IMonth_M=np.array([1,2])
prior_IMonth_U=np.array([1,1])
prior_IDay_M=np.array([1,2])
prior_IDay_U=np.array([1,1])
prior_FMonth_M=np.array([1,2])
prior_FMonth_U=np.array([1,1])
prior_State_M=np.array([1,2])
prior_State_U=np.array([1,1])
prior_Phone_M=np.array([1,2])
prior_Phone_U=np.array([1,1])


#array of size k * nsim. To store all samples of theta m and theta u from every iteration 
theta_M_IMonth=np.zeros((prior_IMonth_M.size,nsim),dtype=float)
theta_U_IMonth=np.zeros((prior_IMonth_U.size,nsim),dtype=float)

theta_M_IDay=np.zeros((prior_IDay_M.size,nsim),dtype=float)
theta_U_IDay=np.zeros((prior_IDay_U.size,nsim),dtype=float)

theta_M_FMonth=np.zeros((prior_FMonth_M.size,nsim),dtype=float)
theta_U_FMonth=np.zeros((prior_FMonth_M.size,nsim),dtype=float)

theta_M_State=np.zeros((prior_State_M.size,nsim),dtype=float)
theta_U_State=np.zeros((prior_State_M.size,nsim),dtype=float)

theta_M_Phone=np.zeros((prior_Phone_M.size,nsim),dtype=float)
theta_U_Phone=np.zeros((prior_Phone_M.size,nsim),dtype=float)

# proportion of matches (not links) - has uniform prior 
pi_M=np.zeros((prior_pi.size,nsim),dtype=float) 


# probability of a link for any given record pair (a,)
#LinkDesignation=pd.DataFrame(0, index=np.arange(Full_Dataset_A.shape[0]))
LinkProbability=np.zeros((Gamma1.shape[0], nsim),dtype=float)



#Leaving this out for now
#ptm <- proc.time()
#Prior will be that the 50 true links are true
alpha=.8
#alpha=0

#n_O=np.array(range(0,Gamma1.shape[0]))[::1000][0:50]
n_O=np.array(range(0,Gamma1.shape[0]))[::1000][0]

Gamma1_prior=Gamma1[:] 
Gamma2_prior=Gamma2[:]
Gamma3_prior=Gamma3[:]
Gamma4_prior=Gamma4[:]
Gamma5_prior=Gamma5[:]
Gamma6_prior=Gamma6[:]
"""
There are still unnnecessarry memory allocations here
I didn't touch them because program crashes when I do
I noted them and inspect again while reimplementation
"""

#Position of non-liknks, assume we know the following are non links
nonlinks=np.array(range(1,Gamma1.shape[0]))[::1000]

#Skipping for now but needs to be dramatically changed
skipFix=1
if(skipFix==0):
    for w in range(0,n_O):
        for m in range (0,n_O):
            if(w==m):
                to_assign = 1
                C[w,m]=1
                LinkDesignation[m,:]=w
            else:
                to_assign = 0
                C[w,m]=0
            Gamma1_prior[w,m]=to_assign
            Gamma2_prior[w,m]=to_assign
            Gamma3_prior[w,m]=to_assign
            Gamma4_prior[w,m]=to_assign
            Gamma5_prior[w,m]=to_assign
            Gamma6_prior[w,m]=to_assign    

if(skipFix==0):
    for w in range(nonlinks-1,Gamma1_prior.shape[0]):
        for m in range (nonlinks-1,Gamma1_prior.shape[1]):
            if(w==m):
                Gamma1_prior[w,m]=0
                Gamma2_prior[w,m]=0
                Gamma3_prior[w,m]=0
                Gamma4_prior[w,m]=0
                Gamma5_prior[w,m]=0
                Gamma6_prior[w,m]=0
                
            else:
                Gamma1_prior[w,m]=0
                Gamma2_prior[w,m]=0
                Gamma3_prior[w,m]=0
                Gamma4_prior[w,m]=0
                Gamma5_prior[w,m]=0
                Gamma6_prior[w,m]=0  
"""
This if-else block is highly unnecessary
it seems to be not pricy but it is because it uses
global variables which are pricy to use in terms of memory position 
"""                
            
#for w in range(0,n_O):
#        for m in range (0,n_O):
#            if(w==m):
#                C[w,m]=1
#                LinkDesignation[m,:]=w
#            else:
#                C[w,m]=0

Full_IndexB=np.unique(Comparison_B.index)
Full_IndexA=np.unique(Comparison_A.index)
Fullrange=np.array(range(0,Gamma1.shape[0]))

PriorLinks_B=Comparison_B.index[n_O]
PriorLinks_A=Comparison_B.index[n_O]
Prior_index=n_O

UnknownLinks_B=Full_IndexB[np.isin(Full_IndexB,PriorLinks_B,invert=True)]
UnknownLinks_A=Full_IndexA[np.isin(Full_IndexA,PriorLinks_A,invert=True)]
Unknown_index=Fullrange[np.isin(Fullrange,n_O,invert=True)]

#np.unique(Comparison_B.index).size
#UnknownLinks.size
#PriorLinks.size
#priorlinks_cols=list(range(0,n_O-1))+list(range(nonlinks-1,Gamma1_prior.shape[1]-1))

theta_M_IMonth_prior=np.zeros((prior_IMonth_M.size,nsim),dtype=float)
theta_U_IMonth_prior=np.zeros((prior_IMonth_U.size,nsim),dtype=float)

theta_M_IDay_prior=np.zeros((prior_IDay_M.size,nsim),dtype=float)
theta_U_IDay_prior=np.zeros((prior_IDay_U.size,nsim),dtype=float)

theta_M_FMonth_prior=np.zeros((prior_FMonth_M.size,nsim),dtype=float)
theta_U_FMonth_prior=np.zeros((prior_FMonth_U.size,nsim),dtype=float)

theta_M_State_prior=np.zeros((prior_State_M.size,nsim),dtype=float)
theta_U_State_prior=np.zeros((prior_State_U.size,nsim),dtype=float)

theta_M_Phone_prior=np.zeros((prior_Phone_M.size,nsim),dtype=float)
theta_U_Phone_prior=np.zeros((prior_Phone_U.size,nsim),dtype=float)

Prior_TrueLinks=np.zeros(nsim)
Prior_FalseLinks=np.zeros(nsim)
Prior_AllLinks=np.zeros(nsim)
Alphas=np.zeros(nsim)
#Likelihood_new_con=np.zeros(C.shape[1])
#Likelihood_old_con=np.zeros(C.shape[1])
#Likelihood_new_con=np.zeros(nsim)
#Likelihood_old_con=np.zeros(nsim)



for t in range(0,nsim):
    #LinkDesignation[0:n_O-1,t]=range(0,n_O-1)
    #For Testing:
    #t=0
    #if (t>0):#Fix this
        #print(np.sum(LinkDesignation[:,t-1]==np.array(range(0,LinkDesignation.shape[0]))))
        #Prior_TrueLinks[t]=np.sum(LinkDesignation[n_O:nonlinks,t-1]==np.array(range(n_O,nonlinks)))
        #LinkDesignationZeroes=LinkDesignation[(LinkDesignation[n_O:,t-1]>0)]
        #Prior_FalseLinks[t]=np.sum((LinkDesignation[n_O:nonlinks,t-1]!=np.array(range(n_O,nonlinks))))
        #Prior_AllLinks[t]=np.sum(LinkDesignation[n_O:,t-1]==np.array(range(n_O,LinkDesignation.shape[0])))
    
    #Sampling Posterior Distribution of parameters
    #To match R code we use np.sum, which adds all of the values in the array together
    #theta_M_IMonth[:,t]=np.random.dirichlet(np.array([prior_IMonth_M[0]+np.sum((C)*Gamma1),\
    #prior_IMonth_M[1]+np.sum((C)*Gamma2)]))
    
    #theta_U_IMonth[:,t]=np.random.dirichlet(np.array([prior_IMonth_U[0]+np.sum((1-C)*Gamma1),prior_IMonth_U[1]\
    #              +np.sum((1-C)*Gamma2)]))   
    theta_M_IMonth[:,t]=np.random.dirichlet(np.array([prior_IMonth_M[0]+np.sum((SuperGamma.C)*Gamma1),\
    prior_IMonth_M[1]+np.sum((SuperGamma.C)*SuperGamma.Gamma2)]))
    
    # supergamma.c - latent linking structure (0s and 1s)
    # supergamma 
    theta_U_IMonth[:,t]=np.random.dirichlet(np.array([prior_IMonth_U[0]+np.sum((1-SuperGamma.C)*Gamma1),prior_IMonth_U[1]\
                  +np.sum((1-SuperGamma.C)*Gamma2)]))  
    
    theta_M_IDay[:,t]=np.random.dirichlet(np.array([prior_IDay_M[0]+np.sum((SuperGamma.C)*Gamma3),\
    prior_IDay_M[1]+np.sum((SuperGamma.C)*Gamma4)]))
    
    theta_U_IDay[:,t]=np.random.dirichlet(np.array([prior_IDay_U[0]+np.sum((1-SuperGamma.C)*Gamma3),prior_IDay_U[1]\
                  +np.sum((1-SuperGamma.C)*Gamma4)]))   
    
    theta_M_FMonth[:,t]=np.random.dirichlet(np.array([prior_FMonth_M[0]+np.sum((SuperGamma.C)*Gamma5),\
    prior_FMonth_M[1]+np.sum((SuperGamma.C)*Gamma6)]))
    
    theta_U_FMonth[:,t]=np.random.dirichlet(np.array([prior_FMonth_U[0]+np.sum((1-SuperGamma.C)*Gamma5),prior_FMonth_U[1]\
                  +np.sum((1-SuperGamma.C)*Gamma6)]))
    
    theta_M_State[:,t]=np.random.dirichlet(np.array([prior_State_M[0]+np.sum((SuperGamma.C)*Gamma7),\
    prior_State_M[1]+np.sum((SuperGamma.C)*Gamma8)]))
    
    theta_U_State[:,t]=np.random.dirichlet(np.array([prior_State_U[0]+np.sum((1-SuperGamma.C)*Gamma7),prior_State_U[1]\
                  +np.sum((1-SuperGamma.C)*Gamma8)]))
    
    theta_M_Phone[:,t]=np.random.dirichlet(np.array([prior_Phone_M[0]+np.sum((SuperGamma.C)*Gamma9),\
    prior_Phone_M[1]+np.sum((SuperGamma.C)*Gamma10)]))
    
    theta_U_Phone[:,t]=np.random.dirichlet(np.array([prior_Phone_U[0]+np.sum((1-SuperGamma.C)*Gamma9),prior_Phone_U[1]\
                  +np.sum((1-SuperGamma.C)*Gamma10)]))
    
    prior_prior=np.random.beta(.5,.5,1)
    
        
    
    pi_M[1,t]=np.random.beta(prior_pi[0]+np.sum(SuperGamma.C),prior_pi[1]+Gamma1.shape[0]-np.sum(SuperGamma.C))
    
    #ForTesting
    #i=0
    
    #Iterate through the rows of i\
    for i in range(0,Unique_AID.size):#Maybe this should be C.shape[0]-1
        
        #This isn't generalizable to any dataset
        check=SuperGamma.Known[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                           SuperGamma.B_ID==Unique_AID[i])]==1
        if(sum(check)==1):
            continue
            #print("here")
        #print(Unique_AID[i])            
        #Resetting the result for row i
        #C[UnknownLinks_B.index[i]==Full_Dataset_B.index.repeat(Full_Dataset_A.shape[0])]=0
        SuperGamma.C[SuperGamma.A_ID==Unique_AID[i]]=0
        #print(i)
        #start=n_O #Changed to 0
        #Extracting individuals in file B that have links, not including the individual currently linked to i
        #B_linked=np.array(range(0,C.shape[1]))[np.sum(C[0:C.shape[1],0:C.shape[1]],axis=0)==1]
        B_linked=SuperGamma.B_ID[np.logical_and(SuperGamma.Known==0,SuperGamma.C==1)]
        B_linked=np.unique(B_linked)
        
        #B_linked=Comparison_B.index[np.logical_and(C==1,np.isin(Comparison_B.index,Unknown_index))]
        #print(np.isin(B_linked,LinkDesignation[i,t]))
        Test=SuperGamma[np.logical_and(SuperGamma.Known==0,SuperGamma.C==1)]
        
        #B_linked_prior=PriorLinks[np.sum(C[PriorLinks],axis=0)==1]
        B_linked_prior=SuperGamma.B_ID[np.logical_and(SuperGamma.Known==1,SuperGamma.C==1)]
        B_linked_prior=np.unique(B_linked_prior)
        
        #If there are no links, create empty link vector. Then extract non-linked individuals in dataset B
        if B_linked.size==0: #This is really only for the first iteration for t, first few for i
          B_unlinked=np.unique(SuperGamma.B_ID[SuperGamma.Known==0])
          #B_unlinked_prior=np.unique(SuperGamma.B_ID[np.logical_and(SuperGamma.Known==1)])

          B_unlinked_prior=np.setdiff1d(np.unique(SuperGamma.B_ID[SuperGamma.Known==1]), B_linked_prior, assume_unique=True)
        else:
          #B_unlinked=np.unique(SuperGamma.B_ID[np.logical_and(SuperGamma.Known==0,SuperGamma.C==0)])
          B_unlinked=np.setdiff1d(np.unique(SuperGamma.B_ID[SuperGamma.Known==0]), B_linked, assume_unique=True)
          

        #set of Bs that are both unlinked and known
          #B_unlinked_prior=np.unique(SuperGamma.B_ID[np.logical_and(SuperGamma.Known==1,SuperGamma.C==0)])
          B_unlinked_prior=np.setdiff1d(np.unique(SuperGamma.B_ID[SuperGamma.Known==1]), B_linked_prior, assume_unique=True)
          
          #B_unlinked finds the xth element of np.array(range(0,C.shape[1])) and removes it, these are the\
          #links we already have
        
        Gamma1A=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked))].sort_values('B_ID').Gamma1
        Gamma2A=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked))].sort_values('B_ID').Gamma2
        Gamma3A=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked))].sort_values('B_ID').Gamma3
        Gamma4A=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked))].sort_values('B_ID').Gamma4
        Gamma5A=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked))].sort_values('B_ID').Gamma5
        Gamma6A=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked))].sort_values('B_ID').Gamma6
        Gamma7A=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked))].sort_values('B_ID').Gamma7
        Gamma8A=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked))].sort_values('B_ID').Gamma8
        Gamma9A=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked))].sort_values('B_ID').Gamma9
        Gamma10A=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked))].sort_values('B_ID').Gamma10 
          
        #Extract Matrix of DOB, ZIP, and Gender Gamma
        #Gamma2A[185]=1
        
        Gamma_IMonth=np.c_[Gamma1A,Gamma2A]
        Gamma_IDay=np.c_[Gamma3A,Gamma4A]
        Gamma_FMonth=np.c_[Gamma5A,Gamma6A]
        Gamma_State=np.c_[Gamma7A,Gamma8A]
        Gamma_Phone=np.c_[Gamma9A,Gamma10A]
        #print(Gamma_State)
        
        Gamma1A_prior=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked_prior))].sort_values('B_ID').Gamma1
        Gamma2A_prior=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked_prior))].sort_values('B_ID').Gamma2
        Gamma3A_prior=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked_prior))].sort_values('B_ID').Gamma3
        Gamma4A_prior=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked_prior))].sort_values('B_ID').Gamma4
        Gamma5A_prior=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked_prior))].sort_values('B_ID').Gamma5
        Gamma6A_prior=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked_prior))].sort_values('B_ID').Gamma6
        Gamma7A_prior=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked_prior))].sort_values('B_ID').Gamma7
        Gamma8A_prior=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked_prior))].sort_values('B_ID').Gamma8
        Gamma9A_prior=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked_prior))].sort_values('B_ID').Gamma9
        Gamma10A_prior=SuperGamma[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                                 np.isin(SuperGamma.B_ID,B_unlinked_prior))].sort_values('B_ID').Gamma10
        
        #Gamma_IDate_prior=np.c_[Gamma1A_prior,Gamma2A_prior\
        #                      ,Gamma3A_prior]
        #Gamma_FMonth_prior=np.c_[Gamma4A_prior,Gamma5A_prior]
        #Gamma_State_prior=np.c_[Gamma6A_prior,Gamma7A_prior]
        
        Gamma_IMonth_prior=np.c_[Gamma1A_prior,Gamma2A_prior]
        Gamma_IDay_prior=np.c_[Gamma3A_prior,Gamma4A_prior]
        Gamma_FMonth_prior=np.c_[Gamma5A_prior,Gamma6A_prior]
        Gamma_State_prior=np.c_[Gamma7A_prior,Gamma8A_prior]
        Gamma_Phone_prior=np.c_[Gamma9A_prior,Gamma10A_prior]
        #Calculate ratio of likelihoods
        
        
        #We need to define these for our apply prod substitute code
        theta_M_IMonthbyGamma=theta_M_IMonth[:,t]**Gamma_IMonth
        theta_M_IMonthbyGamma_prior=theta_M_IMonth[:,t]**Gamma_IMonth_prior
        
        theta_M_IDaybyGamma=theta_M_IDay[:,t]**Gamma_IDay
        theta_M_IDaybyGamma_prior=theta_M_IDay[:,t]**Gamma_IDay_prior
        
        theta_M_FMonthbyGamma=theta_M_FMonth[:,t]**Gamma_FMonth
        theta_M_FMonthbyGamma_prior=theta_M_FMonth[:,t]**Gamma_FMonth_prior
        
        theta_M_StatebyGamma=theta_M_State[:,t]**Gamma_State
        theta_M_StatebyGamma_prior=theta_M_State[:,t]**Gamma_State_prior
        
        theta_M_PhonebyGamma=theta_M_Phone[:,t]**Gamma_Phone
        theta_M_PhonebyGamma_prior=theta_M_Phone[:,t]**Gamma_Phone_prior
        
        #axis=1 is by row
        num=theta_M_IDaybyGamma.prod(axis=1)*theta_M_IMonthbyGamma.prod(axis=1)*\
        theta_M_FMonthbyGamma.prod(axis=1)*theta_M_StatebyGamma.prod(axis=1)*theta_M_PhonebyGamma.prod(axis=1)
        
        num_prior=theta_M_IDaybyGamma_prior.prod(axis=1)*theta_M_FMonthbyGamma_prior.prod(axis=1)*\
        theta_M_StatebyGamma_prior.prod(axis=1)*theta_M_IMonthbyGamma_prior.prod(axis=1)*\
        theta_M_PhonebyGamma_prior.prod(axis=1)
        #We need to define these for our apply prod substitute code
        theta_U_IDaybyGamma=theta_U_IDay[:,t]**Gamma_IDay
        theta_U_IDaybyGamma_prior=theta_U_IDay[:,t]**Gamma_IDay_prior
        
        theta_U_IMonthbyGamma=theta_U_IMonth[:,t]**Gamma_IMonth
        theta_U_IMonthbyGamma_prior=theta_U_IMonth[:,t]**Gamma_IMonth_prior
        
        theta_U_FMonthbyGamma=theta_U_FMonth[:,t]**Gamma_FMonth
        theta_U_FMonthbyGamma_prior=theta_U_FMonth[:,t]**Gamma_FMonth_prior
        
        theta_U_StatebyGamma=theta_U_State[:,t]**Gamma_State
        theta_U_StatebyGamma_prior=theta_U_State[:,t]**Gamma_State_prior
        
        theta_U_PhonebyGamma=theta_U_Phone[:,t]**Gamma_Phone
        theta_U_PhonebyGamma_prior=theta_U_Phone[:,t]**Gamma_Phone_prior
        
        
        den=theta_U_IDaybyGamma.prod(axis=1)*theta_U_FMonthbyGamma.prod(axis=1)*\
        theta_U_StatebyGamma.prod(axis=1)*theta_U_IMonthbyGamma.prod(axis=1)*theta_U_PhonebyGamma.prod(axis=1)
        
        den_prior=theta_U_IDaybyGamma_prior.prod(axis=1)*theta_U_FMonthbyGamma_prior.prod(axis=1)*\
        theta_U_StatebyGamma_prior.prod(axis=1)*theta_U_IMonthbyGamma_prior.prod(axis=1)*\
        theta_U_PhonebyGamma_prior.prod(axis=1)
        
        if(np.sum(den)==0.0):
            print("error")
            continue
        
   
        Likelihood_prior=(num_prior/den_prior)
            #Likelihood_prior=sum(Likelihood_prior/sum(Likelihood_prior))*scipy.stats.beta.cdf(alpha, 5, 1)
        Likelihood_prior=sum(Likelihood_prior)**alpha#*scipy.stats.beta.cdf(alpha, 5, 1)
       
            
            
        
        Likelihood_base=(num/den)
        
        #if(np.sum(den_prior)>0):
        #Make the below a function for the alpha
        #alpha=AlphaGet(alpha,num_prior,den_prior,Likelihood_base)
        #print(alpha)
        
        #Likelihood_prior_new=(num_prior/den_prior)**alpha_new
        #Likelihood_prior_new=sum(Likelihood_prior_new)
        
        #Likelihood_new=Likelihood_prior_new*Likelihood_base*prior_prior
        
        
        
        #Likelihood_base[B_unlinked_prior]=Likelihood_prior*Likelihood_base[B_unlinked_prior]
        Likelihood=Likelihood_base*Likelihood_prior
        #Likelihood=sum(Likelihood_prior)*Likelihood_base#*prior_prior
        
        #Likelihood_old=Likelihood
        
        #Calculate probability of individual i not linking
        #p_nolink=(Full_Dataset_A.shape[0]-B_linked.size+prior_pi[1]-1)\
        #/(B_linked.size+prior_pi[0])
        
        p_nolink=(amount-B_linked.size)*(amount-B_linked.size+prior_pi[1]-1)\
        /(B_linked.size+prior_pi[0])
    
    #Parsing together possible moves and move probability
    #This adds 3000+i to the end of the array of B_unlinked  
        B_unlinked_true=B_unlinked
        #if(i<n_O):
        #    B_unlinked_true=B_unlinked_prior     
       
        B_unlinked_true=np.append(B_unlinked_true,0)
        #B_prob=np.append(Likelihood,p_nolink)/sum(Likelihood,p_nolink)
        B_prob=np.append(Likelihood,p_nolink)\
        /sum(Likelihood,p_nolink)
        
        
        #print(p_nolink/sum(Likelihood,p_nolink))
    #Sample new bipartite link for individual i
        link_designation=np.random.choice(B_unlinked_true,size=1,p=B_prob)
    
        #Try this next, I don't think the probabilities are matching to what they should
        #ProbView=pd.DataFrame(data={[B_prob],[B_unlinked_true]})
    #Store information about the link designation sampled
        LinkProbability[i,t]=B_prob[B_unlinked_true==link_designation]
        #print(LinkProbability[i,t])
        #print(B_prob[B_unlinked_true==LinkDesignation[i,t]])
        #What happens here when the if statemaent fails?
        if(link_designation>0):
            #print("linked!")
            SuperGamma.C[np.logical_and(SuperGamma.A_ID==Unique_AID[i],\
                                           SuperGamma.B_ID==link_designation[0])]=1
            #C[link_designation]=1
            LinkDesignation.iloc[LinkDesignation.index==Unique_AID[i],t]=link_designation
            
    print(t)        
    print(sum(LinkDesignation.index==LinkDesignation.iloc[:,t])-amount_known)
            
    LD = pd.DataFrame(LinkDesignation)
    LD.to_csv('to load path',index=True)  
    
    SuperGamma.to_csv('path')
    
