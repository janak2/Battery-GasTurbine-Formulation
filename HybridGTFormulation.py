# -*- coding: utf-8 -*-
"""
Created on Dec 9 14:19:55 2018

@author: Janak Agrawal

This version implements piecewise linear formulation (3 segments) to approximate heat rate curve
"""

# %%  import packages
from __future__ import division
from pyomo.opt import SolverFactory
import pyomo.environ as en
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import collections
from six.moves import xrange
import glob
from sklearn.cluster import KMeans
from sklearn import datasets
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
#import seaborn as sns


# Set to 1 when running on linux
RUN_ON_LINUX=1

# %% Import generator data
if RUN_ON_LINUX ==0:
    ListAllGenLoadFiles = glob.glob('..\\fits_20180911\\*cems_hourly_55853*.csv')
else:
    ListAllGenLoadFiles = glob.glob('..//fits_20180911//*cems_hourly_55853*.csv')

#Create of set generator units
gval = ['g' + `j` for j in range(1,len(ListAllGenLoadFiles)+1)]

LoadColNames = ['Load_MW' + gval[j] for j in range(len(gval))]


############# Import generator specific parameters ##########################################
if RUN_ON_LINUX ==0:
    GenData = pd.read_excel('..\\Data\\GenData_CCGT.xlsx', index_col=[0])
else:
    GenData = pd.read_excel('..//Data//GenData_CCGT.xlsx', index_col=[0])

# Add missing rows for generators not listed in GenData
AdditionalUnitslist = list(set(gval) -set(GenData.index.tolist()))
AdditionalUnitslist.sort() # making sure order of elements is correct g2,g3, and so on

for j in range(len(AdditionalUnitslist)):
    GenData.loc[AdditionalUnitslist[j]] = GenData.loc['g1']

# % Artifically setting differences in VOM costs of different generator types
    # % THIS IS A HACK - NEED TO BE FIXED
#GenData.loc[:,'VOM_dpMWh'] =[1.56, 1.57, 1.58, 1.59]
#GenData.loc[:,'VOM_dpMWh'] =[1.56, 5, 10, 12]

# Importing coefficients of heat rate variation curve
    
if RUN_ON_LINUX ==0:
    HeatRateCoeff_df = pd.read_csv('..\\fits_20180911\\coef_fit_20180911_55853.csv')    
else:
    HeatRateCoeff_df = pd.read_csv('..//fits_20180911//coef_fit_20180911_55853.csv')    

GenData['HRConst_coeff']     = HeatRateCoeff_df['(Intercept)'].tolist()
GenData['HRLin_coeff']     = HeatRateCoeff_df['gload'].tolist()
GenData['HRQuad_coeff']     = HeatRateCoeff_df['I(gload^2)'].tolist()

# Heat rate piecewise linear segments --------------------------------------
# Breakpoints
xval_normalized = [0.33,0.5,0.75,1] # Breakpoints -assumption!!!!

xval = [xval_normalized[j]*GenData.loc[gval[0],'PMax_MW'] for j in range(len(xval_normalized)) ]

# Evaluate average heat rates for each unit to figure which ones are most efficient (on average)
AvgHRbyGen = dict()
for  i in range(len(gval)):
    AvgHRbyGen[gval[i]] =GenData.loc[gval[i],'HRQuad_coeff']*(GenData.loc[gval[i],'Pmin_MW']
    +GenData.loc[gval[i],'PMax_MW'])/float(2)+ GenData.loc[gval[i],'HRLin_coeff'] +GenData.loc[gval[i],'HRConst_coeff']*np.log(GenData.loc[gval[i],'PMax_MW']/GenData.loc[gval[i],'Pmin_MW'])/(GenData.loc[gval[i],'PMax_MW'] -GenData.loc[gval[i],'Pmin_MW'])  # function value
        

#Creating a sorted list of priorities
GenPriorities = list()
for i in range(len(gval)):
    
    for k,v in AvgHRbyGen.items():
    
        if v==sorted(AvgHRbyGen.values())[i]:
            GenPriorities.append(k)
    

#  Import storage data ------------------------------------------

if RUN_ON_LINUX ==0:
    StorageData = pd.read_excel('..\\Data\\StorageData.xlsx',index_col=[0])
else:
    StorageData = pd.read_excel('..//Data//StorageData.xlsx',index_col=[0])

esval = StorageData.index.values.tolist()


Discount_rate = 0.054
#
CRF_storage = [1/float((Discount_rate+1)/float(Discount_rate)*(1-1/(1+Discount_rate)**StorageData.loc[StorageData.index[0],'Lifetime'])) for j in range(len(StorageData.index))]


# Fuel cost and emissions data --------------------------------------------------
EmissionTperMMBTu =0.053513 # Tonnes per MMBtu 
fuelcost =4 # natural gas price in $/MMBtu HHV


# %% Selecting time series -  representative weeks to model operations
#NumGrpDays_val = 7  # Number of consecutive days

#NClusters_val = 1  # Number of clusters of consecutive days

# Function to select representative weeks from sampled data
def GetKmeansClusteringOutputs(InputData,NumGrpDays,NClusters):

    # Define InputData  as PlantLoadbyUnit
    OldColNames = InputData.columns.tolist()
    Nhours = len(InputData)

# Original data
    LoadMW = InputData.sum(axis=1)
# normalized data to be used for clustering process
    # option anchor values b/w 0 and 1
    LoadNormalized = (LoadMW - min(LoadMW))/float(max(LoadMW) -min(LoadMW))
    # scale values to be less than or equal to
#    LoadNormalized = LoadMW

 
    # Number of samples in a year
    NumDataPoints =int(Nhours/24/NumGrpDays)

    DataPointAsColumns = ['p' + `j` for j in range(1,NumDataPoints+1)]

    # Create a dictionary storing groups of time periods to average over for each hour
    HourlyGroupings = {i:[j for j in range(NumGrpDays*24*(i-1),NumGrpDays*24*i)] for i in range(1,NumDataPoints +1)}
    HourlyGroupingswithP = {DataPointAsColumns[i-1]:[j for j in range(NumGrpDays*24*(i-1),NumGrpDays*24*i)] for i in range(1,NumDataPoints +1)}


    #  Create a new dataframe storing aggregated load and renewables time series
    ModifiedDataNormalized = pd.DataFrame(columns = DataPointAsColumns)
    # Original data organized in concatenated column
    ModifiedData = pd.DataFrame(columns = DataPointAsColumns)

    # Creating the dataframe with concatenated columns
    for j in range(0,NumDataPoints):
    
        ModifiedDataNormalized[DataPointAsColumns[j]] = LoadNormalized.loc[HourlyGroupings[j+1]].tolist()
        ModifiedData[DataPointAsColumns[j]] = LoadMW.loc[HourlyGroupings[j+1]].tolist()
 
################################## k-means clustering process
    # create Kmeans clustering model and specify the number of clusters gathered
    # number of replications =100, squared euclidean distance
    
    model = KMeans(n_clusters=NClusters, n_init=100, init ='k-means++', random_state=42)
        
    model.fit(ModifiedDataNormalized.values.transpose())

    # Store clustered data
    # Create an empty list storing weight of each cluster
    EachClusterWeight  = [None]*NClusters 
    
    # Create an empty list storing name of each data point
    EachClusterRepPoint = [None]*NClusters

    for k in range(NClusters):

    # Number of points in kth cluster (i.e. label=0)
        EachClusterWeight[k] = len(model.labels_[model.labels_==k])

        # Compute Euclidean distance of each point from centroid of cluster k
        dist ={ModifiedDataNormalized.loc[:,model.labels_ ==k].columns[j]:np.linalg.norm(ModifiedDataNormalized.loc[:,model.labels_ ==k].values.transpose()[j] -model.cluster_centers_[k]) for j in range(EachClusterWeight[k])}
        print(dist)

        # Select column name closest with the smallest euclidean distance to the mean
        EachClusterRepPoint[k] = min(dist, key = lambda k: dist[k])

# Storing selected groupings in a new data frame with appropriate dimensions (E.g. load in GW)    
    ClusterOutputDataTemp =  ModifiedData[EachClusterRepPoint]

    SampleweeksAnnualMWh = sum([ClusterOutputDataTemp.loc[:,EachClusterRepPoint[j]].sum()*EachClusterWeight[j] for j in range(NClusters)])    
    ScaleFactor =LoadMW.sum()/SampleweeksAnnualMWh

    # Updates the weights rather than the load values!! - otherwise model will exceed installed capacity of gas turbines
#    ClusterOutputDataTemp.loc[0:24*NumGrpDays-1,:] = ScaleFactor*ClusterOutputDataTemp.loc[0:24*NumGrpDays-1,:]
    ScaledEachClusterWeight = [ScaleFactor*EachClusterWeight[jj] for jj in range(len(EachClusterWeight))]

    # New variable 
    ClusterOutputData =ClusterOutputDataTemp
       
    # Store weights for each selected hour  Number of days *24, for each week
#    ClusteredWeights=pd.DataFrame(EachClusterWeight*np.ones([NumGrpDays*24,len(EachClusterWeight)]), columns = EachClusterRepPoint)
    ClusteredWeights=pd.DataFrame(ScaledEachClusterWeight*np.ones([NumGrpDays*24,len(EachClusterWeight)]), columns = EachClusterRepPoint)

    # Storing data as final outputs
#    NewColNames = ['LoadMW','GrpWeight']
    FinalOutputData = pd.DataFrame(columns =['LoadMW','GrpWeight'])


    # Storing weights in final output data column
    FinalOutputData['GrpWeight'] = ClusteredWeights.melt(id_vars=None)['value']

# Regenerating data organized by time series (columns) and representative time periods (hours)
    FinalOutputData['LoadMW'] =ClusterOutputData.melt(id_vars=None)['value']

# Calculating error metrics and Annual load duration curve
# Error calculation needs to be fixed !!!! -
 
    # Constructing histogram of clustered data
    Hist1,binedges1 =np.histogram(FinalOutputData['LoadMW'].values, bins=10000,weights = FinalOutputData['GrpWeight'].values)
    #Energyvalue of each bin
    BinVal1 = Hist1 * np.diff(binedges1)
    
    # Constructing histogram of original data
    Hist2,binedges2 =np.histogram(InputData.sum(axis=1).values, bins=10000)
    
    # Energyvalue of each bin
    BinVal2 = Hist2 * np.diff(binedges2)
#    #  Root mean square error between the duration curves of each time series 
   
    RMSE = np.linalg.norm(BinVal1 -BinVal2)     
    
   # Store individual load profiles of each turbine - need for the "no storage" case
    UnitlevelLoadMW = pd.DataFrame(columns =InputData.columns.tolist())     
        
      
    my_list1 =[]
    for i in range(len(EachClusterRepPoint)):
#        print(EachClusterRepPoint[i])
        my_list1.extend(HourlyGroupingswithP[EachClusterRepPoint[i]])
    
    for k in range(len(OldColNames)):
        UnitlevelLoadMW.loc[:,OldColNames[k]]= InputData.loc[my_list1,OldColNames[k]]

    UnitlevelLoadMW.reset_index(inplace =True, drop=True)
    
#    plt.figure(figsize=(6,10))
#    plt.plot(np.arange(1,len(FullLengthOutputs)+1,1),np.sort(LoadMW.truncate(after=len(FullLengthOutputs)-1).values),label='Data')
#    plt.plot(np.arange(1,len(FullLengthOutputs)+1,1),np.sort(FullLengthOutputs['LoadMW'].values),label=`NClusters` + '_Output')


    return {'Data':FinalOutputData,                 # Scaled Output Load and Renewables profiles for the sampled representative groupings
            'ClusterWeights': ScaledEachClusterWeight,    # Weight of each for the representative groupings
            'AnnualGenScaleFactor':ScaleFactor,     # Scale factor used to adjust load output to match annual generation of original data
            'RMSE': RMSE,                           # Root mean square error between full year data and modeled full year data (duration curves)
           'UnitLevelMW': UnitlevelLoadMW}          # Unit level gas turbine data

# %%  model formulation

def build_model(gendf,stordf,OpYear, NGrpDays,NClusters,HRbreakpoints, genset, storset, ClusteringOutputs, fuelcost):
    
    tval =range(1,NClusters*NGrpDays*24+1,1)
    
    # Defining set
    sval = ['s' + `j` for  j in range(len(HRbreakpoints))]
    
    # Storing function value at breakpoints for each generator type
    yval = dict()

    for i in range(len(genset)):
    
        for j in range(len(xval)):
            yval[genset[i],sval[j]] = GenData.loc[genset[i],'HRQuad_coeff']*HRbreakpoints[j]**2 + GenData.loc[genset[i],'HRLin_coeff']*HRbreakpoints[j] +GenData.loc[genset[i],'HRConst_coeff']  # function value

    # Load profile to be fed to the model
    pLoad = {tval[j]: ClusteringOutputs['Data']['LoadMW'].tolist()[j] for j in range(len(tval))}

# BEGIN - OPTIMIZATION MDOEL FORMULATION

    m = en.ConcreteModel()

#Time
    m.t=en.Set(initialize = tval)
#m.t=en.Set()

# GT technology
    m.g=en.Set(initialize =genset)

# GT heat rate segments
    m.s =en.Set(initialize=sval)

#Energy storage technology
    m.es = en.Set(initialize =storset)

# hourly load profile
    m.pL = en.Param(m.t, within=en.NonNegativeReals, initialize = pLoad, mutable=True)

# GT related variables -------------------------------------
#Dispatched power to the grid by GT -- VAR --
    m.vPGrid = en.Var(m.t, m.g, within=en.NonNegativeReals)

    #GT Power to charge -- VAR --
    m.vP2Charge = en.Var(m.t, m.g, m.es, within=en.NonNegativeReals)

    ##### FIX THIS
#Total GT power -- VAR -- bounds are roughly fixed - 
    m.vPTot = en.Var(m.t, m.g, within=en.NonNegativeReals, bounds =(0,gendf.loc['g1','PMax_MW'])) 

#GT fuel consumption across all segments --VAR --
    m.vGTFuelUse = en.Var(m.t, m.g, within=en.NonNegativeReals) 

# GT commitment status -- VAR --
    m.vy = en.Var(m.t, m.g, within = en.Binary)

# GT startup condition
    m.vystart = en.Var(m.t, m.g, within = en.Binary)

# GT shutdown condition
    m.vyshut = en.Var(m.t, m.g, within = en.Binary)

#GT parameters  -------------------------------------
#Maximum power
    m.pPMax = en.Param(m.g, within=en.NonNegativeReals, initialize=  {genset[j]:gendf.loc[genset[j],'PMax_MW'] for j in range(len(genset))})

# Minimum power
    m.pPMin = en.Param(m.g, within=en.NonNegativeReals, initialize = {genset[j]:gendf.loc[genset[j],'Pmin_MW'] for j in range(len(genset))})

    m.pFOM_GT = en.Param(m.g, within=en.NonNegativeReals, initialize = {genset[j]:gendf.loc[genset[j],'FOM_dpMWyr'] for j in range(len(genset))})

    m.pVOM_GT = en.Param(m.g, within=en.NonNegativeReals, initialize = {genset[j]:gendf.loc[genset[j],'VOM_dpMWh'] for j in range(len(genset))})
    
    m.pFuelCost = en.Param(within=en.NonNegativeReals, initialize = fuelcost)

    m.pNominalHeatRate = en.Param(m.g, within=en.NonNegativeReals, initialize ={genset[j]:gendf.loc[genset[j],'NomHeatRate_MMBtupMWh'] for j in range(len(genset))} )
    
    m.pStartCostGT = en.Param(m.g, within=en.NonNegativeReals, initialize = {genset[j]:gendf.loc[gval[j],'StartCost_MWpstart'] + fuelcost*gendf.loc[genset[j],'StartupFuelUse_MMBtupMWpStart'] for j in range(len(genset))})


# Storage related variables ---------------------------------------------------
#Storage charge -- VAR --
    m.vStCharge = en.Var(m.t, m.es, within=en.NonNegativeReals)

#Storage discharge -- VAR --
    m.vStDischarge = en.Var(m.t, m.es, within=en.NonNegativeReals)

#Storage cap -- VAR --
    m.vStCap = en.Var(m.t, m.es, within=en.NonNegativeReals)

#Installed storage capacity -- VAR -- bounds are arbitrary
    m.vStInstalledCap = en.Var(m.es, within = en.NonNegativeReals, bounds =(0,3000))

# Storage related parameters ------------------------------------------------

# Efficiency of charging
    m.pStEffChg = en.Param(m.es, within=en.NonNegativeReals, initialize ={storset[j]:StorageData.loc[storset[j],'St_eff_chg'] for j in range(len(storset))})

# Efficiency of discharging
    m.pStEffDischg = en.Param(m.es, within=en.NonNegativeReals, initialize = {storset[j]:StorageData.loc[storset[j],'St_eff_dischg'] for j in range(len(storset))})
# power to energy ratio
    m.pP2E = en.Param(m.es, within=en.NonNegativeReals, initialize={storset[j]:StorageData.loc[storset[j],'Power_to_energy_ratio'] for j in range(len(storset))})

# Annualized capital cost of storage - mutable
    m.pCapCostSt = en.Param(m.es, within=en.NonNegativeReals, initialize ={storset[j]:StorageData.loc[storset[j],'AnnualCapCost_dpMWhpyr'] for j in range(len(storset))}, mutable = True)

    m.pFOMSt = en.Param(m.es, within=en.NonNegativeReals,  initialize ={storset[j]:StorageData.loc[storset[j],'FOM_dpMWyr'] for j in range(len(storset))})

    m.pVOMSt = en.Param(m.es, within=en.NonNegativeReals, initialize ={storset[j]:StorageData.loc[storset[j],'VOM_dpMWh'] for j in range(len(storset))})

# Slack variables  for debugging  ------------------------------------------------
    m.UnmetLoadMW = en.Var(m.t, within = en.NonNegativeReals)

# Clustering weights to scale up generation costs on annual basis -------------
    m.HourlyWeightVal = en.Param(m.t, within =en.NonNegativeReals, initialize = {tval[j]:ClusteringOutputs['Data']['GrpWeight'].tolist()[j] for j in range(len(tval))}, mutable =True)

# Objective function with variable heat rate ($)
# Annualized CAPEX of Storage + FOM COST for GT + FOM COST for Storage +
# VOM COST for GT + FUEL COST for GT
# VOM COST for Storage charging + discharging 
# START COST for GT

# Annualized CAPEX of Storage + FOM COST ($/MWh)
    def StorFixCost_rule(m):
        return sum(m.pCapCostSt[k]*m.vStInstalledCap[k] + m.pFOMSt[k]*m.pP2E[k]*m.vStInstalledCap[k] for k in m.es)

    m.StFixCost = en.Expression(rule =StorFixCost_rule)

# Annualized FOM COST for GT -excluded from ojective value
    def GTFixCost_rule(m):
        return sum(m.pFOM_GT[l]*m.pPMax[l] for l in m.g)

    m.GTFixCost = en.Expression(rule = GTFixCost_rule)

# VOM COST for GT + FUEL COST for GT + START COST for GT
    def GTVOMCost_rule(m):
        return sum(sum(m.pVOM_GT[l]*m.vPTot[j,l] + m.pFuelCost*m.vGTFuelUse[j,l] +m.pStartCostGT[l]*m.vystart[j,l]*m.pPMax[l] for l in m.g)*m.HourlyWeightVal[j] for j in m.t)  
    m.GTVOMCost = en.Expression(rule = GTVOMCost_rule)

# VOM COST for Storage charging + discharging 
    def StVOMCost_rule(m):
        return sum(sum(m.pVOMSt[k]*(m.vStCharge[j,k] + m.vStDischarge[j,k]) for k in m.es)*m.HourlyWeightVal[j]  for j in m.t)
    m.StVOMCost = en.Expression(rule = StVOMCost_rule)

# Lost load penalty
    def UnmetLoad_rule(m):
        return sum(m.HourlyWeightVal[j]*m.UnmetLoadMW[j]*100000 for j in m.t)
    m.LostLoadCost = en.Expression(rule = UnmetLoad_rule)

# Objective function value - includes slack term
    def Obj_fn(m):
#     return m.StFixCost + m.GTFixCost + sum(m.GTVOMCost[j] for j in m.t) + sum(m.StVOMCost[j] for j in m.t) + sum(m.UnmetLoadMW[j]*100000 for j in m.t)
        return m.StFixCost + m.GTVOMCost + m.StVOMCost + m.LostLoadCost
#     return value
    m.total_cost = en.Objective(rule=Obj_fn, sense =en.minimize)

# Demand balance - demand met by combination of gas turbine output and storage discharge (derated by efficiency loss) (MW)
    def demandbalance_rule(m, j):
        value = sum(m.vPGrid[j,l] for l in m.g) + sum(m.vStDischarge[j,k] for k in m.es) + m.UnmetLoadMW[j]
        return value == m.pL[j]
    m.demand_balance = en.Constraint(m.t, rule=demandbalance_rule)

# Balance storage capacity at each time step (MWh)
    def StorageCapBalance(m,j,k): # wrapping storage capacity to ensure first and last period are match
        if j in [NumGrpDays_val*24*kk+1 for kk in range(NClusters_val)]:  # first hour of the week
        #  Wrapping each week around itself
            return m.vStCap[j+NumGrpDays_val*24-1,k]+ m.pStEffChg[k]*m.vStCharge[j,k] - m.vStDischarge[j,k]/m.pStEffDischg[k] == m.vStCap[j,k]
        else:
#    value = 
            return m.vStCap[j-1,k]+ m.pStEffChg[k]*m.vStCharge[j,k] - 1/m.pStEffDischg[k]*m.vStDischarge[j,k] == m.vStCap[j,k]
    m.StorageCapBalance = en.Constraint(m.t, m.es, rule=StorageCapBalance)

# Storage can be charged with power from the gas turbines (MW)
    def StchargebyGT(m,j,k):
        return m.vStCharge[j, k] == sum(m.vP2Charge[j, l, k] for l in m.g)
    m.StchargebyGT= en.Constraint(m.t, m.es, rule = StchargebyGT)

# Power output from GT can either go to storage or to grid (MW)
    def GTPowerBalance(m, j,l):
        return sum(m.vP2Charge[j, l, k] for k in m.es) + m.vPGrid[j,l] == m.vPTot[j,l]
    m.GTPowerBalance = en.Constraint(m.t, m.g, rule = GTPowerBalance)

# Minimum operating level for GT (MW)
    def GTMinOperatingLevel(m, j,l):
        return m.vPTot[j,l]>= m.pPMin[l]* m.vy[j,l]
    m.GTMinOperatingLevel = en.Constraint(m.t, m.g, rule = GTMinOperatingLevel) 

# Maximum operating level for GT (MW)
    def GTMaxOperatingLevel(m, j,l):
        return m.vPTot[j,l]<= m.pPMax[l]* m.vy[j,l] 
    m.GTMaxOperatingLevel = en.Constraint(m.t, m.g, rule = GTMaxOperatingLevel) 

# Unit commitment state for the GT
    def UCommit(m, j, l):
        if j in [NumGrpDays_val*24*kk+1 for kk in range(NClusters_val)]: #  Pre-assigning the initial state of GT tobe based on commitment form last period
            return m.vy[j +NumGrpDays_val*24-1,l] +m.vystart[j,l] - m.vyshut[j,l] == m.vy[j,l]
        else:
            return m.vy[j-1,l] + m.vystart[j,l] - m.vyshut[j,l] == m.vy[j,l]
#    return m.vy[j-1,l] + m.vystart[j,l] - m.vyshut[j,l] == m.vy[j,l]
    m.UCommit = en.Constraint(m.t, m.g, rule = UCommit)
    
# Upper limit on charge rate into the battery
    def LimCharge_rule(m, j, k):
        return m.pStEffChg[k]*m.vStCharge[j,k] <= m.pP2E[k]*m.vStInstalledCap[k] 
    m.Lim_Charge = en.Constraint(m.t, m.es, rule = LimCharge_rule)
    
# Upper limit on discharge rate into the battery
    def LimDischarge_rule(m, j,k):
        return m.vStDischarge[j,k] <=m.pStEffDischg[k]*m.pP2E[k]*m.vStInstalledCap[k]
    m.Lim_Discharge = en.Constraint(m.t, m.es, rule = LimDischarge_rule)

# Storage capacity cannot exceed purchased capacity
    def StCap_rule(m,j,k):
        return m.vStCap[j,k]<=m.vStInstalledCap[k]
    m.St_Cap = en.Constraint(m.t, m.es, rule = StCap_rule)

#------------Piecewise implementation of heat rate segment----------------------
    m.sos_var_indices = en.Set(ordered=True, initialize= sval) # Set defining number of break points

    def SOS_indices_init(m,j,l):
        return [(j,l,i) for i in sval]
    m.SOS_indices = en.Set(m.t,m.g,dimen=3, ordered=True, initialize=SOS_indices_init)

# Indicator variables for each time step and each generator
    m.vySOS2 = en.Var(m.t,m.g, m.sos_var_indices,within=en.NonNegativeReals) # SOS2 variable

    def constraint1_rule(m,j,l): # piecewise linear power input
        return m.vPTot[j,l] == sum(m.vySOS2[j,l,sval[i]]*HRbreakpoints[i] for i in range(len(sval)))

    def constraint2_rule(m,j,l): # piecewise linear output of fuel use
        return m.vGTFuelUse[j,l] == sum(m.vySOS2[j,l,sval[i]]*yval[l,sval[i]] for i in range(len(sval)))
    
    def constraint3_rule(m,j,l): # SOS constraint is only active power plant is turned on
        return sum(m.vySOS2[j,l,sval[i]] for i in range(len(sval))) == m.vy[j,l]

    m.constraint1 = en.Constraint(m.t,m.g,rule=constraint1_rule)
    m.constraint2 = en.Constraint(m.t,m.g,rule=constraint2_rule)
    m.constraint3 = en.Constraint(m.t,m.g,rule=constraint3_rule)
    m.SOS_set_constraint = en.SOSConstraint(m.t,m.g, var=m.vySOS2,index =m.SOS_indices, sos=2)
  
    # Include constraints to break symmetry among generators 
    expr =0
    m.clist = en.ConstraintList()
    for k in range(len(GenPriorities)-1): # All but last element
        for j in range(1,len(tval)+1):
            expr = m.vy[j,GenPriorities[k]] -m.vy[j,GenPriorities[k+1]]
#            print(expr)
            m.clist.add(expr>=0)
            expr =0
    # Return model object
    return m

# %%
#  EXPORT MODEL OUTPUTS ------------------------------------------------------
def GetModelOutputs(instancename):
    # Storing all the outputs in a single dictionary
    OutputsbyUnitbyVarName =collections.OrderedDict()
    
    # Time dependent parameters to be stored
    GenVarNames =['vPGrid','vPTot','vy','vyshut','vystart','vGTFuelUse', 'pGTFuelUseConstantHR']
    StorVarNames =['vStCharge', 'vStDischarge','vStCap']
    
    Timesteps = len(instancename.t.data())
    
    # Creating a list of column names for time-dependent data
    GenColNames ={i:[gval[i] +'_' + GenVarNames[j] for j in range(len(GenVarNames))] for i in range(len(gval))}
    StorColNames={i:[esval[i] +'_' + StorVarNames[j] for j in range(len(StorVarNames))] for i in range(len(esval))}
    TdataAllColNames = np.concatenate(GenColNames.values()).tolist() + np.concatenate(StorColNames.values()).tolist()  +['LostLoadMW']
    
    StorCapColNames ={i:esval[i] +'_InstalledCapMWh' for i in range(len(esval))}
    StaticOutputsColNames =['StFixCost', 'GTVOMCost' , 'StVOMCost', 'LostLoadCost','Totalcost', 'TotalEmissions_tons', 'EI_kgCO2_MWh'] + [esval[i] +'_InstalledCapMWh' for i in range(len(esval))]
    
    # OUTPUT DATA FRAME # 1 -------------------------------------------------------
    # Storing objective function and installed storage capacity in separate dataframe
    OutputsbyUnitbyVarName['StaticOutputs'] = pd.DataFrame(np.zeros([1,len(StaticOutputsColNames)]),columns =StaticOutputsColNames)
    
    OutputsbyUnitbyVarName['StaticOutputs'].loc[0,'StFixCost'] = en.value(instancename.StFixCost)
    OutputsbyUnitbyVarName['StaticOutputs'].loc[0,'GTVOMCost'] = en.value(instancename.GTVOMCost)
    OutputsbyUnitbyVarName['StaticOutputs'].loc[0,'StVOMCost'] = en.value(instancename.StVOMCost)
    OutputsbyUnitbyVarName['StaticOutputs'].loc[0,'TotalCost'] = en.value(instancename.total_cost)
    OutputsbyUnitbyVarName['StaticOutputs'].loc[0,'TotalEmissions_tons'] = EmissionTperMMBTu*sum([sum([en.value(instancename.vGTFuelUse[k,gval[i]]) for k in range(1,Timesteps+1)]) for i in range(len(gval))])
    OutputsbyUnitbyVarName['StaticOutputs'].loc[0,'EI_kgCO2_MWh'] = EmissionTperMMBTu*sum([sum([en.value(instancename.vGTFuelUse[k,gval[i]]) for k in range(1,Timesteps+1)]) for i in range(len(gval))])*1000/sum([en.value(instancename.pL[j]) for j in range(1,Timesteps+1,1)])
    # OUTPUT DATA FRAME # 2 -------------------------------------------------------
    #Storing time series data related to GT  and storage operation
    OutputsbyUnitbyVarName['tdata'] = pd.DataFrame(columns =TdataAllColNames)
    
    # Lost load in each hour
    OutputsbyUnitbyVarName['tdata']['LostLoadMW'] =[en.value(instancename.UnmetLoadMW[k]) for k in range(1,Timesteps+1)]

    for  i in range(len(gval)): # All generator related operational variables
        
        OutputsbyUnitbyVarName['tdata'][GenColNames[i][0]] =[en.value(instancename.vPGrid[k,gval[i]]) for k in range(1,Timesteps+1)]
        OutputsbyUnitbyVarName['tdata'][GenColNames[i][1]] =[en.value(instancename.vPTot[k,gval[i]]) for k in range(1,Timesteps+1)]
        OutputsbyUnitbyVarName['tdata'][GenColNames[i][2]] =[en.value(instancename.vy[k,gval[i]]) for k in range(1,Timesteps+1)]
        OutputsbyUnitbyVarName['tdata'][GenColNames[i][3]] =[en.value(instancename.vystart[k,gval[i]]) for k in range(1,Timesteps+1)]
        OutputsbyUnitbyVarName['tdata'][GenColNames[i][4]] =[en.value(instancename.vyshut[k,gval[i]]) for k in range(1,Timesteps+1)]
        OutputsbyUnitbyVarName['tdata'][GenColNames[i][5]] =[en.value(instancename.vGTFuelUse[k,gval[i]]) for k in range(1,Timesteps+1)]
        OutputsbyUnitbyVarName['tdata'][GenColNames[i][6]] =[en.value(instancename.vPTot[k,gval[i]])*en.value(instancename.pNominalHeatRate[gval[i]]) for k in range(1,Timesteps+1)]
                        
    for i in range(len(esval)):    # storage related operational variables
            # Storing installed storage capacity in MWh
#           print(en.value(instancename.vStInstalledCap[esval[i]]))
#           print(OutputsbyUnitbyVarName['StaticOutputs'].loc[0,StorCapColNames[i]])
           OutputsbyUnitbyVarName['StaticOutputs'].loc[0,StorCapColNames[i]] =en.value(instancename.vStInstalledCap[esval[i]])
           
           OutputsbyUnitbyVarName['tdata'][StorColNames[i][0]] =[en.value(instancename.vStCharge[k,esval[i]]) for k in range(1,Timesteps+1)]
           OutputsbyUnitbyVarName['tdata'][StorColNames[i][1]] =[en.value(instancename.vStDischarge[k,esval[i]]) for k in range(1,Timesteps+1)]
           OutputsbyUnitbyVarName['tdata'][StorColNames[i][2]] =[en.value(instancename.vStCap[k,esval[i]]) for k in range(1,Timesteps+1)]
                
    return OutputsbyUnitbyVarName

# Set solve options
opt = SolverFactory("gurobi")
opt.options['MIPGap'] = 0.02
opt.options['Pre_Solve']=2
opt.options['NodefileStart']=0.5
opt.options['cuts']=3
opt.options['TimeLimit']=10000

# %% Solving model for reference scenario and multiple years of operation
#AvailableYears =[2011]   # Excluded leap year data
AvailableYears =[2011,2013,2014,2015]   # Both units operational from 2011 onwards

# Store all the outputs in one forlder
OutputsbyYearbyInstance= collections.OrderedDict()

NumGrpDays_val = 7  # Number of consecutive days

NClusters_val =52  # Number of clusters of consecutive days

Timesteps =  NClusters_val*NumGrpDays_val*24
TimeIndex =range(1, Timesteps+1,1)


 
# solve for multiple years
for k in range(len(AvailableYears)):
    print(AvailableYears[k])
    # Dataframe of load profiles for the sampled period
    PlantLoadbyUnit = pd.DataFrame(columns = LoadColNames)
    OutputsbyYearbyInstance[AvailableYears[k]] =collections.OrderedDict()

    for j in range(len(ListAllGenLoadFiles)):
        MultiYearLoadFuelData = pd.read_csv(ListAllGenLoadFiles[j])

        # Replace NaN data with 
        MultiYearLoadFuelData =MultiYearLoadFuelData.fillna(value=0)
        
            # Defining time series input - load and renewables data
        StartIndex = MultiYearLoadFuelData.index[MultiYearLoadFuelData.date ==`AvailableYears[k]` + '-01-01'][0]
        FinalIndex = MultiYearLoadFuelData.index[MultiYearLoadFuelData.date ==`AvailableYears[k]` + '-12-31'][-1]

     #PLoadSeries =MultiYearLoadFuelData.loc[StartIndex:FinalIndex,'gload_mwh']
        PlantLoadbyUnit.loc[:,LoadColNames[j]] =MultiYearLoadFuelData.loc[StartIndex:FinalIndex,'gload_mwh'].tolist()
   
    # Excluding hourly generation that is below the design stable minimum generation level
        idx = PlantLoadbyUnit[LoadColNames[j]]<GenData.loc[gval[j],'Pmin_MW']
        PlantLoadbyUnit.loc[idx,LoadColNames[j]]=0

#         Reset index
    PlantLoadbyUnit.reset_index(drop=True,inplace =True)
    
   
    if NClusters_val <52: # Clustering is performed
#     K-means clustering output - selection of representative weeks of grid operations to be modeled
        OutputsbyGrpsbyNClusters = GetKmeansClusteringOutputs(PlantLoadbyUnit,NumGrpDays_val,NClusters_val)
        print(OutputsbyGrpsbyNClusters['RMSE'])
        
    else: # bypass clustering process and use entire dataset
        OutputsbyGrpsbyNClusters=collections.OrderedDict()
        OutputsbyGrpsbyNClusters['Data']   =collections.OrderedDict()
        OutputsbyGrpsbyNClusters['Data']['LoadMW'] =PlantLoadbyUnit.sum(axis=1)
        OutputsbyGrpsbyNClusters['Data']['GrpWeight'] =np.ones(len(PlantLoadbyUnit))
        OutputsbyGrpsbyNClusters['UnitLevelMW'] = PlantLoadbyUnit
#    
#    # Define dictionary to store problem outputs
    OutputsbyYearbyInstance[AvailableYears[k]]['Ref'] = collections.OrderedDict()

    # Range of storage capital costs  
    StorageCapCostperkWh = range(0,250,50) #  $/kWh total installed capital cost

    for i in range(len(StorageCapCostperkWh)):
        print('Storage capital costs = %d $/kWh' %(StorageCapCostperkWh[i]))
    # Update annualized capital cost of storage in $/MW/yr
        StorageData.loc[:,'AnnualCapCost_dpMWhpyr']= [StorageCapCostperkWh[i]*CRF_storage[j]*1000 for j in range(len(StorageData))]

    #def build_model(gendf,stordf,OpYear, NGrpDays,NClusters,HRbreakpoints, genset, storset, ClusteringOutputs, fuelcost):    
        Refinstance = build_model(GenData,StorageData,AvailableYears[k], 
            NumGrpDays_val,NClusters_val,xval, 
            gval, esval, 
            OutputsbyGrpsbyNClusters, fuelcost)

#     Solve model for specified parameters    
        Refresults = opt.solve(Refinstance, tee=True)
    #Store outputs in dictionary
        OutputsbyYearbyInstance[AvailableYears[k]]['Ref'][StorageCapCostperkWh[i]] = GetModelOutputs(Refinstance)

        
# Reference cases Fix storage=0, set power output from each unit same as data - compare effect of storage optimization
    for j in range(1,Timesteps+1):
        for i in range(len(gval)):  # Fixing power output from gas turbines to match data
            Colval =PlantLoadbyUnit.columns.tolist()[i]
            Refinstance.vPTot[j,gval[i]] = OutputsbyGrpsbyNClusters['UnitLevelMW'].loc[j-1,Colval]
            Refinstance.vPTot[j,gval[i]].fixed=True
            
# Deactivate priority constraints
    Refinstance.clist.deactivate()
    
    NoStorresults = opt.solve(Refinstance, tee=True)
    OutputsbyYearbyInstance[AvailableYears[k]]['NoStor'] = GetModelOutputs(Refinstance)

#    #EXPORT OUTPUTS TO CSV
    if RUN_ON_LINUX ==0:
        for i in range(len(StorageCapCostperkWh)):
            OutputsbyYearbyInstance[AvailableYears[k]]['Ref'][StorageCapCostperkWh[i]]['StaticOutputs'].to_csv('..\\Outputs\\Ref_Cluster'+`NClusters_val` +'weeks_StCap' +`StorageCapCostperkWh[i]` + '_'+`AvailableYears[k]` +'_StaticOutputs_CC.csv', index=False)
            OutputsbyYearbyInstance[AvailableYears[k]]['Ref'][StorageCapCostperkWh[i]]['tdata'].to_csv('..\\Outputs\\Ref_Cluster'+`NClusters_val` +'weeks_StCap' +`StorageCapCostperkWh[i]` + '_'+`AvailableYears[k]` +'_TimeSeriesOutputs_CC.csv', index = False)
            
        OutputsbyYearbyInstance[AvailableYears[k]]['NoStor']['StaticOutputs'].to_csv('..\\Outputs\\NoStor_Cluster'+`NClusters_val` +'weeks_' +`AvailableYears[k]` +'_StaticOutputs_CC.csv', index=False)
        OutputsbyYearbyInstance[AvailableYears[k]]['NoStor']['tdata'].to_csv('..\\Outputs\\NoStor_Cluster'+`NClusters_val` +'weeks_' +`AvailableYears[k]` +'_TimeSeriesOutputs_CC.csv', index=False)
    else:
        for i in range(len(StorageCapCostperkWh)):
            OutputsbyYearbyInstance[AvailableYears[k]]['Ref'][StorageCapCostperkWh[i]]['StaticOutputs'].to_csv('..//Outputs//Ref_Cluster'+`NClusters_val` +'weeks_StCap' +`StorageCapCostperkWh[i]` + '_'+`AvailableYears[k]` +'_StaticOutputs_CC.csv', index=False)
            OutputsbyYearbyInstance[AvailableYears[k]]['Ref'][StorageCapCostperkWh[i]]['tdata'].to_csv('..//Outputs//Ref_Cluster'+`NClusters_val` +'weeks_StCap' +`StorageCapCostperkWh[i]` + '_'+`AvailableYears[k]` +'_TimeSeriesOutputs_CC.csv', index = False)
        OutputsbyYearbyInstance[AvailableYears[k]]['NoStor']['StaticOutputs'].to_csv('..//Outputs//NoStor_Cluster'+`NClusters_val` +'weeks_' +`AvailableYears[k]` +'_StaticOutputs_CC.csv', index=False)
        OutputsbyYearbyInstance[AvailableYears[k]]['NoStor']['tdata'].to_csv('..//Outputs//NoStor_Cluster'+`NClusters_val` +'weeks_' +`AvailableYears[k]` +'_TimeSeriesOutputs_CC.csv', index=False)
    
#     EXPORT OUTPUTS TO CSV
    
# %%  PLot time series outputs    
year = 2011
#StorageCapCost =50
#
#dfresults = OutputsbyYearbyInstance[year]['Ref'][StorageCapCost]['tdata']
#
## specified plant load profile for sampled period
#sysloadMW = [en.value(Refinstance.pL[j]) for j in range(1,Timesteps+1)]
#
#start = 5*168+1
#end = 6*168
#
#sns.set(font_scale=1.1, context ='talk',style ='white')
#
#f,ax = plt.subplots(4, sharex=True, figsize=(12,12)) 
## load, storage state of charge and GT power output with storage and GT power output without storage
#
#ax[0].plot(range(start,end+1), 
#           [sysloadMW[j-1] for j in range(start,end+1)],
#           c='C0')
#ax[0].set_ylabel('System Load (MW)')
#
#
#ax[1].plot(range(start,end+1),
#  dfresults.loc[start-1:end-1,'es1_vStCap'],
#           c='C1', label='State of charge')
#ax[1].set_ylabel('Storage energy MWh')
#
## GT1
#ax[2].plot(range(start,end+1),
#  dfresults.loc[start-1:end-1,'g1_vPTot'],
#           c='C2', label='CC 1 output')
#ax[2].set_ylabel('Power output (MW)')
#ax[2].plot(range(start,end+1),
#  OutputsbyGrpsbyNClusters['UnitLevelMW'].loc[start-1:end-1,'Load_MWg1'],
#           c='C3', label='CC 1 original output')
#ax[2].set_ylabel('Power output (MW)')
#ax[2].legend(loc =1, ncol=2)
#
## GT2
#ax[3].plot(range(start,end+1),
#  dfresults.loc[start-1:end-1,'g2_vPTot'],
#           c='C2', label='CC 2 output')
#ax[3].set_ylabel('Power output (MW)')
#ax[3].plot(range(start,end+1),
#  OutputsbyGrpsbyNClusters['UnitLevelMW'].loc[start-1:end-1,'Load_MWg2'],
#           c='C3', label='CC 2 original output')
#ax[3].set_ylabel('Power output (MW)')
#ax[3].legend(loc =1, ncol=2)
#
#for i in range(4):
#    ax[i].grid(axis='x')
#ax[0].set_xlim(start,end)
#f.autofmt_xdate()
#f.subplots_adjust(hspace=0.2)
#
#
## Plot state of charge and charge discharge patterns
#fontsizeval = 14
## Plot Storage installed vs. capital cost of storage
#sns.set(font_scale=1.1, context ='talk',style ='white')
#f1,ax1 = plt.subplots(2, sharex=True, figsize=(12,12)) # load, storage state of charge and GT power output with storage and GT power output without storage
#
#ax1[0].plot(range(start,end+1),
#  dfresults.loc[start-1:end-1,'es1_vStCap'],
#           c='C1', label='State of charge')
#ax1[0].set_ylabel('Storage energy MWh')
#
#
#ax1[1].plot(range(start,end+1),
#  dfresults.loc[start-1:end-1,'es1_vStDischarge'] -dfresults.loc[start-1:end-1,'es1_vStCharge'],
#           c='C1', label='Discharge-charge profile')
#ax1[1].set_ylabel('Power output (MW)')
#ax1[1].axhline(0,lw=0.5,c='k')

# %%  Other plots to make

# Plot of storage  deployment as a function of capital costs

# Storage deployment as a function of years

# Impact of storage deployment on startups

