

# Python packages
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# %matplotlib inline 
!pip install mpld3
import mpld3
mpld3.enable_notebook()

from scipy.integrate import odeint
!pip install lmfit
import lmfit
from lmfit.lineshapes import gaussian, lorentzian
from lmfit import Minimizer, Parameters, fit_report

import copy

from scipy import optimize


import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import math

!pip install pandas_bokeh
import pandas_bokeh
pandas_bokeh.output_notebook()
import seaborn as sns


from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
#from matplotlib import pyplot as plt

import scipy.interpolate

"""# Data Entries"""

# locking the randomized system to get the same results every time
import random
random.seed(30)

np.random.seed(20)

"""## Population of region"""

# POPULATION OF THE REGION
#===============================================================================================================================================

#The initial population of targeted region needs to be specified
population_Region = 5147712 #Q3 estimate of BC population (https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000901)

#it will then needs to be assigned to the total population that model receives 
total_population= population_Region

"""## Daily infections data"""

# DATA RETRIEVAL FUNCTION, PART I
#===============================================================================================================================================

#Inserting the "REAL" data into the model
#This file has been retrieved from: http://www.bccdc.ca/health-info/diseases-conditions/covid-19/data

#raw_data = pd.read_csv(path)

raw_data=pd.read_csv('https://raw.githubusercontent.com/Kavehdkh/covid-data/main/BCCDC_COVID19_Dashboard_Case_Details.csv')

#Creating a data frame with only the reported date
report_date=pd.DataFrame(raw_data['Reported_Date'])

#Counting the number of reported date and creating a data frame with number of daily cases
daily_cases_data=report_date.groupby(report_date.columns.tolist()).size().reset_index().\
    rename(columns={0:'Cases'})

#removing the previous index set by the original file
daily_cases_data = daily_cases_data.reset_index()

#setting the reported date into date format
daily_cases_data['Reported_Date'] = pd.to_datetime(daily_cases_data['Reported_Date'],format='%Y-%m-%d').dt.date

#setting the range of data based on the starting and end date of the available data, which willbe used as an index to fill the gap
idx_1 = pd.date_range(min(daily_cases_data['Reported_Date']),max(daily_cases_data['Reported_Date']))
#idx = pd.date_range('01-26-2020', '10-16-2020')

#setting the index  of data based on the date column of data
daily_cases_data.set_index(daily_cases_data.Reported_Date, inplace=True)

#filling the missing data with zeros
daily_cases_data = daily_cases_data.reindex(idx_1, fill_value=0)
#daily_cases_data=daily_cases_data.fillna(0)
#Droping the extra columns
daily_cases_data = daily_cases_data.drop(columns=['Reported_Date','index'])

#Creating the array of data, in this section we assign the value of desired data into an array
#daily_cases = daily_cases["Cases"].values[::1]

#Creating an array of Cumulative daily cases
#daily_cases_cumulative = (np.cumsum(daily_cases))

daily_cases_data

# DATA RETRIEVAL FUNCTION, PART II
#===============================================================================================================================================

#Importing and cleaning data
#This section needs input from the user
#======================================================================================================================================
#importing the entire data set for canadian number of cases and deaths
#This file has been retrieve from: https://www.canada.ca/en/public-health/services/diseases/2019-novel-coronavirus-infection.html 
comp_data = pd.read_csv('https://raw.githubusercontent.com/Kavehdkh/covid-data/main/covid19-Complete%20Data%20Canada.csv')

#this is the range of data on which we want to perform the analysis on
idx = pd.date_range('01-26-2020', '10-09-2020')

#which provience do you want to do the analysis on:
provience='British Columbia'

#percentage format can be set up as well, the following code can be used for this purpose
#the other line needs to be activated further in the code
#percentage_train_set=80

#======================================================================================================================================

#Filtering the data for number of deaths, you can choose different category from the data
#prname : province name
modified_data=comp_data.filter(items=['prname', 'date','numdeaths','numtoday','numconf'])

#Filtering the data for the specific region
modified_data_provincial=modified_data.loc[modified_data['prname'] == provience]

#removing the previous index set by the original file
modified_data_provincial = modified_data_provincial.reset_index()

#assigning the date column into date format, this is important to do so that the model knows the data is based on specific series
modified_data_provincial['date'] = pd.to_datetime(modified_data_provincial['date'],format='%Y-%m-%d').dt.date

#setting the range of data based on the starting and end date of the available data, which willbe used as an index to fill the gap
idx_2 = pd.date_range(min(modified_data_provincial['date']),max(modified_data_provincial['date']))

#setting the index  of data based on the date column of data
modified_data_provincial.set_index(modified_data_provincial.date, inplace=True)

#filling the missing data with zeros
modified_data_provincial = modified_data_provincial.reindex(idx_2, fill_value=0)
#modified_data_provincial=modified_data_provincial
#Droping the extra columns
modified_data_provincial = modified_data_provincial.drop(columns=['date','index','prname'])

"""## CIHI data

### Retrieval from drive

### Reading data
"""

#CIHI DATA
#===================================================================================================================================================
#Reading the csv file
#skipping the first 7 rows as they are not data related rows
comp_data_hospitals = pd.read_csv('7132 - Acute Inpatient Activity Report_pwp.csv', skiprows=range(16))

#Creating an array with column names
column_names = comp_data_hospitals.columns.values

# ==> IMPORTANT: USER INPUT (This section needs user input)
#=======================================================================================
#specifing the columns that needs to be turn into numeric
#the first 2 columns are facitiy id and facity name
start_col=3
end_column=19

#As per CIHI's Privacy Policy, cells with counts of between 1 and 4 have been suppressed
#Therefore replacing them with:
star_replacement = 1
comp_data_hospitals = comp_data_hospitals.replace('*',star_replacement)
#=======================================================================================

#replacing NaN values with zero
comp_data_hospitals = comp_data_hospitals.replace(np.nan,0)

#looping over the all columns and make them into numeric format
for i in range (start_col,end_column):
    comp_data_hospitals[column_names[i]]=pd.to_numeric(comp_data_hospitals[column_names[i]])

"""### Data seclection (provincial)"""

# ==> IMPORTANT: USER INPUT (This section needs user input)
#=======================================================================================
#which province you want to look into
provience_CIHI='BC'

#filtering the data to ones we are interested in 
modified_data_CIHI=comp_data_hospitals.filter(items=['Date',
                                                     'Hospital Admissions2',
                                                     'Hospital ALOS16 (in days)',
                                                     'Province21',
                                                     'Discharges11',
                                                     'Deaths12'])



#the index of previous sets 
#idx = pd.date_range('01-26-2020', '10-09-2020')
#=======================================================================================

#Filtering the data for the specific region
provincial_data_CIHI=modified_data_CIHI.loc[modified_data_CIHI['Province21'] == provience_CIHI]

#before making any modification on data, make a copy to find constant parameters such as gamma, alpha, ...
modified_data_CIHI_parameter_finder=provincial_data_CIHI

#removing the previous index set by the original file
provincial_data_CIHI = provincial_data_CIHI.reset_index()

#dropping index column and province column since there is no use for it anymore
provincial_data_CIHI = provincial_data_CIHI.drop(columns=['index','Province21'])

#formating column 'Date' into a date format
provincial_data_CIHI['Date'] = pd.to_datetime(provincial_data_CIHI['Date'],format='%d-%b-%Y')

#Adding all pateint on the same date together
provincial_data_CIHI = provincial_data_CIHI.groupby('Date').sum().reset_index()


# ==> IMPORTANT: USER INPUT (This section needs user input)
#=======================================================================================
#Applying modification on the data
#provincial_data_CIHI['combine_admi_stay']=provincial_data_CIHI['Hospital Admissions2']+provincial_data_CIHI['Hospital ALOS16 (in days)']
#=======================================================================================


#setting the range of data based on the starting and end date of the available data, which will be used as an index to fill the gap
idx_CIHI = pd.date_range(min(provincial_data_CIHI['Date']),max(provincial_data_CIHI['Date']))

#setting the index  of data based on the date column of data
provincial_data_CIHI.set_index(provincial_data_CIHI.Date, inplace=True)

#filling the missing data with zeros
provincial_data_CIHI = provincial_data_CIHI.reindex(idx_CIHI, fill_value=0)


#Droping the extra columns
provincial_data_CIHI = provincial_data_CIHI.drop(columns=['Date'])



#provincial_data_CIHI
#combine

"""### Hospitalization calculation"""

# Calculation of hospitalization 
#==================================================================================
#calculating the influx of patients into and out of hospitals
provincial_data_CIHI['hospital_influx']=provincial_data_CIHI['Hospital Admissions2']-provincial_data_CIHI['Discharges11']-provincial_data_CIHI['Deaths12']
#calculating the daily discharge, combining both dead and recovered
provincial_data_CIHI['discharge']=provincial_data_CIHI['Discharges11']+provincial_data_CIHI['Deaths12']
# the daily hospitalization is the cumulative of hospital influx is calculated
provincial_data_CIHI['daily_Hospitalizations']=provincial_data_CIHI['hospital_influx'].cumsum()
provincial_data_CIHI['daily_Hospitalizations'][3]=0
provincial_data_CIHI['discharge'][3]=0
np.array(provincial_data_CIHI['daily_Hospitalizations'])
provincial_data_CIHI

test=provincial_data_CIHI.copy()
test['daily_Hospitalizations']=test['daily_Hospitalizations'].replace(0,1)
(test['Hospital ALOS16 (in days)']/test['daily_Hospitalizations']).mean()

plt.plot(provincial_data_CIHI['daily_Hospitalizations'])

"""### Data filtering"""

# ==> IMPORTANT: requires modification if different interval or data set is needed
#=======================================================================================
#combing the index of previus sets with the current one
idx_combine=pd.date_range(max(min(idx_1),min(idx_2),min(idx_CIHI)),min(max(idx_1),max(idx_2),max(idx_CIHI)))
#=======================================================================================

#adjusting the data frames based on the shared date in each data frame
daily_cases_data = daily_cases_data.reindex(idx_combine)
modified_data_provincial = modified_data_provincial.reindex(idx_combine)
provincial_data_CIHI = provincial_data_CIHI.reindex(idx_combine)

#Creating an array of daily cases from the canada.ca website
daily_cases = modified_data_provincial["numtoday"].values[::1]

#Creating the array of data, in this section we assign the value of desired data into an array
provincial_data = modified_data_provincial["numdeaths"].values[::1]

#Creating an array of cumulatuve daily cases from the canada.ca website
daily_cases_cumulative = modified_data_provincial["numconf"].values[::1]

#namespace = globals()
#for col in provincial_data_CIHI_modified.columns.values:
#    namespace['%s' % col] = provincial_data_CIHI_modified[col].values[::1]

# Creating an array of daily data for each desired compratment
daily_death_H = provincial_data_CIHI["Deaths12"].values[::1]
daily_recovered_H = provincial_data_CIHI["Discharges11"].values[::1]
daily_admitted_H = provincial_data_CIHI["Hospital Admissions2"].values[::1]
daily_hospitalization = provincial_data_CIHI["daily_Hospitalizations"].values[::1]
daily_discharge = provincial_data_CIHI["discharge"].values[::1]

provincial_data_CIHI.head()

plt.plot(daily_hospitalization)
len(daily_hospitalization)
#daily_hospitalization
daily_hospitalization

"""## Functions

### Interpolation function:
"""

def linear_interpolation (sr_origin,sr_grouped,period_info_intpl):

  # assiging the dataframes
  #df_origin=test#complete_pandemic_data['daily_admitted_H']
  sr_grouped=sr_grouped.reset_index()
  limit_complete_period=math.floor(len(sr_origin)/period_info_intpl)
  limit_complete_total=math.ceil(len(sr_origin)/period_info_intpl)

  # Creating two separate dataframe based on the completeness of the periods
  df_temp_1=sr_grouped.iloc[:limit_complete_period+1,]
  df_temp_2=sr_grouped.iloc[limit_complete_period:limit_complete_total+1,]

  # x and y of the data that interpolation will be based on
  x_temp_1 = df_temp_1.iloc[:,0]
  x_temp_2 = df_temp_2.iloc[:,0]
  y_temp_1 = df_temp_1.iloc[:,1]
  y_temp_2 = df_temp_2.iloc[:,1]
  if len(df_temp_2)>1:
    # creating two interpolation functions:
    y_interpy_temp_1 = scipy.interpolate.interp1d(x_temp_1, y_temp_1)
    y_interpy_temp_2 = scipy.interpolate.interp1d(x_temp_2, y_temp_2)

    #creating x-arrays based on the number of x in the original array
    x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=limit_complete_period*period_info_intpl, endpoint=True)    
    x_origin_2=np.linspace(df_temp_2.iloc[0,0], df_temp_2.iloc[-1,0], num=len(sr_origin)-limit_complete_period*period_info_intpl+1, endpoint=True)    

    # perfoming the interpolation:
    y_interpolated_origin_1=y_interpy_temp_1(x_origin_1)
    y_interpolated_origin_2=y_interpy_temp_2(x_origin_2)

    # putting the two arrays into one another, we need to remove the overlapping period
    final_array=np.concatenate([y_interpolated_origin_1,y_interpolated_origin_2[1:]])

  elif len(df_temp_2)==1:

    y_interpy_temp_1 = scipy.interpolate.interp1d(x_1_temp_1, y_temp_1)
    x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=limit_complete_period*period_info_intpl, endpoint=True)    
    y_interpolated_origin_1=y_admitted_interpy_admitted_temp_1(x_origin_1)
    final_array=y_interpolated_origin_1

  return final_array

def linear_interpolation_2 (sr_origin,sr_grouped,period_info_intpl,inter_kind):

  # assiging the dataframes
  #df_origin=test#complete_pandemic_data['daily_admitted_H']
  sr_grouped=sr_grouped.reset_index()
  limit_complete_period=math.floor(len(sr_origin)/period_info_intpl)
  limit_complete_total=math.ceil(len(sr_origin)/period_info_intpl)

  # Creating two separate dataframe based on the completeness of the periods
  df_temp_1=sr_grouped.iloc[:limit_complete_period+1,]
  df_temp_2=sr_grouped.iloc[limit_complete_period:limit_complete_total+1,]

  # x and y of the data that interpolation will be based on
  x_temp_1 = df_temp_1.iloc[:,0]
  x_temp_2 = df_temp_2.iloc[:,0]
  y_temp_1 = df_temp_1.iloc[:,1]
  y_temp_2 = df_temp_2.iloc[:,1]
  if len(df_temp_2)>1:
    # creating two interpolation functions:
    y_interpy_temp_1 = interp1d(x_temp_1, y_temp_1,kind=inter_kind)
    y_interpy_temp_2 = interp1d(x_temp_2, y_temp_2,kind=inter_kind)

    #creating x-arrays based on the number of x in the original array
    x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=limit_complete_period*period_info_intpl, endpoint=True)    
    x_origin_2=np.linspace(df_temp_2.iloc[0,0], df_temp_2.iloc[-1,0], num=len(sr_origin)-limit_complete_period*period_info_intpl+1, endpoint=True)    

    # perfoming the interpolation:
    y_interpolated_origin_1=y_interpy_temp_1(x_origin_1)
    y_interpolated_origin_2=y_interpy_temp_2(x_origin_2)

    # putting the two arrays into one another, we need to remove the overlapping period
    final_array=np.concatenate([y_interpolated_origin_1,y_interpolated_origin_2[1:]])

  elif len(df_temp_2)==1:

    y_interpy_temp_1 = scipy.interpolate.interp1d(x_temp_1, y_temp_1,kind=inter_kind)
    x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=limit_complete_period*period_info_intpl, endpoint=True)    
    y_interpolated_origin_1=y_interpy_temp_1(x_origin_1)
    final_array=y_interpolated_origin_1

  return final_array

from scipy import interpolate

def linear_interpolation_3 (sr_origin,sr_grouped,period_info_intpl,inter_kind):

  #df_origin=test#complete_pandemic_data['daily_admitted_H']
  sr_grouped=sr_grouped.reset_index()
  limit_complete_period=math.floor(len(sr_origin)/period_info_intpl)
  limit_complete_total=math.ceil(len(sr_origin)/period_info_intpl)

  # Creating two separate dataframe based on the completeness of the periods
  df_temp_1=sr_grouped.iloc[:limit_complete_period+1,]
  df_temp_2=sr_grouped.iloc[limit_complete_period:limit_complete_total+1,]

  # x and y of the data that interpolation will be based on
  x_temp_1 = df_temp_1.iloc[:,0]
  x_temp_2 = df_temp_2.iloc[:,0]
  y_temp_1 = df_temp_1.iloc[:,1]
  y_temp_2 = df_temp_2.iloc[:,1]

  if len(df_temp_2)>1:
    
    x_temp_2.iloc[1]=x_temp_2.iloc[0]+len(sr_origin[limit_complete_period*period_info_intpl:])/period_info_intpl

    x_adj=x_temp_1.append(x_temp_2.iloc[1:])
    y_adj=y_temp_1.append(y_temp_2.iloc[1:])

    x_adj[1:limit_complete_total]=x_adj[1:limit_complete_total]-0.5
    x_adj[limit_complete_total]=x_adj[limit_complete_total]-(len(sr_origin[limit_complete_period*period_info_intpl:])/period_info_intpl)/2

    x_origin_1=np.linspace(x_temp_1.iloc[0], x_temp_2.iloc[-2], num=limit_complete_period*period_info_intpl, endpoint=True)    
    x_origin_2=(np.linspace(x_temp_2.iloc[-2], x_adj.iloc[-1], num=len(sr_origin[limit_complete_period*period_info_intpl:])+1, endpoint=True))[1:]
    x_origin_new=np.concatenate([x_origin_1,x_origin_2])

    y_interpy_temp_1 = interp1d(x_adj, y_adj,kind=inter_kind)
    final_array=y_interpy_temp_1(x_origin_new)

  elif len(df_temp_2)==1:

    x_adj=x_temp_1
    x_adj[1:limit_complete_total]=x_temp_1[1:(limit_complete_total)]-0.5
    x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=limit_complete_period*period_info_intpl, endpoint=True)    
    y_interpy_temp_1 = interp1d(x_adj, y_temp_1,kind=inter_kind)
    final_array=y_interpy_temp_1(x_origin_1)

  return final_array

from scipy import interpolate

def linear_interpolation_4 (sr_origin,sr_grouped,period_info_intpl,inter_kind):

  #df_origin=test#complete_pandemic_data['daily_admitted_H']
  sr_grouped=sr_grouped.reset_index()
  limit_complete_period=math.floor(len(sr_origin)/period_info_intpl)
  limit_complete_total=math.ceil(len(sr_origin)/period_info_intpl)

  if limit_complete_period==0:
    df_temp_1=sr_grouped.iloc[0:limit_complete_total+1,]

    # x and y of the data that interpolation will be based on
    x_temp_1 = df_temp_1.iloc[:,0]
    y_temp_1 = df_temp_1.iloc[:,1]

    x_adj=x_temp_1
    x_adj[1:limit_complete_total]=x_temp_1[1:(limit_complete_total)]-0.5
    x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=len(sr_origin), endpoint=True)    
    y_interpy_temp_1 = interp1d(x_adj, y_temp_1,kind=inter_kind)
    final_array=y_interpy_temp_1(x_origin_1)
  else:

    # Creating two separate dataframe based on the completeness of the periods
    df_temp_1=sr_grouped.iloc[:limit_complete_period+1,]
    df_temp_2=sr_grouped.iloc[limit_complete_period:limit_complete_total+1,]

    # x and y of the data that interpolation will be based on
    x_temp_1 = df_temp_1.iloc[:,0]
    x_temp_2 = df_temp_2.iloc[:,0]
    y_temp_1 = df_temp_1.iloc[:,1]
    y_temp_2 = df_temp_2.iloc[:,1]

    if len(df_temp_2)>1:
      
      x_temp_2.iloc[1]=x_temp_2.iloc[0]+len(sr_origin[limit_complete_period*period_info_intpl:])/period_info_intpl

      x_adj=x_temp_1.append(x_temp_2.iloc[1:])
      y_adj=y_temp_1.append(y_temp_2.iloc[1:])

      x_adj[1:limit_complete_total]=x_adj[1:limit_complete_total]-0.5
      x_adj[limit_complete_total]=x_adj[limit_complete_total]-(len(sr_origin[limit_complete_period*period_info_intpl:])/period_info_intpl)/2

      x_origin_1=np.linspace(x_temp_1.iloc[0], x_temp_2.iloc[-2], num=limit_complete_period*period_info_intpl, endpoint=True)    
      x_origin_2=(np.linspace(x_temp_2.iloc[-2], x_adj.iloc[-1], num=len(sr_origin[limit_complete_period*period_info_intpl:])+1, endpoint=True))[1:]
      x_origin_new=np.concatenate([x_origin_1,x_origin_2])

      y_interpy_temp_1 = interp1d(x_adj, y_adj,kind=inter_kind)
      final_array=y_interpy_temp_1(x_origin_new)

    elif len(df_temp_2)==1:

      x_adj=x_temp_1
      x_adj[1:limit_complete_total]=x_temp_1[1:(limit_complete_total)]-0.5
      x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=limit_complete_period*period_info_intpl, endpoint=True)    
      y_interpy_temp_1 = interp1d(x_adj, y_temp_1,kind=inter_kind)
      final_array=y_interpy_temp_1(x_origin_1)

  return final_array

from scipy import interpolate

def linear_interpolation_6 (sr_origin,sr_grouped,period_info_intpl,inter_kind):


  #df_origin=test#complete_pandemic_data['daily_admitted_H']
  sr_grouped=sr_grouped.reset_index()
  limit_complete_period=math.floor(len(sr_origin)/period_info_intpl)
  limit_complete_total=math.ceil(len(sr_origin)/period_info_intpl)

  if limit_complete_period==0:
    df_temp_1=sr_grouped.iloc[0:limit_complete_total+1,]

    # x and y of the data that interpolation will be based on
    x_temp_1 = df_temp_1.iloc[:,0]
    y_temp_1 = df_temp_1.iloc[:,1]

    x_adj=x_temp_1
    x_adj[1:limit_complete_total]=x_temp_1[1:(limit_complete_total)]-0.5
    x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=len(sr_origin), endpoint=True)    
    y_interpy_temp_1 = interp1d(x_adj, y_temp_1,kind=inter_kind)
    final_array=y_interpy_temp_1(x_origin_1)
  else:

    # Creating two separate dataframe based on the completeness of the periods
    df_temp_1=sr_grouped.iloc[:limit_complete_period+1,]
    df_temp_2=sr_grouped.iloc[limit_complete_period:limit_complete_total+1,]

    # x and y of the data that interpolation will be based on
    x_temp_1 = df_temp_1.iloc[:,0]
    x_temp_2 = df_temp_2.iloc[:,0]
    y_temp_1 = df_temp_1.iloc[:,1]
    y_temp_2 = df_temp_2.iloc[:,1]

    if len(df_temp_2)>1:
      
      x_temp_2.iloc[1]=x_temp_2.iloc[0]+len(sr_origin[limit_complete_period*period_info_intpl:])/period_info_intpl

      x_adj=x_temp_1.copy()
      x_adj[1:limit_complete_total]=(x_temp_1[1:limit_complete_total]-0.5)
      x_adj=x_adj.append(x_temp_2,ignore_index=True)
      y_adj=y_temp_1.copy()
      y_adj=y_adj.append(y_temp_2,ignore_index=True)
      y_adj.iloc[-2]=y_temp_2.iloc[1]

      x_origin_1=np.linspace(x_temp_1.iloc[0], x_temp_2.iloc[-2], num=limit_complete_period*period_info_intpl, endpoint=True)    
      x_origin_2=(np.linspace(x_temp_2.iloc[-2], x_adj.iloc[-1], num=len(sr_origin[limit_complete_period*period_info_intpl:])+1, endpoint=True))[1:]
      x_origin_new=np.concatenate([x_origin_1,x_origin_2])

      y_interpy_temp_1 = interp1d(x_adj, y_adj,kind=inter_kind)
      final_array=y_interpy_temp_1(x_origin_new)

    elif len(df_temp_2)==1:

      x_adj=x_temp_1.copy()
      #dividing all period_info in half so that the interpolation graph would pass through them all
      x_adj[1:limit_complete_total]=x_temp_1[1:(limit_complete_total)]-0.5
      # adding one more data for the last value so that it would better represent the data
      x_adj[limit_complete_total]=x_temp_1[limit_complete_total-1]
      x_adj[limit_complete_total+1]=x_temp_1[limit_complete_total]
      x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=limit_complete_period*period_info_intpl, endpoint=True)    
      y_temp_1[limit_complete_total]=y_temp_1[limit_complete_total]
      y_temp_1[limit_complete_total+1]=y_temp_1[limit_complete_total]
      y_interpy_temp_1 = interp1d(x_adj, y_temp_1,kind=inter_kind)
      final_array=y_interpy_temp_1(x_origin_1)

  return final_array

"""### Replenishement function"""

def replenished_func_variable_period_info(y_to_rep,rep_no):
  y_temp=y_to_rep.to_frame()
  data_for_grouping=(y_temp.reset_index()).rename(columns={'index':'date'})
  temp_index_grouped_origin=(variable_period_info_df.iloc[rep_no,:].dropna()).astype(int)
  data_for_grouping['grouping_index']=data_for_grouping['daily_consumption']
  data_for_grouping['grouping_index']=np.nan

  for i in range(1,len(temp_index_grouped_origin)):
    data_for_grouping['grouping_index'][temp_index_grouped_origin[0:i].sum():temp_index_grouped_origin[0:i+1].sum()]=i-1#temp_index_grouped[i]

  df_grouped=(data_for_grouping.groupby(data_for_grouping['grouping_index'],sort=False).sum()).div(temp_index_grouped_origin.values[1:],axis=0)#/len(data_for_grouping)

  daily_grouped_data=data_for_grouping.set_index('grouping_index')
  daily_grouped_data['daily_consumption']=df_grouped.iloc[:,0]
  y_rep=daily_grouped_data.set_index('date').dropna()
  return y_rep['daily_consumption']

def replenishment_func (data_for_grouping,period_info_grouped):
  # summing the data over 7 day interval, starting with day 1, this part will discard whcih day we will start from 
  # either monday or any other day,

  #creating a separate dataframe for a weekly pandemic data with correct days as its index
  df_grouped_temp=data_for_grouping.copy()
  df_grouped_temp=df_grouped_temp.to_frame()
  #reseting the index and and rename it as the date column
  df_grouped=df_grouped_temp.reset_index().rename(columns = {'index':'date'}, inplace = False)
  # Formating the value of date into date format
  df_grouped['date'] = pd.to_datetime(df_grouped['date'],format='%Y-%m-%d')
  # groupby the data frame to get the seven day sum
  # and divide everything by 7 since we want the average 
  df_grouped=(df_grouped.groupby(df_grouped.index // period_info_grouped).sum())/period_info_grouped

  # adding a row of zero for time zero
  df_grouped.loc[len(df_grouped)] = 0
  df_grouped = round(df_grouped.shift()).apply(np.int64)
  df_grouped.loc[0] = 0

  # sending the grouped data into the lineat interpolation function and placing it back into original dataframe 
  # so that the index would be the same
  df_grouped_temp.iloc[:,0]=linear_interpolation(data_for_grouping,df_grouped.iloc[:,0],period_info_grouped)
 
  # sending the data back with the seires format type
  return df_grouped_temp.iloc[:,0]

# this function only grouped the data based on the specified period info
def replenishment_func_2 (data_for_grouping,period_info_grouped):
  # summing the data over 7 day interval, starting with day 1, this part will discard whcih day we will start from 
  # either monday or any other day,

  #creating a separate dataframe for a weekly pandemic data with correct days as its index
  df_grouped_temp=data_for_grouping.copy()
  df_grouped_temp=df_grouped_temp.to_frame()
  #reseting the index and and rename it as the date column
  df_grouped=df_grouped_temp.reset_index().rename(columns = {'index':'date'}, inplace = False)
  # Formating the value of date into date format
  df_grouped['date'] = pd.to_datetime(df_grouped['date'],format='%Y-%m-%d')
  # groupby the data frame to get the seven day sum
  # and divide everything by 7 since we want the average 
  df_grouped=(df_grouped.groupby(df_grouped.index // period_info_grouped).sum())

  # adding a row of zero for time zero
  df_grouped.loc[len(df_grouped)] = 0
  df_grouped = round(df_grouped.shift()).apply(np.int64)
  df_grouped.loc[0] = 0
 
  # sending the data back with the seires format type
  return df_grouped.iloc[:,0]

from scipy import interpolate

def replenishment_func_3 (data_for_grouping,period_info_grouped):

  # assiging the dataframes
  #df_origin=test#complete_pandemic_data['daily_admitted_H']
  #sr_grouped=sr_grouped.reset_index()
  limit_complete_period=math.floor(len(data_for_grouping)/period_info_grouped)
  limit_complete_total=math.ceil(len(data_for_grouping)/period_info_grouped)
  if len(data_for_grouping)/limit_complete_period>1:

    df_temp_1=data_for_grouping.iloc[:limit_complete_period*period_info_grouped,]
    df_temp_2=data_for_grouping.iloc[limit_complete_period*period_info_grouped:limit_complete_total*period_info_grouped+1,]

    df_grouped_1=(df_temp_1.groupby(df_temp_1.index // period_info_grouped).sum())/period_info_grouped
    df_grouped_2=(df_temp_2.groupby(df_temp_2.index // period_info_grouped).sum())/len(df_temp_2)
    df_grouped=df_grouped_1.append(df_grouped_2)
  elif (len(data_for_grouping)/limit_complete_period)==1:
    df_grouped=(df_grouped.groupby(df_grouped.index // period_info_grouped).sum())/period_info_grouped

  # adding a row of zero for time zero
  df_grouped.loc[len(df_grouped)] = 0
  df_grouped = (df_grouped.shift())#.apply(np.int64)
  df_grouped.loc[0] = 0
 
  # sending the data back with the seires format type
  return df_grouped

# this function only grouped the data based on the specified period info
def replenishment_func_4 (data_for_grouping,period_info_grouped):
  # summing the data over 7 day interval, starting with day 1, this part will discard whcih day we will start from 
  # either monday or any other day,

  #creating a separate dataframe for a weekly pandemic data with correct days as its index
  df_grouped_temp=data_for_grouping.copy()
  #df_grouped_temp=df_grouped_temp.to_frame()
  #reseting the index and and rename it as the date column
  df_grouped=df_grouped_temp.reset_index().rename(columns = {'index':'date'}, inplace = False)
  # Formating the value of date into date format
  df_grouped['date'] = pd.to_datetime(df_grouped['date'],format='%Y-%m-%d')
  # groupby the data frame to get the seven day sum
  # and divide everything by 7 since we want the average 
  df_grouped=(df_grouped.groupby(df_grouped.index // period_info_grouped).sum())

  # adding a row of zero for time zero
  df_grouped.loc[len(df_grouped)] = 0
  df_grouped = round(df_grouped.shift()).apply(np.int64)
  df_grouped.loc[0] = 0
 
  # sending the data back with the seires format type
  return df_grouped.iloc[:,0]

def replenishment_func_5 (data_for_grouping,period_info_grouped):

  # assiging the dataframes
  #df_origin=test#complete_pandemic_data['daily_admitted_H']
  #sr_grouped=sr_grouped.reset_index()
  limit_complete_period=math.floor(len(data_for_grouping)/period_info_grouped)
  limit_complete_total=math.ceil(len(data_for_grouping)/period_info_grouped)
  if limit_complete_period==0:

    df_grouped=(data_for_grouping.groupby(data_for_grouping.index // len(data_for_grouping)).sum())/len(data_for_grouping)
  else:
    if len(data_for_grouping)/limit_complete_period>1:

      df_temp_1=data_for_grouping.iloc[:limit_complete_period*period_info_grouped,]
      df_temp_2=data_for_grouping.iloc[limit_complete_period*period_info_grouped:limit_complete_total*period_info_grouped+1,]

      df_grouped_1=(df_temp_1.groupby(df_temp_1.index // period_info_grouped).sum())/period_info_grouped
      df_grouped_2=(df_temp_2.groupby(df_temp_2.index // period_info_grouped).sum())/len(df_temp_2)
      df_grouped=df_grouped_1.append(df_grouped_2)
    elif (len(data_for_grouping)/limit_complete_period)==1:
      df_grouped=(data_for_grouping.groupby(data_for_grouping.index // period_info_grouped).sum())/period_info_grouped

  # adding a row of zero for time zero
  df_grouped.loc[len(df_grouped)] = 0
  df_grouped = (df_grouped.shift())#.apply(np.int64)
  df_grouped.loc[0] = 0
 
  # sending the data back with the seires format type
  return df_grouped

def replenishment_func_6 (data_for_grouping,period_info_grouped):

  # assiging the dataframes
  #df_origin=test#complete_pandemic_data['daily_admitted_H']
  #sr_grouped=sr_grouped.reset_index()


  #creating a separate dataframe for a weekly pandemic data with correct days as its index
  #df_grouped_temp=data_for_grouping.copy()
  data_for_grouping=data_for_grouping.to_frame()
  #reseting the index and and rename it as the date column
  data_for_grouping=data_for_grouping.reset_index().rename(columns = {'index':'date'}, inplace = False)
  # Formating the value of date into date format
  data_for_grouping['date'] = pd.to_datetime(data_for_grouping['date'],format='%Y-%m-%d')

  limit_complete_period=math.floor(len(data_for_grouping)/period_info_grouped)
  limit_complete_total=math.ceil(len(data_for_grouping)/period_info_grouped)
  if limit_complete_period==0:

    df_grouped=(data_for_grouping.groupby(data_for_grouping.index // len(data_for_grouping)).sum())/len(data_for_grouping)
  else:
    if len(data_for_grouping)/limit_complete_period>1:

      df_temp_1=data_for_grouping.iloc[:limit_complete_period*period_info_grouped,]
      df_temp_2=data_for_grouping.iloc[limit_complete_period*period_info_grouped:limit_complete_total*period_info_grouped+1,]

      df_grouped_1=(df_temp_1.groupby(df_temp_1.index // period_info_grouped).sum())/period_info_grouped
      df_grouped_2=(df_temp_2.groupby(df_temp_2.index // period_info_grouped).sum())/len(df_temp_2)
      df_grouped=df_grouped_1.append(df_grouped_2)
    elif (len(data_for_grouping)/limit_complete_period)==1:
      df_grouped=(data_for_grouping.groupby(data_for_grouping.index // period_info_grouped).sum())/period_info_grouped

  # adding a row of zero for time zero
  df_grouped.loc[len(df_grouped)] = 0
  df_grouped = (df_grouped.shift())#.apply(np.int64)
  df_grouped.loc[0] = 0
 
  # sending the data back with the seires format type
  return df_grouped.iloc[:,0]

"""### Versus graph presentation function"""

def graph_generator_for_epochs(data1,data2,data3,y_axis_name,data_type):
  fig = plt.figure(figsize=(15,8))
  ax = fig.add_subplot(111)

  #taking the mean of each column and removing the epochs that order cannot be placed (based on the lead time)
  y_bias_SIR=data1.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Holt=data2.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Naive=data3.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  x=range(len(y_bias_SIR))

  #spine placement data centered
  ax.spines['left'].set_position(('data', 0.0))
  ax.spines['bottom'].set_position(('data', 0.0))
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')

  plt.plot(x,y_bias_SIR, color = 'blue',label='SEIRHD',marker='^')
  plt.plot(x,y_bias_Holt, color = 'green',label='Holt',marker='^')
  plt.plot(x,y_bias_Naive, color = 'red',label='Naive',marker='^')



  plt.legend(['SEIRHD','Holt','Naive'])

  plt.xlabel('Epoch',fontsize=16)
  plt.ylabel(y_axis_name,fontsize=16)
  #plt.title(title,fontsize=20)
  #plt.savefig('%s%% data,%s '%(data_type,y_axis_name))

  return

def graph_generator_for_epochs_2(data1,data2,data3,y_axis_name,data_type,title):
  fig = plt.figure(figsize=(15,8))
  ax = fig.add_subplot(111)

  #taking the mean of each column and removing the epochs that order cannot be placed (based on the lead time)
  y_bias_SIR=data1.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Holt=data2.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Naive=data3.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  x=range(len(y_bias_SIR))

  #spine placement data centered
  ax.spines['left'].set_position(('data', 0.0))
  ax.spines['bottom'].set_position(('data', 0.0))
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')

  plt.plot(x,y_bias_SIR, color = 'blue',label='SEIRHD',marker='^')
  plt.plot(x,y_bias_Holt, color = 'green',label='Holt',marker='^')
  plt.plot(x,y_bias_Naive, color = 'red',label='Naive',marker='^')



  plt.legend(['SEIRHD','Holt','Naive'])

  plt.xlabel('Epoch',fontsize=16)
  plt.ylabel(y_axis_name,fontsize=16)
  plt.title(title,fontsize=20)
  #plt.savefig('%s%% data,%s '%(data_type,y_axis_name))

  return

def graph_generator_for_epochs_title(data1,data2,data3,y_axis_name,name1,name2,name3,title):
  fig = plt.figure(figsize=(15,8))
  ax = fig.add_subplot(111)

  #taking the mean of each column and removing the epochs that order cannot be placed (based on the lead time)
  y_bias_SIR=data1.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Holt=data2.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Naive=data3.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  x=range(len(y_bias_SIR))

  #spine placement data centered
  ax.spines['left'].set_position(('data', 0.0))
  ax.spines['bottom'].set_position(('data', 0.0))
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')

  plt.plot(x,y_bias_SIR, color = 'blue',label=name1,marker='^')
  plt.plot(x,y_bias_Holt, color = 'green',label=name2,marker='^')
  plt.plot(x,y_bias_Naive, color = 'red',label=name3,marker='^')



  plt.legend([name1,name2,name3])

  plt.xlabel('Epoch',fontsize=16)
  plt.ylabel(y_axis_name,fontsize=16)

  plt.title(title,fontsize=16)

  #plt.savefig('graph of %s'%(title),dpi=300)

  return


#plt.savefig('Relative error of epochs for each method',dpi=300)

def graph_generator_for_epochs_title_2(data1,data2,data3,data4,data5,y_axis_name,name1,name2,name3,name4,name5,title):
  fig = plt.figure(figsize=(15,8))
  ax = fig.add_subplot(111)

  #taking the mean of each column and removing the epochs that order cannot be placed (based on the lead time)
  y_1=data1.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_2=data2.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_3=data3.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_4=data4.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_5=data5.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  x=range(len(y_1))

  #spine placement data centered
  ax.spines['left'].set_position(('data', 0.0))
  ax.spines['bottom'].set_position(('data', 0.0))
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')

  plt.plot(x,y_1, color = 'darkkhaki',label=name1,marker='^')
  plt.plot(x,y_2, color = 'aqua',label=name2,marker='^')
  plt.plot(x,y_3, color = 'blue',label=name3,marker='^')
  plt.plot(x,y_4, color = 'green',label=name4,marker='^')
  plt.plot(x,y_5, color = 'red',label=name5,marker='^')

  plt.legend([name1,name2,name3,name4,name5])

  plt.xlabel('Epoch',fontsize=16)
  plt.ylabel(y_axis_name,fontsize=16)

  plt.title(title,fontsize=16)

  #plt.savefig('graph of %s'%(title),dpi=300)

  return


#plt.savefig('Relative error of epochs for each method',dpi=300)

def df_maker (data_1,data_2,data_3):
  data_SIR=data_1
  data_SIR.columns = data_SIR.columns.str.replace(r'^SIR', '')

  data_Holt=data_2
  data_Holt.columns = data_Holt.columns.str.replace(r'^Holt', '')

  data_Naive=data_3
  data_Naive.columns = data_Naive.columns.str.replace(r'^Naive', '')

  df_data=data_SIR.append([data_Holt,data_Naive])
  df_data['index']=['SIR','Holt','Naive']

  df_data=df_data.set_index(['index'])
  
  return df_data

!python --version

"""# SIR Model

## Derivate function
"""

#NOTE: set a random seed
# MODEL FUNCTION
#===============================================================================================================================================

#This is the derivate function. The function returns the change in any compartment at any given time that is defined for it
#y is the total population of any compartment
    #NOTE: by compartment we mean each section of the model (e.g. S for susceptible, is the total population of this compartment at any time t),
    #      the model can have many compartment as long as the related parameters are being defined for it
    
def deriv(y, t, N, beta, rate_OTH, rate_H): 
      
    # rate_OTH : rate at which people leaving the infection 'I' compartment to the outside of hospital 'O' compartment
    # rate_H   : rate at which people leaving the infection 'I' compartment to the hospital 'H' compartment
    # rho_H    : rate at which people leaving the hospitals 'H' compartment to the dead at hospital 'DH' compartment
   
    S, E, I, O, H, ReDH = y 

    dSdt = -beta(t) * S * I / N
    dEdt = beta(t) * S * I / N - sigma * E    
    dIdt = sigma * E - prob_H * rate_H * I - (1-prob_H) * rate_OTH * I    
    dOdt = (1-prob_H)* rate_OTH * I    
    dHdt = prob_H * rate_H * I - gamma_D * H   
    dReDHdt = gamma_D * H
    
    return dSdt, dEdt, dIdt, dOdt, dHdt, dReDHdt

#The logsitic function smooth the trnasition of R0 over time
#def logistic_R_0(t):
#    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end

#The Model function returns the array of population in any compartment 
def Model(day_start,day_end, total_population, R_0_start, x0,rate_OTH, rate_H, starting_data=[],k=None,R_0_end=None):

    if k is None:
      k=k_constant
    if R_0_end is None:
      R_0_end=R_0_end_constant
    #start=0
    #The logsitic function smooth the trnasition of R0 over time
    def logistic_R_0(t):
        return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end
    
    #Beta function returns the value of beta at any given time, Beta is not a constant number as it is dependant on the value of R0
    def beta(t):
        return logistic_R_0(t) * (prob_H * rate_H + (1-prob_H) * rate_OTH)
    
    N=total_population
    
    #Initial values of each compartment should be specfied manually, every single compartment should have an initial value
    S0, E0, I0, O0, H0, ReDH0, I_cumulative0, EtoI0, ItoH0, HtoReDH = starting_data  # initial conditions: one exposed
    y0 = S0, E0, I0, O0, H0, ReDH0 # Initial conditions vector
    #t, S, E, I, O, H, ReH, DH, I_cumulative, EtoI, ItoH, HtoReH, HtoDH 
    #creating an array of time for integral function to perform integration process based on the interval we specifies here
    t = np.linspace(day_start, day_end-1, (day_end-day_start)) # Grid of time points (in days)

    #Integrating the SIR equations over the time grid, t. The function returns an array of values that is the total population of each compartment
    ret = odeint(deriv, y0, t, args=(N, beta, rate_OTH, rate_H))
    
    #Assigning the results of intergration to its respective variable
    S, E, I, O, H, ReDH = ret.T
    
    #parameter will be used to demonstrate the transition in R0 over time
    #R0_over_time = [logistic_R_0(i) for i in range(len(t))]  # to plot R_0 over time: get function values
    
    #daily infection
    EtoI = sigma * E
    #daily admission to hospital
    ItoH = prob_H * rate_H * I
    #daily recovered from hospital
    HtoReDH = gamma_D * H
  
    #Cumulative number of cases
    #============================================================================================================================================

    I_cumulative = np.cumsum(EtoI)+I_cumulative0

    #============================================================================================================================================
    
    return t, S, E, I, O, H, ReDH, I_cumulative, EtoI, ItoH, HtoReDH

"""## Global Fitting function"""

# MODEL FITTING FUNCTION, Multi - Objective Version
#=======================================================================================================================================

# this model receives two sets of data and their counterpart in the model in the form of compartment number
# "compartment_train1" is the number of compartment within the model whcich will be used to fit the curve on the "training_data1"
def model_fitting_function_multi(training_data1_cumulative,training_data1, training_data2,training_data3,training_data4, outbreak_shift, 
                                 compartment_array, initial_parameter, day_start,day_end,k_inclusion,brute_step,graphs):

    #DATA SEGMENTATION
    #==========================================================================================================
    #Creating arrays of real data based on the given interval of start and end day
    #training_data1_cumulative = training_data1_cumulative [day_start:day_end]*1000/max(training_data1_cumulative [day_start:day_end])
    #training_data1 = training_data1 [day_start:day_end]*1000/max(training_data1 [day_start:day_end])
    #training_data2 = training_data2 [day_start:day_end]*1000/max(training_data2 [day_start:day_end])
    #training_data3 = training_data3 [day_start:day_end]*1000/max(training_data3 [day_start:day_end])
    #training_data4 = training_data4 [day_start:day_end]*1000/max(training_data4 [day_start:day_end])

    training_data1_cumulative = training_data1_cumulative [day_start:day_end]#*1000/max(training_data1_cumulative [day_start:day_end])
    training_data1 = training_data1 [day_start:day_end]#*1000/max(training_data1 [day_start:day_end])
    training_data2 = training_data2 [day_start:day_end]#*1000/max(training_data2 [day_start:day_end])
    training_data3 = training_data3 [day_start:day_end]#*1000/max(training_data3 [day_start:day_end])
    training_data4 = training_data4 [day_start:day_end]#*1000/max(training_data4 [day_start:day_end])



    # the data will be feed into the "data_adjust" function
    if outbreak_shift >= 0:
        adjusted_data1_cumulative = np.concatenate((np.zeros(outbreak_shift), training_data1_cumulative))
        adjusted_data1 = np.concatenate((np.zeros(outbreak_shift), training_data1))
        adjusted_data2 = np.concatenate((np.zeros(outbreak_shift), training_data2))
        adjusted_data3 = np.concatenate((np.zeros(outbreak_shift), training_data3))
        adjusted_data4 = np.concatenate((np.zeros(outbreak_shift), training_data4))
    else:
        adjusted_data1_cumulative = adjusted_data1_cumulative[-outbreak_shift:]
        adjusted_data1 = adjusted_data1[-outbreak_shift:]
        adjusted_data2 = adjusted_data2[-outbreak_shift:]
        adjusted_data3 = adjusted_data3[-outbreak_shift:]
        adjusted_data4 = adjusted_data4[-outbreak_shift:]
    
    #adjusted_data1 = data_adjust(train_data=training_data1, shift=outbreak_shift)
    #adjusted_data2 = data_adjust(train_data=training_data2, shift=outbreak_shift)
    #you need a check here to make sure they have the same length
    

    #DATA SELECTION
    #==========================================================================================================
    # data will transformed into an array format to be inserted into "multi_fit_objective" function
    y_data = []

    #depending on how many real data needs to be used for curve fitting, we need to find the count:
    compartment_count,=np.array(compartment_array).shape

    #length of the array of real data which the curve fitting will be built on
    data_length=min(len(adjusted_data1),len(adjusted_data1_cumulative),len(adjusted_data2),len(adjusted_data3),len(adjusted_data4))

    #creating an array of zeros to fit the real data into 
    y_data=np.zeros((compartment_count,data_length))

    #this "for" loop puts the correct data into the array with the correct sequence based on the requested compartment for fitting:
    for i in range (compartment_count):
      #compartment 8 corresponds to the cumulative number of daily infection
      if compartment_array[i]==7:
        y_data[i,:] = y_data[i,:] + adjusted_data1_cumulative
      #compartment 9 corresponds to the number of daily infection
      elif compartment_array[i]==8:
        y_data[i] = y_data[i,:] + adjusted_data1
      #compartment 10 corresponds to the number of daily adimission to hospitals
      elif compartment_array[i]==9:
        y_data[i] = y_data[i,:] + adjusted_data2
      #compartment 11 corresponds to the number of daily discharge from hospitals
      elif compartment_array[i]==10:
        y_data[i] = y_data[i,:] + adjusted_data3
      #compartment 12 corresponds to the number of hospitalization (total number of patients at hospitals at any given time)
      elif compartment_array[i]==5:
        y_data[i] = y_data[i,:] + adjusted_data4

    y_data = np.array(y_data)
    
    # x_value is defined
    x_value = np.linspace(day_start, day_end-1, (day_end - day_start))


    #PARAMETER SELECTION
    #==========================================================================================================
    # checking if the k is being estimated for curve fitting purposes
    # optimization function to minimize the residuals of the different data set
    if k_inclusion==False:
        fitter = Minimizer(multi_fit_objective, initial_parameter, 
                          fcn_args=(y_data, compartment_array))
        result = fitter.minimize(method='brute', Ns=brute_step, keep=brute_step)

        #using the best initial guess found in the brute method and applying the least squared method on it:
        best_result = copy.deepcopy(result)
        for candidate in result.candidates:
            trial = fitter.minimize(method='leastsq', params=candidate.params)
            if trial.chisqr < best_result.chisqr:
                best_result = trial

        # the result is inserted into "estimate_data" function to get the estimated data for each compartment of the model
        estimited_data_result = estimate_data (model_fit_param = best_result.params)
    else:
        fitter = Minimizer(multi_fit_objective_k, initial_parameter, 
                          fcn_args=(y_data, compartment_array))
        result = fitter.minimize(method='brute', Ns=brute_step, keep=brute_step)

        #using the best initial guess found in the brute method and applying the least squared method on it:
        best_result = copy.deepcopy(result)
        for candidate in result.candidates:
            trial = fitter.minimize(method='leastsq', params=candidate.params)
            if trial.chisqr < best_result.chisqr:
                best_result = trial
                
        # the result is inserted into "estimate_data" function to get the estimated data for each compartment of the model
        estimited_data_result = estimate_data_k (model_fit_param = best_result.params)

    if graphs=='yes':
      #DATA ESTIMATION AND PLOTTING
      #==========================================================================================================
      # the result is inserted into "estimate_data" function to get the estimated data for each compartment of the model
      #estimited_data_result = estimate_data (model_fit_param = best_result.params)
      
      # plotting the graph estimated data vs. real data
      plot_curvefit_result(x_value=x_value, y_pred=estimited_data_result[7], y_true=adjusted_data1_cumulative, title='Cumulative cases of infected',yaxistitle='cumulative number of cases')
      plot_curvefit_result(x_value=x_value, y_pred=estimited_data_result[8], y_true=adjusted_data1, title='Number of daily infection',yaxistitle='daily number of cases')
      plot_curvefit_result(x_value=x_value, y_pred=estimited_data_result[9],
                          y_true=adjusted_data2, title='Daily admission to hospitals',yaxistitle='number of cases' )
      plot_curvefit_result(x_value=x_value, y_pred=estimited_data_result[10],
                          y_true=adjusted_data3, title='Daily discharge (dead or alive) from hospitals',yaxistitle='number of cases' )
      plot_curvefit_result(x_value=x_value, y_pred=estimited_data_result[5],
                          y_true=adjusted_data4, title='COVID-19 hospitalization',yaxistitle='Hospitalization' )

    return estimited_data_result,best_result

"""### Residual fitting functions:"""

# Objective function for multi-fitting curve function
#===============================================================================================================================================

#The structure of this section of the code has been retrieved from : 
#https://lmfit.github.io/lmfit-py/examples/example_fit_multi_datasets.html#sphx-glr-download-examples-example-fit-multi-datasets-py

def multi_fit_objective (initial_param, y_data, compartment_array):
   
    # this will give the dimension of the array (y_data)
    n_y_data, _ = y_data.shape
    
    #setting all element to zero
    resididuals = 0.0*y_data[:]
    
    estimated_data = estimate_data (model_fit_param=initial_param)

    # make residual per data set
    for i in range(n_y_data):
        
        # "y_data[i, :]" : the representation of the actual data 
        # "estimate_data (model...)" : the estimated data based on the paramters and the desired compartmental set in the model
        resididuals[i, :] = y_data[i, :] - estimated_data[compartment_array[i]]

    #resididuals[0, :] = ((abs(y_data[0, :] - estimated_data[compartment_array[0]])/(max(max(y_data[0, :]),max(estimated_data[compartment_array[0]]))))/2)*100000
    #resididuals[1, :] = ((abs(y_data[1, :] - estimated_data[compartment_array[1]])/(max(max(y_data[1, :]),max(estimated_data[compartment_array[1]]))))/2)*100000
    #resididuals[2, :] = (abs(y_data[2, :] - estimated_data[compartment_array[2]])/(max(max(y_data[2, :]),max(estimated_data[compartment_array[2]]))))*100000


        #resididuals[i, :] = resididuals[i, :]/max(resididuals[i, :])
    # now flatten this to a 1D array, as minimize() needs
    return resididuals.flatten()

# Objective function for multi-fitting curve function
#===============================================================================================================================================

#The structure of this section of the code has been retrieved from : 
#https://lmfit.github.io/lmfit-py/examples/example_fit_multi_datasets.html#sphx-glr-download-examples-example-fit-multi-datasets-py

def multi_fit_objective_k (initial_param, y_data, compartment_array):
   
    # this will give the dimension of the array (y_data)
    n_y_data, _ = y_data.shape
    
    #setting all element to zero
    resididuals = 0.0*y_data[:]
    
    estimated_data = estimate_data_k (model_fit_param=initial_param)

    # make residual per data set
    for i in range(n_y_data):
        
        # "y_data[i, :]" : the representation of the actual data 
        # "estimate_data (model...)" : the estimated data based on the paramters and the desired compartmental set in the model
        resididuals[i, :] = y_data[i, :] - estimated_data[compartment_array[i]]

    #resididuals[0, :] = ((abs(y_data[0, :] - estimated_data[compartment_array[0]])/(max(max(y_data[0, :]),max(estimated_data[compartment_array[0]]))))/2)*100000
    #resididuals[1, :] = ((abs(y_data[1, :] - estimated_data[compartment_array[1]])/(max(max(y_data[1, :]),max(estimated_data[compartment_array[1]]))))/2)*100000
    #resididuals[2, :] = (abs(y_data[2, :] - estimated_data[compartment_array[2]])/(max(max(y_data[2, :]),max(estimated_data[compartment_array[2]]))))*100000


        #resididuals[i, :] = resididuals[i, :]/max(resididuals[i, :])
    # now flatten this to a 1D array, as minimize() needs
    return resididuals.flatten()

"""### Accuracy function:"""

# ACCURACY FUNCTION
#===============================================================================================================================================

#Creating a function to check the accuracy of estimates
def Accuracy_check(y_true,y_pred):
    
    #Mean Squared Error
    err_MSE2 = np.mean((y_true - y_pred)**2)
    
    #Mean Absolute Error
    err_MAE2 = np.mean(np.abs(y_true - y_pred))
    
    #Symmetric Mean Absolute Percentage Error
    #(https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
    #First Version
    #err_SMAPE2 = np.mean(np.abs(y_true - y_pred)/((np.abs(y_true)+np.abs(y_pred))/2))*100
    #third Version
    err_SMAPE2 = sum(np.abs(y_true - y_pred))/sum(y_true+y_pred)
    
    #Mean Absolute Scaled Error
    MASE_data2=np.zeros(len(y_true)-1)
    for i in range(1,len(y_true)):
        MASE_data2[(i-1)]=np.abs(y_pred[i]-y_pred[(i-1)])
    err_MASE2 = np.mean(MASE_data2)/((len(y_true))-1)

    error_result2= [err_MAE2,err_MSE2,err_SMAPE2,err_MASE2]
    
    return error_result2

"""### Plotting function:"""

# PLOTTING FUNCTION
#===============================================================================================================================================

# Plot the real data vs the fitted data
#the ratio is the number which the data will be divided by
def plot_curvefit_result(x_value,y_pred,y_true,title,yaxistitle):


    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    #fig.subplots_adjust(bottom=0.15, left=0.2)
    #fig = plt.figure(facecolor='w')
    #ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    
    #Normalizing the values of the real and fitted data for better representation
    ax.plot(x_value, y_pred/(max(np.max(y_pred),np.max(y_true))), 'b', alpha=0.5, lw=2, label='Fitted data')
    ax.plot(x_value, y_true/(max(np.max(y_pred),np.max(y_true))), 'r', alpha=0.5, lw=2, label='Real data')
    
    #Normalizing the values of the real and fitted data for better representation
    #ax.plot(x_value, y_pred, 'b', alpha=0.5, lw=2, label='Fitted data')
    #ax.plot(x_value, y_true, 'r', alpha=0.5, lw=2, label='Real data')

    ax.set_xlabel('Time (days)',fontsize=12)
    ax.set_ylabel('Normalized '+ yaxistitle,fontsize=12)
    
    ax.set_ylim(0,1.2)
    #ax.yaxis.set_tick_params()
    #ax.xaxis.set_tick_params(True)
    #ax.grid(b=True, which='major', c='w', lw=2, ls='dotted')
    ax.legend(fontsize=13)

    ax.set_title(title,fontsize = 16)
    plt.savefig(title,dpi=300)
    plt.show();

"""### Data simulation functions:"""

# ESTIMATED DATA FUNCTION (with k being fixed)
#===============================================================================================================================================

# Creating a function that recieves the parameters for the model, 
# Returing a matrix of result for each compartment based on the best values of parameters that model estimated

#model_fit_param=initial_param,day_start=day_start,day_end=day_end,starting_data=starting_data

def estimate_data (model_fit_param):
    
    model_fit_data = Model(day_start = day_start,
                           day_end   = day_end,
                           total_population = total_population, 
                           R_0_start = model_fit_param["R_0_start"], 
                           x0        = model_fit_param["x0"], 
                           R_0_end   = model_fit_param["R_0_end"],
                           rate_OTH  = model_fit_param["rate_OTH"],
                           rate_H    = model_fit_param["rate_H"],
                           starting_data = starting_data)
    return model_fit_data
  #k         = model_fit_param["k"],

# ESTIMATED DATA FUNCTION (with k being estimated)
#===============================================================================================================================================

# Creating a function that recieves the parameters for the model, 
# Returing a matrix of result for each compartment based on the best values of parameters that model estimated

#model_fit_param=initial_param,day_start=day_start,day_end=day_end,starting_data=starting_data

def estimate_data_k (model_fit_param):
    
    model_fit_data = Model(day_start = day_start,
                           day_end   = day_end,
                           total_population = total_population, 
                           R_0_start = model_fit_param["R_0_start"], 
                           x0        = model_fit_param["x0"], 
                           k         = model_fit_param["k"],
                           R_0_end   = model_fit_param["R_0_end"],
                           rate_OTH  = model_fit_param["rate_OTH"],
                           rate_H    = model_fit_param["rate_H"],
                           starting_data = starting_data)
    return model_fit_data
  #k         = model_fit_param["k"],

provincial_data.shape

daily_recovered_H.shape

plt.plot(daily_cases_cumulative)

"""## Forecast:

### Data entry and results

#### Daily
"""

# SECTION 1
#========================================================================================================================
#fitting the curve on the first 120 days of pandemic

#Creating initial guesses and upper and lower bound for the parameters that model needs to estimate
#first 50 days
params_init_min_max = Parameters()
params_init_min_max.add("R_0_start",   min = 1.5,    max = 10   )
params_init_min_max.add("x0",          min = 50,      max = 120    )
#params_init_min_max.add("k",            min = 0.1,   max = 5  )
params_init_min_max.add("R_0_end",     min = 0.3,    max = 10     )
#params_init_min_max.add("alpha",       min = 0.003,  max = 0.05   )
params_init_min_max.add("rate_OTH",         min = 0.1,    max = 0.7    )
params_init_min_max.add("rate_H",         min = 0.1,    max = 0.7    )
#params_init_min_max.add("rho_H",         min = 0.01,    max = 0.7    )

#Fixed parameters
k_constant=1.0
#rho=0.13
sigma=1/5.1
gamma_D=1/12.1689
alpha_H = 0.14876
prob_H = 0.188885

# setting the initial values of each compartment
start_point_1 = [population_Region,1,0,0,0,0,0,0,0,0]
starting_data = start_point_1

# setting up the date on which the curve fitting function will be perfomed
Start_day_training = 0
end_day_training = (16*7)
start_day_estimation = end_day_training

#This is the array of number of each compartment that needs to be fitted on the real data
# 7  : Cumulative number of cases
# 8  : Daily number of cases 
# 9  : Daily admission to hospital
# 10 : Daily number of discharge at hospitals
# 5  : Daily Number of hospitalization
Compartment_train = [8,9,10,5]

daily_death_H = provincial_data_CIHI["Deaths12"].values[::1]
daily_recovered_H = provincial_data_CIHI["Discharges11"].values[::1]
daily_admitted_H

# creating data set based on the specified interval
#data_set=data_set_create(Start_day_training=Start_day_training, end_day_training=end_day_training)

# this is important to have it like this format since these variables will be used throughout the model
#training_set1 = data_set[0]
#training_set1_cumulative = data_set[1]
#training_set2 = data_set[2]
day_start=Start_day_training
day_end=end_day_training




# perfoming the curve fitting function
#k = true k is being estimated
#k = false k is constant
# NOTE 1 : do NOT change the ordering sequence of you input the data inside model
Result_curve_fitting_1=model_fitting_function_multi(training_data1_cumulative = daily_cases_cumulative,
                                                    training_data1 = daily_cases,
                                                    training_data2 = daily_admitted_H, 
                                                    training_data3 = daily_discharge,
                                                    training_data4 = daily_hospitalization,
                                                    outbreak_shift = 0,
                                                    compartment_array = Compartment_train,
                                                    initial_parameter= params_init_min_max,
                                                    day_start=Start_day_training,
                                                    day_end=end_day_training,
                                                    k_inclusion=False,
                                                    brute_step=4)

# Creating a data frame of result and real data
#dataframe_result_1 = dataframe_transform(Result_curve_fitting=Result_curve_fitting_1,
#                                         data1_cumulative = daily_cases_cumulative,
#                                         data1 = daily_cases, 
#                                         data2 = daily_admitted_H,
#                                         data3 = daily_recovered_H,
#                                         data4 = daily_death_H,
#                                         day_start=Start_day_training,
#                                         day_end=end_day_training,
#                                         outbreak_shift = 0)

# daily for lagged
#creating a dataframe with correct index and to put the results in
daily_hospitalization=provincial_data_CIHI['daily_Hospitalizations']
hospitalization_data_SIR_forecast=daily_hospitalization.to_frame()

complete_interpolated_pandemic_data=complete_pandemic_data.copy()

period_info=7

for i in range(91,len(complete_pandemic_data)):
  # SECTION 1
  #========================================================================================================================
  #fitting the curve on the first 120 days of pandemic

  #Creating initial guesses and upper and lower bound for the parameters that model needs to estimate
  #first 50 days
  params_init_min_max = Parameters()
  params_init_min_max.add("R_0_start",   min = 1.5,    max = 10   )
  params_init_min_max.add("x0",          min = 50,      max = 120    )
  #params_init_min_max.add("k",            min = 0.1,   max = 5  )
  params_init_min_max.add("R_0_end",     min = 0.3,    max = 10     )
  #params_init_min_max.add("alpha",       min = 0.003,  max = 0.05   )
  params_init_min_max.add("rate_OTH",         min = 0.1,    max = 0.7    )
  params_init_min_max.add("rate_H",         min = 0.1,    max = 0.7    )
  #params_init_min_max.add("rho_H",         min = 0.01,    max = 0.7    )

  #Fixed parameters
  k_constant=1.0
  #rho=0.13
  sigma=1/5.1
  gamma_D=1/12.1689
  alpha_H = 0.14876
  prob_H = 0.188885

  # setting the initial values of each compartment
  start_point_1 = [population_Region,1,0,0,0,0,0,0,0,0]
  starting_data = start_point_1

  # setting up the date on which the curve fitting function will be perfomed
  Start_day_training = 0
  end_day_training = (i)
  start_day_estimation = end_day_training

  #This is the array of number of each compartment that needs to be fitted on the real data
  # 7  : Cumulative number of cases
  # 8  : Daily number of cases 
  # 9  : Daily admission to hospital
  # 10 : Daily number of discharge at hospitals
  # 5  : Daily Number of hospitalization
  Compartment_train = [8,9,10,5]

  #daily_death_H = provincial_data_CIHI["Deaths12"].values[::1]
  #daily_recovered_H = provincial_data_CIHI["Discharges11"].values[::1]
  #daily_admitted_H


  day_start=Start_day_training
  day_end=end_day_training


  # perfoming the curve fitting function
  #k = true k is being estimated
  #k = false k is constant
  # NOTE 1 : do NOT change the ordering sequence of you input the data inside model
  Result_curve_fitting_1=model_fitting_function_multi(training_data1_cumulative = complete_interpolated_pandemic_data['daily_cases_cumulative'],
                                                      training_data1 = complete_interpolated_pandemic_data['daily_cases'],
                                                      training_data2 = complete_interpolated_pandemic_data['daily_admitted_H'], 
                                                      training_data3 = complete_interpolated_pandemic_data['daily_discharge'],
                                                      training_data4 = complete_interpolated_pandemic_data['daily_hospitalization'],
                                                      outbreak_shift = 0,
                                                      compartment_array = Compartment_train,
                                                      initial_parameter= params_init_min_max,
                                                      day_start=Start_day_training,
                                                      day_end=end_day_training,
                                                      k_inclusion=False,
                                                      brute_step=4,
                                                      graphs='no')

  presentation_SIR_hospitalization()
  compart_scenarios=5
  day_start_scenarios = 0
  day_end_scenarios = len(complete_interpolated_pandemic_data)
  parameter_scenarios = Result_curve_fitting_1[1]
  estimated_data_for_table = ((Model(day_start=day_start_scenarios,
                                  day_end = day_end_scenarios,
                                  total_population=total_population,
                                  R_0_start = parameter_scenarios.params['R_0_start'].value,
                                  x0 = parameter_scenarios.params['x0'].value,
                                  rate_OTH = parameter_scenarios.params['rate_OTH'].value,
                                  rate_H = parameter_scenarios.params['rate_H'].value,
                                  starting_data = start_point_1,
                                  k= k_constant,
                                  R_0_end = parameter_scenarios.params['R_0_end'].value))[compart_scenarios])[Start_day_training:day_end_scenarios]#:end_day_training]

  #putting the simulated data into the dataframe
  hospitalization_data_SIR_forecast['forecast with %d days'%end_day_training]=estimated_data_for_table


hospitalization_data_SIR_forecast.to_csv('forecast SIR.csv')

hospitalization_data_SIR_forecast.to_csv('forecast SIR.csv')

"""#### Weekly

##### Data filtering
"""

# summing the data over 7 day interval, starting with day 1, this part will discard whcih day we will start from 
# either monday or any other day,

#creating a separate dataframe for a weekly pandemic data with correct days as its index
complete_pandemic_data=provincial_data_CIHI['daily_Hospitalizations']
complete_pandemic_data=complete_pandemic_data.to_frame().drop(columns='daily_Hospitalizations')

#assiging the overlappped data into the dataframe
complete_pandemic_data['daily_cases']=daily_cases
complete_pandemic_data['daily_cases_cumulative']=daily_cases_cumulative
complete_pandemic_data['daily_admitted_H']=daily_admitted_H
complete_pandemic_data['daily_discharge']=daily_discharge
complete_pandemic_data['daily_hospitalization']=daily_hospitalization

#reseting the index and and rename it as the date column
complete_pandemic_data=complete_pandemic_data.reset_index().rename(columns = {'index':'date'}, inplace = False)

# Formating the value of date into date format
complete_pandemic_data['date'] = pd.to_datetime(complete_pandemic_data['date'],format='%Y-%m-%d')
#complete_pandemic_data['week_number'] = complete_pandemic_data['date'].dt.week
#complete_pandemic_data['year'] = complete_pandemic_data['date'].dt.year

complete_pandemic_data.to_csv('pandemic data.csv')

complete_pandemic_data

# complete_pandemic_data.to_csv('pandemic_data.csv')

# groupby the data frame to get the seven day sum
# and divide everything by 7 since we want the average 
complete_pandemic_data_weekly_1=(complete_pandemic_data.groupby(complete_pandemic_data.index // 7).sum())/7

# adding a row of zero for time zero
complete_pandemic_data_weekly_1.loc[len(complete_pandemic_data_weekly_1)] = 0
complete_pandemic_data_weekly_1 = round(complete_pandemic_data_weekly_1.shift()).apply(np.int64)
complete_pandemic_data_weekly_1.loc[0] = 0

# complete_pandemic_data.to_csv('pandemic_data.csv')
period_info=6
# groupby the data frame to get the seven day sum
# and divide everything by 'period_info' since we want the average
# mpi = multi period info 
complete_pandemic_data_weekly_mpi=(complete_pandemic_data.groupby(complete_pandemic_data.index // period_info).sum())/period_info

# adding a row of zero for time zero
complete_pandemic_data_weekly_mpi.loc[len(complete_pandemic_data_weekly_mpi)] = 0
complete_pandemic_data_weekly_mpi = (complete_pandemic_data_weekly_mpi.shift())#.apply(np.int64)
complete_pandemic_data_weekly_mpi.loc[0] = 0

complete_pandemic_data_weekly_mpi

"""##### Forecast"""

# SECTION 1
#========================================================================================================================
#fitting the curve on the first 120 days of pandemic
period_info=7

#Creating initial guesses and upper and lower bound for the parameters that model needs to estimate
#first 50 days
params_init_min_max = Parameters()
params_init_min_max.add("R_0_start",   min = 0 ,    max = 10  )
params_init_min_max.add("x0",          min = 50/period_info ,      max = 122/period_info   )
#params_init_min_max.add("k",            min = 0.1,   max = 5  )
params_init_min_max.add("R_0_end",     min = 0,    max = 10     )
#params_init_min_max.add("alpha",       min = 0.003,  max = 0.05   )
params_init_min_max.add("rate_OTH",         min = 0.1*period_info,    max = 0.7*period_info    )
params_init_min_max.add("rate_H",         min = 0.1*period_info,    max = 0.7*period_info    )
#params_init_min_max.add("rho_H",         min = 0.01,    max = 0.7    )

#Fixed parameters
k_constant=1.0
#rho=0.13
sigma=1/(5.1/period_info)
gamma_D=1/(12.1689/period_info)
alpha_H = 0.14876
prob_H = 0.188885

# setting the initial values of each compartment
start_point_1 = [population_Region,1,0,0,0,0,0,0,0,0]
starting_data = start_point_1

# setting up the date on which the curve fitting function will be perfomed
Start_day_training = 0
end_day_training = (16)
start_day_estimation = end_day_training

#This is the array of number of each compartment that needs to be fitted on the real data
# 7  : Cumulative number of cases
# 8  : Daily number of cases 
# 9  : Daily admission to hospital
# 10 : Daily number of discharge at hospitals
# 5  : Daily Number of hospitalization
Compartment_train = [8,9,10,5]

daily_death_H = provincial_data_CIHI["Deaths12"].values[::1]
daily_recovered_H = provincial_data_CIHI["Discharges11"].values[::1]
daily_admitted_H

# creating data set based on the specified interval
#data_set=data_set_create(Start_day_training=Start_day_training, end_day_training=end_day_training)

# this is important to have it like this format since these variables will be used throughout the model
#training_set1 = data_set[0]
#training_set1_cumulative = data_set[1]
#training_set2 = data_set[2]
day_start=Start_day_training
day_end=end_day_training




# perfoming the curve fitting function
#k = true k is being estimated
#k = false k is constant
# NOTE 1 : do NOT change the ordering sequence of you input the data inside model
Result_curve_fitting_1=model_fitting_function_multi(training_data1_cumulative = complete_pandemic_data_weekly['daily_cases_cumulative'],
                                                    training_data1 = complete_pandemic_data_weekly['daily_cases'],
                                                    training_data2 = complete_pandemic_data_weekly['daily_admitted_H'], 
                                                    training_data3 = complete_pandemic_data_weekly['daily_discharge'],
                                                    training_data4 = complete_pandemic_data_weekly['daily_hospitalization'],
                                                    outbreak_shift = 0,
                                                    compartment_array = Compartment_train,
                                                    initial_parameter= params_init_min_max,
                                                    day_start=Start_day_training,
                                                    day_end=end_day_training,
                                                    k_inclusion=False,
                                                    brute_step=4,
                                                    graphs='yes')

# Creating a data frame of result and real data
#dataframe_result_1 = dataframe_transform(Result_curve_fitting=Result_curve_fitting_1,
#                                         data1_cumulative = daily_cases_cumulative,
#                                         data1 = daily_cases, 
#                                         data2 = daily_admitted_H,
#                                         data3 = daily_recovered_H,
#                                         data4 = daily_death_H,
#                                         day_start=Start_day_training,
#                                         day_end=end_day_training,
#                                         outbreak_shift = 0)

"""#### Daily-interpolated

##### Interpolation process

###### Method I:
"""

period_info=7

from scipy.interpolate import interp1d

interpolation_kind='linear'
#creating a separate dataframe for a weekly pandemic data with correct days as its index
complete_pandemic_data_daily_inter_1=provincial_data_CIHI['daily_Hospitalizations']
complete_pandemic_data_daily_inter_1=complete_pandemic_data_daily_inter_1.to_frame().drop(columns='daily_Hospitalizations')

#assiging the overlappped data into the dataframe
complete_pandemic_data_daily_inter_1['daily_cases']=daily_cases
complete_pandemic_data_daily_inter_1['daily_cases_cumulative']=daily_cases_cumulative
complete_pandemic_data_daily_inter_1['daily_admitted_H']=linear_interpolation(complete_pandemic_data['daily_admitted_H'],
                                                                            complete_pandemic_data_weekly_1_2['daily_admitted_H'],
                                                                            period_info)
complete_pandemic_data_daily_inter_1['daily_discharge']=linear_interpolation(complete_pandemic_data['daily_discharge'],
                                                                           complete_pandemic_data_weekly_1_2['daily_discharge'],
                                                                           period_info)
complete_pandemic_data_daily_inter_1['daily_hospitalization']=linear_interpolation(complete_pandemic_data['daily_hospitalization'],
                                                                                 complete_pandemic_data_weekly_1_2['daily_hospitalization'],
                                                                                 period_info)

period_info=7

from scipy.interpolate import interp1d

interpolation_kind='linear'
#creating a separate dataframe for a weekly pandemic data with correct days as its index
complete_pandemic_data_daily_inter_1_2=provincial_data_CIHI['daily_Hospitalizations']
complete_pandemic_data_daily_inter_1_2=complete_pandemic_data_daily_inter_1_2.to_frame().drop(columns='daily_Hospitalizations')

#assiging the overlappped data into the dataframe
complete_pandemic_data_daily_inter_1_2['daily_cases']=daily_cases
complete_pandemic_data_daily_inter_1_2['daily_cases_cumulative']=daily_cases_cumulative
complete_pandemic_data_daily_inter_1_2['daily_admitted_H']=linear_interpolation_3(complete_pandemic_data['daily_admitted_H'],
                                                                            complete_pandemic_data_weekly_1_2['daily_admitted_H'],
                                                                            period_info,interpolation_kind)
complete_pandemic_data_daily_inter_1_2['daily_discharge']=linear_interpolation_3(complete_pandemic_data['daily_discharge'],
                                                                           complete_pandemic_data_weekly_1_2['daily_discharge'],
                                                                           period_info,interpolation_kind)
complete_pandemic_data_daily_inter_1_2['daily_hospitalization']=linear_interpolation_3(complete_pandemic_data['daily_hospitalization'],
                                                                                 complete_pandemic_data_weekly_1_2['daily_hospitalization'],
                                                                                 period_info,interpolation_kind)

from scipy.interpolate import interp1d



period_info=10
interpolation_kind='linear'

dict_of_interpolated_data = {}

for i in range(1,math.ceil(len(complete_pandemic_data)/period_info)+1):
  temp_daily_data=complete_pandemic_data.iloc[0:i*period_info,]
  complete_pandemic_data_weekly_mpi=replenishment_func_5(temp_daily_data,period_info)

  interpolated_data=provincial_data_CIHI['daily_Hospitalizations'].iloc[0:i*period_info,]
  interpolated_data=interpolated_data.to_frame().drop(columns='daily_Hospitalizations')

  interpolated_data['daily_cases']=linear_interpolation_6(temp_daily_data['daily_cases'],
                                                          complete_pandemic_data_weekly_mpi['daily_cases'],
                                                          period_info,interpolation_kind)
  
  interpolated_data['daily_cases_cumulative']=linear_interpolation_6(temp_daily_data['daily_cases_cumulative'],
                                                                     complete_pandemic_data_weekly_mpi['daily_cases_cumulative'],
                                                                     period_info,interpolation_kind)

  interpolated_data['daily_admitted_H']=linear_interpolation_6(temp_daily_data['daily_admitted_H'],
                                                               complete_pandemic_data_weekly_mpi['daily_admitted_H'],
                                                               period_info,interpolation_kind)

  interpolated_data['daily_discharge']=linear_interpolation_6(temp_daily_data['daily_discharge'],
                                                              complete_pandemic_data_weekly_mpi['daily_discharge'],
                                                              period_info,interpolation_kind)
  
  interpolated_data['daily_hospitalization']=linear_interpolation_6(temp_daily_data['daily_hospitalization'],
                                                                    complete_pandemic_data_weekly_mpi['daily_hospitalization'],
                                                                    period_info,interpolation_kind)
  dict_of_interpolated_data["df_interpolated_epoch_{}".format(i)]=interpolated_data.copy()
  interpolated_data=interpolated_data.drop(interpolated_data.iloc[:,:], inplace = True, axis = 1)

from scipy import interpolate

def linear_interpolation_6 (sr_origin,sr_grouped,period_info_intpl,inter_kind):


  #df_origin=test#complete_pandemic_data['daily_admitted_H']
  sr_grouped=sr_grouped.reset_index()
  limit_complete_period=math.floor(len(sr_origin)/period_info_intpl)
  limit_complete_total=math.ceil(len(sr_origin)/period_info_intpl)

  if limit_complete_period==0:
    df_temp_1=sr_grouped.iloc[0:limit_complete_total+1,]

    # x and y of the data that interpolation will be based on
    x_temp_1 = df_temp_1.iloc[:,0]
    y_temp_1 = df_temp_1.iloc[:,1]

    x_adj=x_temp_1
    x_adj[1:limit_complete_total]=x_temp_1[1:(limit_complete_total)]-0.5
    x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=len(sr_origin), endpoint=True)    
    y_interpy_temp_1 = interp1d(x_adj, y_temp_1,kind=inter_kind)
    final_array=y_interpy_temp_1(x_origin_1)
  else:

    # Creating two separate dataframe based on the completeness of the periods
    df_temp_1=sr_grouped.iloc[:limit_complete_period+1,]
    df_temp_2=sr_grouped.iloc[limit_complete_period:limit_complete_total+1,]

    # x and y of the data that interpolation will be based on
    x_temp_1 = df_temp_1.iloc[:,0]
    x_temp_2 = df_temp_2.iloc[:,0]
    y_temp_1 = df_temp_1.iloc[:,1]
    y_temp_2 = df_temp_2.iloc[:,1]

    if len(df_temp_2)>1:
      
      x_temp_2.iloc[1]=x_temp_2.iloc[0]+len(sr_origin[limit_complete_period*period_info_intpl:])/period_info_intpl

      x_adj=x_temp_1.copy()
      x_adj[1:limit_complete_total]=(x_temp_1[1:limit_complete_total]-0.5)
      x_adj=x_adj.append(x_temp_2,ignore_index=True)
      y_adj=y_temp_1.copy()
      y_adj=y_adj.append(y_temp_2,ignore_index=True)
      y_adj.iloc[-2]=y_temp_2.iloc[1]

      x_origin_1=np.linspace(x_temp_1.iloc[0], x_temp_2.iloc[-2], num=limit_complete_period*period_info_intpl, endpoint=True)    
      x_origin_2=(np.linspace(x_temp_2.iloc[-2], x_adj.iloc[-1], num=len(sr_origin[limit_complete_period*period_info_intpl:])+1, endpoint=True))[1:]
      x_origin_new=np.concatenate([x_origin_1,x_origin_2])

      y_interpy_temp_1 = interp1d(x_adj, y_adj,kind=inter_kind)
      final_array=y_interpy_temp_1(x_origin_new)

    elif len(df_temp_2)==1:

      x_adj=x_temp_1.copy()
      #dividing all period_info in half so that the interpolation graph would pass through them all
      x_adj[1:limit_complete_total]=x_temp_1[1:(limit_complete_total)]-0.5
      # adding one more data for the last value so that it would better represent the data
      x_adj[limit_complete_total]=x_temp_1[limit_complete_total-1]
      x_adj[limit_complete_total+1]=x_temp_1[limit_complete_total]
      x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=limit_complete_period*period_info_intpl, endpoint=True)    
      y_temp_1[limit_complete_total]=y_temp_1[limit_complete_total]
      y_temp_1[limit_complete_total+1]=y_temp_1[limit_complete_total]
      y_interpy_temp_1 = interp1d(x_adj, y_temp_1,kind=inter_kind)
      final_array=y_interpy_temp_1(x_origin_1)

  return final_array

from scipy.interpolate import interp1d

R=7
period_info=6
interpolation_kind='linear'

i=9

temp_daily_data=complete_pandemic_data.iloc[0:i*R,]
complete_pandemic_data_weekly_mpi=replenishment_func_5(temp_daily_data,period_info)

interpolated_data=provincial_data_CIHI['daily_Hospitalizations'].iloc[0:i*R,]
interpolated_data=interpolated_data.to_frame().drop(columns='daily_Hospitalizations')

interpolated_data['daily_cases']=daily_cases[0:i*R]
interpolated_data['daily_cases_cumulative']=daily_cases_cumulative[0:i*R]

interpolated_data['daily_admitted_H']=linear_interpolation_4(temp_daily_data['daily_admitted_H'],
                                                                          complete_pandemic_data_weekly_mpi['daily_admitted_H'],
                                                                          period_info,interpolation_kind)

interpolated_data['daily_discharge']=linear_interpolation_4(temp_daily_data['daily_discharge'],
                                                                          complete_pandemic_data_weekly_mpi['daily_discharge'],
                                                                          period_info,interpolation_kind)
interpolated_data['daily_hospitalization']=linear_interpolation_4(temp_daily_data['daily_hospitalization'],
                                                                                complete_pandemic_data_weekly_mpi['daily_hospitalization'],
                                                                                period_info,interpolation_kind)
test=interpolated_data.copy()
interpolated_data=interpolated_data.drop(interpolated_data.iloc[:,:], inplace = True, axis = 1)

#interpolated_data=interpolated_data.iloc[0:0,0:0]

fig = plt.figure(figsize=(17,8.5))
ax = fig.add_subplot(111)

ax.plot(temp_daily_data['date'],temp_daily_data['daily_hospitalization'],'b',label='original',linewidth=2)
ax.plot(temp_daily_data['date'],test['daily_hospitalization'],'r',label='interpolated',linewidth=4)

period_info=8
R=7



complete_pandemic_data_weekly_mpi=replenishment_func_3(complete_pandemic_data,period_info)
i=1
test_data=complete_pandemic_data.iloc[0:i*R,]

complete_pandemic_data_weekly_mpi=replenishment_func_5(test_data,period_info)


from scipy.interpolate import interp1d

interpolation_kind='linear'
#creating a separate dataframe for a weekly pandemic data with correct days as its index
complete_pandemic_data_daily_inter_1_2=provincial_data_CIHI['daily_Hospitalizations']
complete_pandemic_data_daily_inter_1_2=complete_pandemic_data_daily_inter_1_2.to_frame().drop(columns='daily_Hospitalizations')

#assiging the overlappped data into the dataframe
complete_pandemic_data_daily_inter_1_2['daily_cases']=daily_cases
complete_pandemic_data_daily_inter_1_2['daily_cases_cumulative']=daily_cases_cumulative
complete_pandemic_data_daily_inter_1_2['daily_admitted_H']=linear_interpolation_3(complete_pandemic_data['daily_admitted_H'],
                                                                            complete_pandemic_data_weekly_1_2['daily_admitted_H'],
                                                                            period_info,interpolation_kind)
complete_pandemic_data_daily_inter_1_2['daily_discharge']=linear_interpolation_3(complete_pandemic_data['daily_discharge'],
                                                                           complete_pandemic_data_weekly_1_2['daily_discharge'],
                                                                           period_info,interpolation_kind)
complete_pandemic_data_daily_inter_1_2['daily_hospitalization']=linear_interpolation_3(complete_pandemic_data['daily_hospitalization'],
                                                                                 complete_pandemic_data_weekly_1_2['daily_hospitalization'],
                                                                                 period_info,interpolation_kind)

from scipy import interpolate

def linear_interpolation_3 (sr_origin,sr_grouped,period_info_intpl,inter_kind):

  #df_origin=test#complete_pandemic_data['daily_admitted_H']
  sr_grouped=sr_grouped.reset_index()
  limit_complete_period=math.floor(len(sr_origin)/period_info_intpl)
  limit_complete_total=math.ceil(len(sr_origin)/period_info_intpl)

  # Creating two separate dataframe based on the completeness of the periods
  df_temp_1=sr_grouped.iloc[:limit_complete_period+1,]
  df_temp_2=sr_grouped.iloc[limit_complete_period:limit_complete_total+1,]

  # x and y of the data that interpolation will be based on
  x_temp_1 = df_temp_1.iloc[:,0]
  x_temp_2 = df_temp_2.iloc[:,0]
  y_temp_1 = df_temp_1.iloc[:,1]
  y_temp_2 = df_temp_2.iloc[:,1]

  if len(df_temp_2)>1:
    
    x_temp_2.iloc[1]=x_temp_2.iloc[0]+len(sr_origin[limit_complete_period*period_info_intpl:])/period_info_intpl

    x_adj=x_temp_1.append(x_temp_2.iloc[1:])
    y_adj=y_temp_1.append(y_temp_2.iloc[1:])

    x_adj[1:limit_complete_total]=x_adj[1:limit_complete_total]-0.5
    x_adj[limit_complete_total]=x_adj[limit_complete_total]-(len(sr_origin[limit_complete_period*period_info_intpl:])/period_info_intpl)/2

    x_origin_1=np.linspace(x_temp_1.iloc[0], x_temp_2.iloc[-2], num=limit_complete_period*period_info_intpl, endpoint=True)    
    x_origin_2=(np.linspace(x_temp_2.iloc[-2], x_adj.iloc[-1], num=len(sr_origin[limit_complete_period*period_info_intpl:])+1, endpoint=True))[1:]
    x_origin_new=np.concatenate([x_origin_1,x_origin_2])

    y_interpy_temp_1 = interp1d(x_adj, y_adj,kind=inter_kind)
    final_array=y_interpy_temp_1(x_origin_new)

  elif len(df_temp_2)==1:

    x_adj=x_temp_1
    x_adj[1:limit_complete_total]=x_temp_1[1:(limit_complete_total)]-0.5
    x_origin_1=np.linspace(df_temp_1.iloc[0,0], df_temp_1.iloc[-1,0], num=limit_complete_period*period_info_intpl, endpoint=True)    
    y_interpy_temp_1 = interp1d(x_adj, y_temp_1,kind=inter_kind)
    final_array=y_interpy_temp_1(x_origin_1)

  return final_array

"""###### Method II:"""

# Computing the weekly data, in this section we simply sum the data every period_info interval
# complete_pandemic_data.to_csv('pandemic_data.csv')

# groupby the data frame to get the seven day sum
# and divide everything by 7 since we want the average 
complete_pandemic_data_weekly_2=(complete_pandemic_data.groupby(complete_pandemic_data.index // 7).sum())

# adding a row of zero for time zero
complete_pandemic_data_weekly_2.loc[len(complete_pandemic_data_weekly_2)] = 0
complete_pandemic_data_weekly_2 = round(complete_pandemic_data_weekly_2.shift()).apply(np.int64)
complete_pandemic_data_weekly_2.loc[0] = 0

period_info=7

from scipy.interpolate import interp1d

interpolation_kind='cubic'
#creating a separate dataframe for a weekly pandemic data with correct days as its index
complete_pandemic_data_daily_inter_2_2=provincial_data_CIHI['daily_Hospitalizations'][:119]
complete_pandemic_data_daily_inter_2_2=complete_pandemic_data_daily_inter_2_2.to_frame().drop(columns='daily_Hospitalizations')

#assiging the overlappped data into the dataframe
complete_pandemic_data_daily_inter_2_2['daily_cases']=daily_cases[:119]
complete_pandemic_data_daily_inter_2_2['daily_cases_cumulative']=daily_cases_cumulative[:119]
complete_pandemic_data_daily_inter_2_2['daily_admitted_H']=linear_interpolation_2(complete_pandemic_data['daily_admitted_H'][:119],
                                                                            complete_pandemic_data_weekly_1_2['daily_admitted_H'][:18],
                                                                            period_info,interpolation_kind)
complete_pandemic_data_daily_inter_2_2['daily_discharge']=linear_interpolation_2(complete_pandemic_data['daily_discharge'][:119],
                                                                           complete_pandemic_data_weekly_1_2['daily_discharge'][:18],
                                                                           period_info,interpolation_kind)
complete_pandemic_data_daily_inter_2_2['daily_hospitalization']=linear_interpolation_2(complete_pandemic_data['daily_hospitalization'][:119],
                                                                                 complete_pandemic_data_weekly_1_2['daily_hospitalization'][:18],
                                                                                 period_info,interpolation_kind)

#creating a dataframe to assign the new interpolated data into
complete_pandemic_data_daily_inter_2=complete_pandemic_data.set_index('date').copy()
for j in range(len(complete_pandemic_data_daily_inter_2.columns)):
  for i in range(1,math.ceil(len(complete_pandemic_data_daily_inter_2)/period_info)+1):
    complete_pandemic_data_daily_inter_2.iloc[(i-1)*period_info:i*period_info,j]=complete_pandemic_data_weekly_2.iloc[i,j]/len(complete_pandemic_data_daily_inter_2.iloc[(i-1)*period_info:i*period_info,j])

from scipy.interpolate import interp1d



period_info=10
interpolation_kind='linear'

dict_of_interpolated_data = {}

for i in range(1,math.ceil(len(complete_pandemic_data)/period_info)+1):
  temp_daily_data=complete_pandemic_data.iloc[0:i*period_info,]
  complete_pandemic_data_weekly_mpi=replenishment_func_5(temp_daily_data,period_info)

  interpolated_data=provincial_data_CIHI['daily_Hospitalizations'].iloc[0:i*period_info,]
  interpolated_data=interpolated_data.to_frame().drop(columns='daily_Hospitalizations')

  interpolated_data['daily_cases']=daily_cases[0:i*period_info]
  interpolated_data['daily_cases_cumulative']=daily_cases_cumulative[0:i*period_info]

  interpolated_data['daily_admitted_H']=linear_interpolation_4(temp_daily_data['daily_admitted_H'],
                                                                            complete_pandemic_data_weekly_mpi['daily_admitted_H'],
                                                                            period_info,interpolation_kind)

  interpolated_data['daily_discharge']=linear_interpolation_4(temp_daily_data['daily_discharge'],
                                                                            complete_pandemic_data_weekly_mpi['daily_discharge'],
                                                                            period_info,interpolation_kind)
  interpolated_data['daily_hospitalization']=linear_interpolation_4(temp_daily_data['daily_hospitalization'],
                                                                                  complete_pandemic_data_weekly_mpi['daily_hospitalization'],
                                                                                  period_info,interpolation_kind)
  dict_of_interpolated_data["df_interpolated_epoch_{}".format(i)]=interpolated_data.copy()
  interpolated_data=interpolated_data.drop(interpolated_data.iloc[:,:], inplace = True, axis = 1)

test=dict_of_interpolated_data['df_interpolated_epoch_13']['daily_hospitalization']

fig = plt.figure(figsize=(17,8.5))
ax = fig.add_subplot(111)

#ax.plot(complete_pandemic_data_daily_inter_2.index[:119],complete_pandemic_data_daily_inter_2['daily_hospitalization'][:119],'b',label='new method',linewidth=2)
ax.plot(test.index,test,'r',label='old method',linewidth=4)
ax.plot(test.index[:119],complete_pandemic_data['daily_hospitalization'][:119],'g',label='new method',linewidth=4)
#ax.plot(complete_pandemic_data_daily_inter_2.index[:119],complete_pandemic_data_daily_inter_1['daily_hospitalization'][:119],'y',label='new method',linewidth=2)
#ax.plot(complete_pandemic_data_daily_inter_2.index[:119],complete_pandemic_data_daily_inter_2_2['daily_hospitalization'][:119],'p',label='new method',linewidth=2)

fig = plt.figure(figsize=(17,8.5))
ax = fig.add_subplot(111)


ax.plot(complete_pandemic_data_daily_inter_2.index[:119],complete_pandemic_data_daily_inter_2['daily_hospitalization'][:119],'b',label='new method',linewidth=2)
#ax.plot(complete_pandemic_data_daily_inter_2.index[:119],y_new_inter,'r',label='old method',linewidth=2)
ax.plot(complete_pandemic_data_daily_inter_2.index[:119],complete_pandemic_data['daily_hospitalization'][:119],'g',label='new method',linewidth=2)

fig = plt.figure(figsize=(17,8.5))
ax = fig.add_subplot(111)

ax.plot(complete_pandemic_data_daily_inter_2.index,complete_pandemic_data_daily_inter_2['daily_hospitalization'],'b',label='new method 1',linewidth=2)
ax.plot(complete_pandemic_data_daily_inter_2.index,complete_pandemic_data_daily_inter_1['daily_hospitalization'],'r',label='old method',linewidth=2)
ax.plot(complete_pandemic_data_daily_inter_2.index,complete_pandemic_data_daily_inter_1_2['daily_hospitalization'],'g',label='new method 2',linewidth=2)


plt.title('new vs old',fontsize=16)
plt.xlabel('Time (days)',fontsize=16)
plt.ylabel('Hospitalization',fontsize=16)



plt.legend(fontsize=13)

ily

"""##### Forecast"""

# summing the data over 7 day interval, starting with day 1, this part will discard whcih day we will start from 
# either monday or any other day,

#creating a separate dataframe for a weekly pandemic data with correct days as its index
complete_pandemic_data=provincial_data_CIHI['daily_Hospitalizations']
complete_pandemic_data=complete_pandemic_data.to_frame().drop(columns='daily_Hospitalizations')

#assiging the overlappped data into the dataframe
complete_pandemic_data['daily_cases']=daily_cases
complete_pandemic_data['daily_cases_cumulative']=daily_cases_cumulative
complete_pandemic_data['daily_admitted_H']=daily_admitted_H
complete_pandemic_data['daily_discharge']=daily_discharge
complete_pandemic_data['daily_hospitalization']=daily_hospitalization

#reseting the index and and rename it as the date column
complete_pandemic_data=complete_pandemic_data.reset_index().rename(columns = {'index':'date'}, inplace = False)

# Formating the value of date into date format
complete_pandemic_data['date'] = pd.to_datetime(complete_pandemic_data['date'],format='%Y-%m-%d')
#complete_pandemic_data['week_number'] = complete_pandemic_data['date'].dt.week
#complete_pandemic_data['year'] = complete_pandemic_data['date'].dt.year

################################

(variable_period_info_df.iloc[18,:].dropna()).astype(int)

variable_period_info_df=pd.read_csv('variable 10000 period info.csv')
variable_period_info_df=variable_period_info_df.drop(columns='Unnamed: 0')

complete_vpi_SIR_forecast=pd.DataFrame()
start_it=19
end_it=20

for k in range(start_it,end_it):
  data_for_grouping=complete_pandemic_data.copy()
  temp_index_grouped_origin=(variable_period_info_df.iloc[k,:].dropna()).astype(int)
  data_for_grouping['grouping_index']=data_for_grouping['daily_cases']
  data_for_grouping['grouping_index']=np.nan
  s0 = pd.Series([0])
  temp_index_grouped=(s0.append(temp_index_grouped_origin, ignore_index=True)).astype(int)

  for i in range(1,len(temp_index_grouped)):
    data_for_grouping['grouping_index'][temp_index_grouped[0:i].sum():temp_index_grouped[0:i+1].sum()]=i-1#temp_index_grouped[i]

  df_grouped=(data_for_grouping.groupby(data_for_grouping['grouping_index'],sort=False).sum()).div(temp_index_grouped_origin.values,axis=0)#/len(data_for_grouping)

  daily_grouped_data=data_for_grouping.set_index('grouping_index')
  daily_grouped_data.iloc[:,1:6]=df_grouped
  daily_grouped_data_SIR=daily_grouped_data.set_index('date').dropna()
  daily_grouped_data_SIR



  #creating a dataframe with correct index and to put the results in
  daily_hospitalization=provincial_data_CIHI['daily_Hospitalizations']
  hospitalization_data_SIR_forecast=daily_hospitalization.to_frame()

  #complete_interpolated_pandemic_data=complete_pandemic_data_daily_inter_1_2.copy()

  #period_info=7

  for i in range(1,len(temp_index_grouped)):
    # Since the curve fitting cannot be applied to dat with only 1 day, we simply ignore the forecasting process for
    # day=1 and move to the next one
    if temp_index_grouped[i]==1:
      hospitalization_data_SIR_forecast['forecast with 1 days']=np.full(122, 0, dtype=int)
    else:
      # SECTION 1
      #========================================================================================================================
      #fitting the curve on the first 120 days of pandemic

      #Creating initial guesses and upper and lower bound for the parameters that model needs to estimate
      #first 50 days
      params_init_min_max = Parameters()
      params_init_min_max.add("R_0_start",   min = 1.5,    max = 10   )
      params_init_min_max.add("x0",          min = 50,      max = 120    )
      #params_init_min_max.add("k",            min = 0.1,   max = 5  )
      params_init_min_max.add("R_0_end",     min = 0.3,    max = 10     )
      #params_init_min_max.add("alpha",       min = 0.003,  max = 0.05   )
      params_init_min_max.add("rate_OTH",         min = 0.1,    max = 0.7    )
      params_init_min_max.add("rate_H",         min = 0.1,    max = 0.7    )
      #params_init_min_max.add("rho_H",         min = 0.01,    max = 0.7    )

      #Fixed parameters
      k_constant=1.0
      #rho=0.13
      sigma=1/5.1
      gamma_D=1/12.1689
      alpha_H = 0.14876
      prob_H = 0.188885

      # setting the initial values of each compartment
      start_point_1 = [population_Region,1,0,0,0,0,0,0,0,0]
      starting_data = start_point_1

      # setting up the date on which the curve fitting function will be perfomed
      Start_day_training = 0
      end_day_training = temp_index_grouped[0:i+1].sum()
      start_day_estimation = end_day_training

      complete_interpolated_pandemic_data=daily_grouped_data_SIR.iloc[0:temp_index_grouped[0:i+1].sum(),:]

      #This is the array of number of each compartment that needs to be fitted on the real data
      # 7  : Cumulative number of cases
      # 8  : Daily number of cases 
      # 9  : Daily admission to hospital
      # 10 : Daily number of discharge at hospitals
      # 5  : Daily Number of hospitalization
      Compartment_train = [8,9,10,5]

      #daily_death_H = provincial_data_CIHI["Deaths12"].values[::1]
      #daily_recovered_H = provincial_data_CIHI["Discharges11"].values[::1]
      #daily_admitted_H


      day_start=Start_day_training
      day_end=end_day_training


      # perfoming the curve fitting function
      #k = true k is being estimated
      #k = false k is constant
      # NOTE 1 : do NOT change the ordering sequence of you input the data inside model
      Result_curve_fitting_1=model_fitting_function_multi(training_data1_cumulative = complete_interpolated_pandemic_data['daily_cases_cumulative'],
                                                          training_data1 = complete_interpolated_pandemic_data['daily_cases'],
                                                          training_data2 = complete_interpolated_pandemic_data['daily_admitted_H'], 
                                                          training_data3 = complete_interpolated_pandemic_data['daily_discharge'],
                                                          training_data4 = complete_interpolated_pandemic_data['daily_hospitalization'],
                                                          outbreak_shift = 0,
                                                          compartment_array = Compartment_train,
                                                          initial_parameter= params_init_min_max,
                                                          day_start=Start_day_training,
                                                          day_end=end_day_training,
                                                          k_inclusion=False,
                                                          brute_step=4,
                                                          graphs='no')

      #presentation_SIR_hospitalization()
      compart_scenarios=5
      day_start_scenarios = 0
      day_end_scenarios = len(hospitalization_data_SIR_forecast)
      parameter_scenarios = Result_curve_fitting_1[1]
      estimated_data_for_table = ((Model(day_start=day_start_scenarios,
                                      day_end = day_end_scenarios,
                                      total_population=total_population,
                                      R_0_start = parameter_scenarios.params['R_0_start'].value,
                                      x0 = parameter_scenarios.params['x0'].value,
                                      rate_OTH = parameter_scenarios.params['rate_OTH'].value,
                                      rate_H = parameter_scenarios.params['rate_H'].value,
                                      starting_data = start_point_1,
                                      k= k_constant,
                                      R_0_end = parameter_scenarios.params['R_0_end'].value))[compart_scenarios])[Start_day_training:day_end_scenarios]#:end_day_training]

      #putting the simulated data into the dataframe
      hospitalization_data_SIR_forecast['forecast with %d days'%end_day_training]=estimated_data_for_table

  complete_vpi_SIR_forecast=complete_vpi_SIR_forecast.append(hospitalization_data_SIR_forecast,sort=False)
  #dict_variable_period_info_SIR_forecast['forecast SIR iteration %d'%k]=hospitalization_data_SIR_forecast
  #hospitalization_data_SIR_forecast.to_csv('forecast SIR with period info of %d.csv'%period_info)

complete_vpi_SIR_forecast.to_csv('result of SIR forecast vpi %dto%d.csv'%(start_it,end_it))

complete_vpi_SIR_forecast.to_csv('result of SIR forecast vpi 220,230.csv')

test1.to_csv('result of SIR forecast iteration 0.csv')
test2.to_csv('result of SIR forecast iteration 1.csv')

df_SIR_forecast_results_variable=df_SIR_forecast_results_variable.rename(columns={'Unnamed: 0':'Date'})
#setting the date as the index of the dataframe
df_SIR_forecast_results_variable=df_SIR_forecast_results_variable.set_index(df_SIR_forecast_results_variable['Date'])

complete_vpi_SIR_forecast

test0=pd.read_csv('result of SIR forecast iteration 0.csv')
test1=pd.read_csv('result of SIR forecast iteration 1.csv')
df_SIR_forecast_results_variable=test0.append(test1,sort=False)
df_SIR_forecast_results_variable

#creating a dataframe with correct index and to put the results in
daily_hospitalization=provincial_data_CIHI['daily_Hospitalizations']
hospitalization_data_SIR_forecast=daily_hospitalization.to_frame()

#complete_interpolated_pandemic_data=complete_pandemic_data_daily_inter_1_2.copy()

#period_info=7

for i in range(1,math.ceil(len(hospitalization_data_SIR_forecast)/period_info)+1):
  # SECTION 1
  #========================================================================================================================
  #fitting the curve on the first 120 days of pandemic

  #Creating initial guesses and upper and lower bound for the parameters that model needs to estimate
  #first 50 days
  params_init_min_max = Parameters()
  params_init_min_max.add("R_0_start",   min = 1.5,    max = 10   )
  params_init_min_max.add("x0",          min = 50,      max = 120    )
  #params_init_min_max.add("k",            min = 0.1,   max = 5  )
  params_init_min_max.add("R_0_end",     min = 0.3,    max = 10     )
  #params_init_min_max.add("alpha",       min = 0.003,  max = 0.05   )
  params_init_min_max.add("rate_OTH",         min = 0.1,    max = 0.7    )
  params_init_min_max.add("rate_H",         min = 0.1,    max = 0.7    )
  #params_init_min_max.add("rho_H",         min = 0.01,    max = 0.7    )

  #Fixed parameters
  k_constant=1.0
  #rho=0.13
  sigma=1/5.1
  gamma_D=1/12.1689
  alpha_H = 0.14876
  prob_H = 0.188885

  # setting the initial values of each compartment
  start_point_1 = [population_Region,1,0,0,0,0,0,0,0,0]
  starting_data = start_point_1

  # setting up the date on which the curve fitting function will be perfomed
  Start_day_training = 0
  end_day_training = (i*period_info)
  start_day_estimation = end_day_training

  complete_interpolated_pandemic_data=dict_of_interpolated_data[('df_interpolated_epoch_%d'%i)]

  #This is the array of number of each compartment that needs to be fitted on the real data
  # 7  : Cumulative number of cases
  # 8  : Daily number of cases 
  # 9  : Daily admission to hospital
  # 10 : Daily number of discharge at hospitals
  # 5  : Daily Number of hospitalization
  Compartment_train = [8,9,10,5]

  #daily_death_H = provincial_data_CIHI["Deaths12"].values[::1]
  #daily_recovered_H = provincial_data_CIHI["Discharges11"].values[::1]
  #daily_admitted_H


  day_start=Start_day_training
  day_end=end_day_training


  # perfoming the curve fitting function
  #k = true k is being estimated
  #k = false k is constant
  # NOTE 1 : do NOT change the ordering sequence of you input the data inside model
  Result_curve_fitting_1=model_fitting_function_multi(training_data1_cumulative = complete_interpolated_pandemic_data['daily_cases_cumulative'],
                                                      training_data1 = complete_interpolated_pandemic_data['daily_cases'],
                                                      training_data2 = complete_interpolated_pandemic_data['daily_admitted_H'], 
                                                      training_data3 = complete_interpolated_pandemic_data['daily_discharge'],
                                                      training_data4 = complete_interpolated_pandemic_data['daily_hospitalization'],
                                                      outbreak_shift = 0,
                                                      compartment_array = Compartment_train,
                                                      initial_parameter= params_init_min_max,
                                                      day_start=Start_day_training,
                                                      day_end=end_day_training,
                                                      k_inclusion=False,
                                                      brute_step=4,
                                                      graphs='no')

  presentation_SIR_hospitalization()
  compart_scenarios=5
  day_start_scenarios = 0
  day_end_scenarios = len(hospitalization_data_SIR_forecast)
  parameter_scenarios = Result_curve_fitting_1[1]
  estimated_data_for_table = ((Model(day_start=day_start_scenarios,
                                  day_end = day_end_scenarios,
                                  total_population=total_population,
                                  R_0_start = parameter_scenarios.params['R_0_start'].value,
                                  x0 = parameter_scenarios.params['x0'].value,
                                  rate_OTH = parameter_scenarios.params['rate_OTH'].value,
                                  rate_H = parameter_scenarios.params['rate_H'].value,
                                  starting_data = start_point_1,
                                  k= k_constant,
                                  R_0_end = parameter_scenarios.params['R_0_end'].value))[compart_scenarios])[Start_day_training:day_end_scenarios]#:end_day_training]

  #putting the simulated data into the dataframe
  hospitalization_data_SIR_forecast['forecast with %d days'%end_day_training]=estimated_data_for_table


hospitalization_data_SIR_forecast.to_csv('forecast SIR with period info of %d.csv'%period_info)

hospitalization_data_SIR_forecast.to_csv('forecast SIR with period info of %d.csv'%period_info)

hospitalization_data_SIR_forecast

"""### Data presentation (Real data with forecast)"""

def presentation_SIR_hospitalization():

  #RESULT PRESENTATIONS
  #============================================================================================================
  #INPUT SECTION
  #===============================================================================
  #input different scenarios for when the plato happens (in days:)
  #X0_range=[55]
  parameter_scenarios = Result_curve_fitting_1[1]
  X0_range=[parameter_scenarios.params['x0'].value]
  #input different scenarios for how much the R0 will after the plato:
  #R0end_range=[0.8,3.86391]
  R0end_range=[parameter_scenarios.params['R_0_end'].value]
  #Starting point from the last curve fitting process:
  start_point_scenarios = start_point_1

  # IMPORTANT: you need to change these parameters ot the those of the last fitting process:
  #==================================================
  parameter_scenarios = Result_curve_fitting_1[1]
  k_scenarios=k_constant
  sigma_scenarios=sigma
  #gamma_scenarios=gamma
  day_start_scenarios = 0
  day_end_scenarios = 122

  #==================================================
  #which compartment do you want to see:

  # 7  : Cumulative number of cases
  # 8  : Daily number of cases 
  # 9  : Daily admission to hospital
  # 10 : Daily number of discharge at hospitals
  # 5  : Daily Number of hospitalization

  compart_scenarios = 5
  #===============================================================================


  #PLOTTING FUNCTION
  #===============================================================================
  t_scenarios = np.linspace(day_start_scenarios, day_end_scenarios-1, (day_end_scenarios-day_start_scenarios))

  #selectin the real data related to the requested compartment 
  if compart_scenarios == 7:
    real_data_scenario = daily_cases_cumulative[day_start_scenarios:day_end_scenarios]
  elif compart_scenarios == 8:
    real_data_scenario = daily_cases[day_start_scenarios:day_end_scenarios]
  elif compart_scenarios == 9:
    real_data_scenario = daily_admitted_H[day_start_scenarios:day_end_scenarios]
  elif compart_scenarios == 10:
    real_data_scenario = daily_discharge[day_start_scenarios:day_end_scenarios]
  elif compart_scenarios == 5:
    real_data_scenario = daily_hospitalization[day_start_scenarios:day_end_scenarios]


  fig = plt.figure(figsize=(17,8.5))
  ax = fig.add_subplot(111)



  ax.plot (t_scenarios[Start_day_training:end_day_training],((Model(day_start=day_start_scenarios,
                                  day_end = day_end_scenarios,
                                  total_population=total_population,
                                  R_0_start = parameter_scenarios.params['R_0_start'].value,
                                  x0=X0_range[0],
                                  rate_OTH = parameter_scenarios.params['rate_OTH'].value,
                                  rate_H = parameter_scenarios.params['rate_H'].value,
                                  starting_data = start_point_scenarios,
                                  k= k_scenarios,
                                  R_0_end = R0end_range[0]))[compart_scenarios])[Start_day_training:end_day_training],'b',label='Curve fitting result made by the model',linewidth=2)
  ax.plot (t_scenarios[(end_day_training-1):day_end_scenarios],((Model(day_start=day_start_scenarios,
                                  day_end = day_end_scenarios,
                                  total_population=total_population,
                                  R_0_start = parameter_scenarios.params['R_0_start'].value,
                                  x0=X0_range[0],
                                  rate_OTH = parameter_scenarios.params['rate_OTH'].value,
                                  rate_H = parameter_scenarios.params['rate_H'].value,
                                  starting_data = start_point_scenarios,
                                  k= k_scenarios,
                                  R_0_end = R0end_range[0]))[compart_scenarios])[(end_day_training-1):day_end_scenarios],'orange',label='Prediction made by the model',linewidth=2)

      #ax.set_xlim(xmin=0)
      #ax.set_ylim(ymax=200)

  #Start_day_training = 0
  #end_day_training = 50

  #plotting the training set of data
  ax.plot(t_scenarios[Start_day_training:end_day_training], real_data_scenario[Start_day_training:end_day_training], 'r',label='Real hospitalization data (training set)',linewidth=4)
  #plotting the test set of data
  ax.plot(t_scenarios[(end_day_training-1):day_end_scenarios], real_data_scenario[(end_day_training-1):day_end_scenarios], 'C12',label='Real hospitalization data (test set)',linewidth=4)

  ax.set_ylim(ymin=0,ymax=500)
  plt.title('Prediction of COVID-19 hospitalization in BC',fontsize=16)
  plt.xlabel('Time (days)',fontsize=16)
  plt.ylabel('Hospitalization',fontsize=16)



  plt.legend(fontsize=13)

  plt.savefig('Prediction based on the first %d days'%end_day_training,dpi=300)

  return

#RESULT PRESENTATIONS
#============================================================================================================
#INPUT SECTION
#===============================================================================
#input different scenarios for when the plato happens (in days:)
#X0_range=[55]
parameter_scenarios = Result_curve_fitting_1[1]
X0_range=[parameter_scenarios.params['x0'].value]
#input different scenarios for how much the R0 will after the plato:
#R0end_range=[0.8,3.86391]
R0end_range=[parameter_scenarios.params['R_0_end'].value]
#Starting point from the last curve fitting process:
start_point_scenarios = start_point_1

# IMPORTANT: you need to change these parameters ot the those of the last fitting process:
#==================================================
parameter_scenarios = Result_curve_fitting_1[1]
k_scenarios=k_constant
sigma_scenarios=sigma
#gamma_scenarios=gamma
day_start_scenarios = 0
day_end_scenarios = 122

#==================================================
#which compartment do you want to see:

# 7  : Cumulative number of cases
# 8  : Daily number of cases 
# 9  : Daily admission to hospital
# 10 : Daily number of discharge at hospitals
# 5  : Daily Number of hospitalization

compart_scenarios = 5
#===============================================================================


#PLOTTING FUNCTION
#===============================================================================
t_scenarios = np.linspace(day_start_scenarios, day_end_scenarios-1, (day_end_scenarios-day_start_scenarios))

#selectin the real data related to the requested compartment 
if compart_scenarios == 7:
  real_data_scenario = daily_cases_cumulative[day_start_scenarios:day_end_scenarios]
elif compart_scenarios == 8:
  real_data_scenario = daily_cases[day_start_scenarios:day_end_scenarios]
elif compart_scenarios == 9:
  real_data_scenario = daily_admitted_H[day_start_scenarios:day_end_scenarios]
elif compart_scenarios == 10:
  real_data_scenario = daily_discharge[day_start_scenarios:day_end_scenarios]
elif compart_scenarios == 5:
  real_data_scenario = daily_hospitalization[day_start_scenarios:day_end_scenarios]


fig = plt.figure(figsize=(17,8.5))
ax = fig.add_subplot(111)



ax.plot (t_scenarios[Start_day_training:end_day_training],((Model(day_start=day_start_scenarios,
                                day_end = day_end_scenarios,
                                total_population=total_population,
                                R_0_start = parameter_scenarios.params['R_0_start'].value,
                                x0=X0_range[0],
                                rate_OTH = parameter_scenarios.params['rate_OTH'].value,
                                rate_H = parameter_scenarios.params['rate_H'].value,
                                starting_data = start_point_scenarios,
                                k= k_scenarios,
                                R_0_end = R0end_range[0]))[compart_scenarios])[Start_day_training:end_day_training],'b',label='Curve fitting result made by the model',linewidth=2)
ax.plot (t_scenarios[(end_day_training-1):day_end_scenarios],((Model(day_start=day_start_scenarios,
                                day_end = day_end_scenarios,
                                total_population=total_population,
                                R_0_start = parameter_scenarios.params['R_0_start'].value,
                                x0=X0_range[0],
                                rate_OTH = parameter_scenarios.params['rate_OTH'].value,
                                rate_H = parameter_scenarios.params['rate_H'].value,
                                starting_data = start_point_scenarios,
                                k= k_scenarios,
                                R_0_end = R0end_range[0]))[compart_scenarios])[(end_day_training-1):day_end_scenarios],'orange',label='Prediction made by the model',linewidth=2)

    #ax.set_xlim(xmin=0)
    #ax.set_ylim(ymax=200)

#Start_day_training = 0
#end_day_training = 50

#plotting the training set of data
ax.plot(t_scenarios[Start_day_training:end_day_training], real_data_scenario[Start_day_training:end_day_training], 'r',label='Real hospitalization data (training set)',linewidth=4)
#plotting the test set of data
ax.plot(t_scenarios[(end_day_training-1):day_end_scenarios], real_data_scenario[(end_day_training-1):day_end_scenarios], 'C12',label='Real hospitalization data (test set)',linewidth=4)

ax.set_ylim(ymin=0,ymax=500)
plt.title('Prediction of COVID-19 hospitalization in BC',fontsize=16)
plt.xlabel('Time (days)',fontsize=16)
plt.ylabel('Hospitalization',fontsize=16)



plt.legend(fontsize=13)

plt.savefig('Prediction based on the first %d days'%end_day_training,dpi=300)

"""### Data presentation (Real data without forecast)"""

# IMPORTANT NOTE: RUN WITH CAUTION
#RESULT PRESENTATIONS
#============================================================================================================
#INPUT SECTION
#===============================================================================
#input different scenarios for when the plato happens (in days:)
#X0_range=[55]
#parameter_scenarios = Result_curve_fitting_1[1]
#X0_range=[parameter_scenarios.params['x0'].value]
#input different scenarios for how much the R0 will after the plato:
#R0end_range=[0.8,3.86391]
#R0end_range=[parameter_scenarios.params['R_0_end'].value]
#Starting point from the last curve fitting process:
#start_point_scenarios = start_point_1

real_data_scenario=y.copy()
day_start_scenarios=0
day_end_scenarios=len(real_data_scenario)

#==================================================
#which compartment do you want to see:

# 7  : Cumulative number of cases
# 8  : Daily number of cases 
# 9  : Daily admission to hospital
# 10 : Daily number of discharge at hospitals
# 5  : Daily Number of hospitalization

compart_scenarios = 5
#===============================================================================


#PLOTTING FUNCTION
#===============================================================================
#t_scenarios = np.linspace(day_start_scenarios, day_end_scenarios-1, (day_end_scenarios-day_start_scenarios))
t_scenarios = np.linspace(day_start_scenarios, day_end_scenarios-1, (day_end_scenarios-day_start_scenarios))




fig = plt.figure(figsize=(17,8.5))
ax = fig.add_subplot(111)

    #ax.set_xlim(xmin=0)
    #ax.set_ylim(ymax=200)

#Start_day_training = 0
#end_day_training = 50

#plotting the training set of data
ax.plot(t_scenarios[day_start_scenarios:day_end_scenarios], real_data_scenario, 'orange',linewidth=4)

ax.set_ylim(ymin=0,ymax=800)
plt.title('Simulated PPE consumption in British Columbia',fontsize=16)
plt.xlabel('Time (days)',fontsize=16)
plt.ylabel('PPE consumption',fontsize=16)



plt.legend(fontsize=13)

plt.savefig('Simulated PPE consumption in British Columbia in BC',dpi=300)

"""### Data saving module"""

# >>>>> IMPORTANT NOTE: this code should be run only ONCE <<<<<
#creating a dataframe with correct index and to put the results in
daily_hospitalization=provincial_data_CIHI['daily_Hospitalizations']
hospitalization_data_SIR_forecast=daily_hospitalization.to_frame()

#in this section, we will use the parameters found by the model to produce data for 
#the desired compartment specified in the previous section
estimated_data = ((Model(day_start=day_start_scenarios,
                                day_end = day_end_scenarios,
                                total_population=total_population,
                                R_0_start = parameter_scenarios.params['R_0_start'].value,
                                x0=X0_range[0],
                                rate_OTH = parameter_scenarios.params['rate_OTH'].value,
                                rate_H = parameter_scenarios.params['rate_H'].value,
                                starting_data = start_point_scenarios,
                                k= k_scenarios,
                                R_0_end = R0end_range[0]))[compart_scenarios])[Start_day_training:day_end_scenarios]#:end_day_training]

#putting the simulated data into the dataframe
hospitalization_data_SIR_forecast['forecast with %d days'%end_day_training]=estimated_data
hospitalization_data_SIR_forecast.head()

#Exporting the data into the a csv file which will be saved into the drive
hospitalization_data_SIR_forecast.to_csv('drive/My Drive/HEC CLASSES/forecast SIR.csv')

"""# Periodic Review System

## Parameters, coefficients, initial state
"""

# Parameters, coefficients, initial state
#==============================================================================
# >>> USER INPUT <<<

# in this section we simulate the demand/consumption of medical supplies 

#protocol coef is the coefficient that will be multiplied by the hospitalization to get the consumption data.
#this coefficient is derived from government protocols and can be dynamics and changes through time
#For simplicity we have used a constant at this point 
try:
  protocol_coef=protocol_coef
except NameError:
  protocol_coef=4.5

#setting the LENGTH OF PERIOD for each epoch 
# this means that epochs R days apart, the unit is in (days)
try:
  R=R
except NameError:
  R=7

# setting the LEAD TIME for each order 
# NOTE: the order arrives at the end pf the day, therefore it will be used as a supply for the next day demand
# the unit is (days)
try:
  L=L
except NameError:
  L=5
  
# desired fill rate of inventory 
fill_rate=0.95 

# Cycle service level
try:
  service_level=service_level
except NameError:
  service_level=0.99

# Holding cost per unit
c_h=0.5 #dollars per unit

# Ordering cost per order
c_o= 1000 #500 #dollars per order

# Unit cost per unit
c_u= 10 #dollars per unit

# Shortage cost
c_s= 60 #20 #dollars per unit

#initial inventory on hand should be specified by the user
try:
  initial_inventory=initial_inventory
except NameError:
  initial_inventory=300 #100

#supplier has minimum and maximum order quantity
try:
  suppliers_capacity_min=suppliers_capacity_min
except NameError:
  suppliers_capacity_min=600 #200

try:
  suppliers_capacity_max=suppliers_capacity_max
except NameError:
  suppliers_capacity_max= 3000 #1000


#the inventory has a maximum capacity corresponding to space area, budget, political concern and ...

try:
  inventory_capactiy=inventory_capactiy
except NameError:
  inventory_capactiy=10000 #2000

# How many items are in a box:
item_in_box=20
# How many items are in a pallet:
box_in_pallet=40

item_in_pallet= item_in_box * box_in_pallet

#creating an inventory data frame, this will be used  for all scenarios, the index of this data frame is the epoch number
# the real cost is defined as total cost without shortage cost
inventory_data = pd.DataFrame(np.array([[0,initial_inventory,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),columns=['consumption in last epoch',
                                                                                              'current',
                                                                                              'order',
                                                                                              'shortage',
                                                                                              'std of previous epoch',
                                                                                              'std from beginning',
                                                                                              'holding cost',
                                                                                              'ordering cost',
                                                                                              'unit cost',
                                                                                              'shortage cost',
                                                                                              'total cost',
                                                                                              'real cost',
                                                                                              'MAE',
                                                                                              'MAPE',
                                                                                               'percentage bias of previous epoch',
                                                                                               'Actual order'])


inventory_data

"""## Data simulation"""

#DATA SIMULATION
#==============================================================================
#For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.

provincial_data_CIHI['daily_consumption']=protocol_coef * provincial_data_CIHI['daily_Hospitalizations']
provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()

provincial_data_CIHI

# >>>USER INPUT<<<
# in this section we assign the desired data to a variable 'y' which will be used throughout this section 
# ======================================================================================================
forecast_data_1 = provincial_data_CIHI.copy()

import warnings
import matplotlib.pyplot as plt

# Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
consumption_date_type='daily_consumption'

y = forecast_data_1[consumption_date_type]
y_cumulative = forecast_data_1['cumulative_daily_consumption']

"""### Monthly average of the real data"""

fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Weekly')
ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Orders')
ax.legend();

"""### Seasonal decomposition of data"""

import statsmodels.api as sm

# graphs to show seasonal_decompose
def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()

"""### Stationary test of data"""

### plot for Rolling Statistic for testing Stationarity
def test_stationarity(timeseries, title):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean() 
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timeseries, label= title)
    ax.plot(rolmean, label='rolling mean');
    ax.plot(rolstd, label='rolling std (x10)');
    ax.legend()

# Rolling mean of the data
pd.options.display.float_format = '{:.8f}'.format
test_stationarity(y,'raw data')

"""## Forecasting:

### Simple Exponential smoothing method
"""

import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing 

def ses(y, y_to_train,y_to_test,smoothing_level,predict_date):
    y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    fit1 = SimpleExpSmoothing(y_to_train).fit(smoothing_level=smoothing_level,optimized=False)
    fcast1 = fit1.forecast(predict_date).rename(r'$\alpha={}$'.format(smoothing_level))
    # specific smoothing level
    fcast1.plot(marker='o', color='blue', legend=True)
    fit1.fittedvalues.plot(marker='o',  color='blue')
    mse1 = ((fcast1 - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of our forecasts with smoothing level of {} is {}'.format(smoothing_level,round(np.sqrt(mse1), 2)))
    
    ## auto optimization
    fit2 = SimpleExpSmoothing(y_to_train).fit()
    fcast2 = fit2.forecast(predict_date).rename(r'$\alpha=%s$'%fit2.model.params['smoothing_level'])
    # plot
    fcast2.plot(marker='o', color='green', legend=True)
    fit2.fittedvalues.plot(marker='o', color='green')
    
    mse2 = ((fcast2 - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of our forecasts with auto optimization is {}'.format(round(np.sqrt(mse2), 2)))
    
    plt.show()

"""### Holt method"""

from statsmodels.tsa.api import Holt
import scipy.stats as st
import sys

def holt_2(y,y_to_train_holt,y_to_test_holt,epoch_number,previous_epoch): #,CC_deviation_holt=None):
    #y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    # This condition is in placed for the last epoch, if the last epoch is on the last day of the horizon 
    # then, there will be no more day to forecast after that, therefore, it should return and empty series
    # Since the holt function can not do this, we defined a conditional code, to solve this problem
    if holt_data=='normal':
      y_to_train_holt=y_to_train_holt
      y_to_test_holt=y_to_test_holt
    elif holt_data=='cumulative':
      y_to_train_holt=y_to_train_holt.cumsum()
      y_to_test_holt=y_to_test_holt.cumsum()
    else:
      print("Specify the type of data to be used by holt method: 'normal' or 'cumulative'")
      sys.exit()

    
    if (len(y)-epoch_number*R) > 0:
      number_of_predict_date=len(y_to_test_holt)
      fit1 = Holt(y_to_train_holt).fit()
      fcast1 = fit1.forecast(number_of_predict_date).rename("Holt's linear trend")   
      #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
      rmse1 = round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2)
      MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
      MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
      # Caluclating the percentage bias only for the interval within each epoch
      # becuase this is the interval the inventory function is based on 
      percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
    else:
      fcast1=y_to_val
      rmse1= round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2) 
      MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
      MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
      percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
    
    # once the forecast is made, id the data is cumulative, we need to adjust it back to the normal way
    if holt_data=='cumulative':
      temp_sr_rev = (y_to_train_holt.append(fcast1))[i:]
      fcast1 = pd.Series(np.diff(temp_sr_rev))


    #if CC_deviation_holt is None:
    #  CC_dev_holt = 0
    #else:
    #  CC_dev_holt = CC_deviation_holt/100
    CC_dev_holt=0
    
    # removing the negative data 
    fcast1 [fcast1 <0]=0  

    return fcast1*(1+CC_dev_holt),rmse1,MAPE,percentage_bias_holt,MAE

from statsmodels.tsa.api import Holt
import scipy.stats as st

def holt(y,y_to_train_holt,y_to_test_holt,epoch_number,previous_epoch): #,CC_deviation_holt=None):
    #y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    # This condition is in placed for the last epoch, if the last epoch is on the last day of the horizon 
    # then, there will be no more day to forecast after that, therefore, it should return and empty series
    # Since the holt function can not do this, we defined a conditional code, to solve this problem
    if (len(y)-epoch_number*R) > 0:
      number_of_predict_date=len(y_to_test_holt)
      fit1 = Holt(y_to_train_holt).fit()
      fcast1 = fit1.forecast(number_of_predict_date).rename("Holt's linear trend")   
      #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
      rmse1 = round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2)
      MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
      MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
      # Caluclating the percentage bias only for the interval within each epoch
      # becuase this is the interval the inventory function is based on 
      percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
    else:
      fcast1=y_to_val
      rmse1= round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2) 
      MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
      MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
      percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
    
    if hospitalization_data_type=='cumulative_daily_consumption':
      dumy_set_1=fcast1-previous_epoch[-1]
      dumy_set_2=fcast1-previous_epoch[-1]
    
      for j in range(len(dumy_set_1)-1):
        dumy_set_1=fcast1-previous_epoch[-1]
        dumy_set_2[j+1]=dumy_set_1[j+1]-dumy_set_1[j]

      fcast1=dumy_set_2

    #if CC_deviation_holt is None:
    #  CC_dev_holt = 0
    #else:
    #  CC_dev_holt = CC_deviation_holt/100
    CC_dev_holt=0
    
    # removing the negative data 
    fcast1 [fcast1 <0]=0  

    return fcast1*(1+CC_dev_holt),rmse1,MAPE,percentage_bias_holt,MAE

def holt_lagged(y_set_holt,y_to_test_holt,epoch_number,previous_epoch,lagged_epoch): #,CC_deviation_holt=None):
    #y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    # This condition is in placed for the last epoch, if the last epoch is on the last day of the horizon 
    # then, there will be no more day to forecast after that, therefore, it should return and empty series
    # Since the holt function can not do this, we defined a conditional code, to solve this problem

    if lagged_epoch >= epoch_number:
      # if the lag number is bigger or equal to the number of current epoch, then the forecast is zero since there is
      # no data to work with 
      holt_temp_df=y_to_test_holt.to_frame()
      holt_temp_df['holt_forecast']=0
      fcast1=holt_temp_df['holt_forecast']
      rmse1 = round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2)
      MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
      MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
      percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
    else:
      # applying the lagged epoch number to the training set
      y_to_train_holt = y_set_holt[0:(epoch_number-lagged_epoch)*R]

      if (len(y_set_holt)-epoch_number*R) > 0:
        number_of_predict_date=len(y_to_test_holt)+lagged_epoch*R
        fit1 = Holt(y_to_train_holt).fit()
        fcast1 = fit1.forecast(number_of_predict_date).rename("Holt's linear trend")   
        # getting the days that are actually required for forecasting based on the array of "y_to_test_holt"
        fcast1=fcast1[-(len(y_to_test_holt)):]
        #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
        rmse1 = round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2)
        MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
        MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
        # Caluclating the percentage bias only for the interval within each epoch
        # becuase this is the interval the inventory function is based on 
        percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
      else:
        fcast1=y_to_val
        rmse1= round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2) 
        MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
        MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
        percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
      

    #if CC_deviation_holt is None:
    #  CC_dev_holt = 0
    #else:
    #  CC_dev_holt = CC_deviation_holt/100
    CC_dev_holt=0
    
    # removing the negative data 
    fcast1 [fcast1 <0]=0  

    return fcast1*(1+CC_dev_holt),rmse1,MAPE,percentage_bias_holt,MAE

def holt_lagged_daily(y_set_holt,y_to_test_holt,epoch_number,previous_epoch,lagged_daily): #,CC_deviation_holt=None):
    #y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    # This condition is in placed for the last epoch, if the last epoch is on the last day of the horizon 
    # then, there will be no more day to forecast after that, therefore, it should return and empty series
    # Since the holt function can not do this, we defined a conditional code, to solve this problem

    if lagged_daily >= epoch_number*R:
      # if the lag number is bigger or equal to the number of current epoch, then the forecast is zero since there is
      # no data to work with 
      holt_temp_df=y_to_test_holt.to_frame()
      holt_temp_df['holt_forecast']=0
      fcast1=holt_temp_df['holt_forecast']
      rmse1 = round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2)
      MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
      MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
      percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
    else:
      # applying the lagged epoch number to the training set
      y_to_train_holt = y_set_holt[0:(epoch_number*R)-lagged_daily]

      if (len(y_set_holt)-epoch_number*R) > 0:
        if len(y_to_train_holt)<2:
          fcast1=y_to_test_holt.to_frame()
          fcast1['holt_forecast']=max(y_to_train_holt)
          fcast1=fcast1['holt_forecast']
          rmse1 = round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2)
          MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
          MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
          # Caluclating the percentage bias only for the interval within each epoch
          # becuase this is the interval the inventory function is based on 
          percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
        else:
          number_of_predict_date=len(y_to_test_holt)+lagged_daily
          fit1 = Holt(y_to_train_holt).fit()
          fcast1 = fit1.forecast(number_of_predict_date).rename("Holt's linear trend")   
          # getting the days that are actually required for forecasting based on the array of "y_to_test_holt"
          fcast1=fcast1[-(len(y_to_test_holt)):]
          #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
          rmse1 = round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2)
          MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
          MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
          # Caluclating the percentage bias only for the interval within each epoch
          # becuase this is the interval the inventory function is based on 
          percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
      else:
        fcast1=y_to_val
        rmse1= round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2) 
        MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
        MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
        percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
      

    #if CC_deviation_holt is None:
    #  CC_dev_holt = 0
    #else:
    #  CC_dev_holt = CC_deviation_holt/100
    CC_dev_holt=0
    
    # removing the negative data 
    fcast1 [fcast1 <0]=0  

    return fcast1*(1+CC_dev_holt),rmse1,MAPE,percentage_bias_holt,MAE

from statsmodels.tsa.api import Holt
import scipy.stats as st
import sys

def holt_3(y_set_holt,y_to_train_holt,y_to_test_holt,period_info_holt): #,CC_deviation_holt=None):
    #y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    # This condition is in placed for the last epoch, if the last epoch is on the last day of the horizon 
    # then, there will be no more day to forecast after that, therefore, it should return and empty series
    # Since the holt function can not do this, we defined a conditional code, to solve this problem
    if holt_data=='normal':
      y_to_train_holt=y_to_train_holt
      y_to_test_holt=y_to_test_holt
    elif holt_data=='cumulative':
      y_to_train_holt=y_to_train_holt.cumsum()
      y_to_test_holt=y_to_test_holt.cumsum()
    else:
      print("Specify the type of data to be used by holt method: 'normal' or 'cumulative'")
      sys.exit()

    if period_info_holt==0:
      fcast1=y_to_test_holt.copy()
      fcast1.iloc[:]=0

      rmse1= round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2) 
      MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
      MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
      percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
    else:
    # total number of periods (based on the specified period_info) that the forecast should be made
      number_of_predict_date=math.ceil((len(y_set_holt)-period_info_holt*period_info)/period_info)
      if number_of_predict_date > 0:
        fit1 = Holt(y_to_train_holt).fit()
        fcast = fit1.forecast(number_of_predict_date).rename("Holt's linear trend") 
        
        # Putting the train data and forecast data (which is based on the period_info) into a one series
        complete_fcast1=(pd.concat([y_to_train_holt, fcast])).copy()
        # creating a copy of the y_set_holt to place the forecast data into
        temp_holt=y_set_holt.copy()

        #Looping through the copy of Y_set_data which is the daily data and place the average forecast data into it
        for period_in_period_info in range(1,len(complete_fcast1)):
          temp_holt[(period_in_period_info-1)*period_info:period_in_period_info*period_info]=complete_fcast1[period_in_period_info]/period_info

        # filtering the forecast data based on the index of the y_to_test_holt to get the correct number of days 
        fcast1=(pd.concat([y_to_test_holt,temp_holt], axis=1, join="inner")).iloc[:,1]

        rmse1 = round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2)
        MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
        MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
        # Caluclating the percentage bias only for the interval within each epoch
        # becuase this is the interval the inventory function is based on 
        percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)

      else:
        fcast1=y_to_test_holt
        rmse1= round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2) 
        MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
        MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
        percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
    
    # removing the negative data 
    fcast1 [fcast1 <0]=0  

    return fcast1,rmse1,MAPE,percentage_bias_holt,MAE

from statsmodels.tsa.api import Holt
import scipy.stats as st
import sys

def holt_4(y_set_holt,y_to_train_holt,y_to_test_holt): #,CC_deviation_holt=None):
    #y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    # This condition is in placed for the last epoch, if the last epoch is on the last day of the horizon 
    # then, there will be no more day to forecast after that, therefore, it should return and empty series
    # Since the holt function can not do this, we defined a conditional code, to solve this problem
    if holt_data=='normal':
      y_to_train_holt=y_to_train_holt
      y_to_test_holt=y_to_test_holt
    elif holt_data=='cumulative':
      y_to_train_holt=y_to_train_holt.cumsum()
      y_to_test_holt=y_to_test_holt.cumsum()
    else:
      print("Specify the type of data to be used by holt method: 'normal' or 'cumulative'")
      sys.exit()

    # if there is no information for the forecast to be trained, then the forecast is zero
    if len(y_to_train_holt)==0:
      fcast1=y_to_test_holt.copy()
      fcast1.iloc[:]=0

      rmse1= round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2) 
      MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
      MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
      percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
    
    # if there is only one day of information for the forecast to be trained, then the forecast is a constant value of that day
    elif len(y_to_train_holt)==1:

      fcast1=y_to_test_holt.copy()
      fcast1.iloc[:]=y_to_train_holt[0]

      rmse1= round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2) 
      MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
      MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
      percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)

    else:
    # total number of periods (based on the specified period_info) that the forecast should be made
      number_of_predict_date=len(y_set_holt)-len(y_to_train_holt)
      if number_of_predict_date > 0:
        fit1 = Holt(y_to_train_holt).fit()
        fcast = fit1.forecast(number_of_predict_date).rename("Holt's linear trend") 
        
        # Putting the train data and forecast data (which is based on the period_info) into a one series
        complete_fcast1=(pd.concat([y_to_train_holt, fcast], ignore_index=True)).copy()
        complete_fcast1.index=y_set.index
        # creating a copy of the y_set_holt to place the forecast data into
        #temp_holt=y_set_holt.copy()

        #Looping through the copy of Y_set_data which is the daily data and place the average forecast data into it
        #for period_in_period_info in range(1,len(complete_fcast1)):
        #  temp_holt[(period_in_period_info-1)*period_info:period_in_period_info*period_info]=complete_fcast1[period_in_period_info]/period_info

        # filtering the forecast data based on the index of the y_to_test_holt to get the correct number of days 
        fcast1=(pd.concat([y_to_test_holt,complete_fcast1], axis=1, join="inner")).iloc[:,1]

        rmse1 = round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2)
        MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
        MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
        # Caluclating the percentage bias only for the interval within each epoch
        # becuase this is the interval the inventory function is based on 
        percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)

      else:
        fcast1=y_to_test_holt
        rmse1= round(np.sqrt(((fcast1 - y_to_test_holt) ** 2).mean()), 2) 
        MAPE =  round(np.abs(((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100, 2)
        MAE =  round(np.abs((fcast1 - y_to_test_holt).mean()), 2)
        percentage_bias_holt=round((((fcast1 - y_to_test_holt)/y_to_test_holt).mean())*100,2)
    
    # removing the negative data 
    fcast1 [fcast1 <0]=0  

    return fcast1,rmse1,MAPE,percentage_bias_holt,MAE

"""### Myopic forecast"""

def myopic_forecast_method(y_to_test): #,CC_deviation_naive=None):
  #taking the maximum of last 2 epoch


  #Turning the column of naive forecast in the dumy dataframe into a series
  fcast1=y_to_test

  #if CC_deviation_naive is None:
  #  CC_dev_naive = 0
  #else:
  #  CC_dev_naive = CC_deviation_naive/100
  CC_dev_myopic=0
  
  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_myopic_1 = 0
  MAPE_myopic =  0
  MAE_myopic =  0

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_myopic=0

  # removing the negative data 
  fcast1 [fcast1 <0]=0  

  return fcast1*(1+CC_dev_myopic),rmse_myopic_1,MAPE_myopic,percentage_bias_myopic,MAE_myopic

"""### Naive method"""

def naive_forecast_method(y_to_train,y_to_test): #,CC_deviation_naive=None):
  #taking the maximum of last 2 epoch
  max_in_train=max(y_to_train)

  #creating a dataframe to include the naive forecast, primary reason for this command is to assign the appropriate index to the naive forecast
  naive_temp_df=y_to_test.to_frame()
  naive_temp_df['naive_forecast']=max_in_train

  #Turning the column of naive forecast in the dumy dataframe into a series
  naive_temp_sr=naive_temp_df['naive_forecast']

  #if CC_deviation_naive is None:
  #  CC_dev_naive = 0
  #else:
  #  CC_dev_naive = CC_deviation_naive/100
  CC_dev_naive=0
  
  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_naive_1 = round(np.sqrt(((naive_temp_sr - y_to_test) ** 2).mean()), 2)
  MAPE_naive =  round(np.abs(((naive_temp_sr - y_to_test)/y_to_test).mean())*100, 2)
  MAE_naive =  round(np.abs((naive_temp_sr - y_to_test).mean()), 2)

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_naive=round((((naive_temp_sr - y_to_test)/y_to_test).mean())*100,2)

  # removing the negative data 
  naive_temp_sr [naive_temp_sr <0]=0  

  return naive_temp_sr*(1+CC_dev_naive),rmse_naive_1,MAPE_naive,percentage_bias_naive,MAE_naive

def naive_forecast_method_lagged(y_set_naive,y_to_test,epoch_number_naive,lagged_epoch): #,CC_deviation_naive=None):
  #taking the maximum of last 2 epoch
  
  if lagged_epoch >= epoch_number_naive:
  #creating a dataframe to include the naive forecast, primary reason for this command is to assign the appropriate index to the naive forecast
    naive_temp_df=y_to_test.to_frame()
    naive_temp_df['naive_forecast']=0
  else:
    y_to_train = y_set_naive[max(0,(epoch_number_naive-lagged_epoch)-2)*R:(epoch_number_naive-lagged_epoch)*R]

    max_in_train=max(y_to_train)

  #creating a dataframe to include the naive forecast, primary reason for this command is to assign the appropriate index to the naive forecast
    naive_temp_df=y_to_test.to_frame()
    naive_temp_df['naive_forecast']=max_in_train

  #Turning the column of naive forecast in the dumy dataframe into a series
  naive_temp_sr=naive_temp_df['naive_forecast']

  #if CC_deviation_naive is None:
  #  CC_dev_naive = 0
  #else:
  #  CC_dev_naive = CC_deviation_naive/100
  CC_dev_naive=0

  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_naive_1 = round(np.sqrt(((naive_temp_sr - y_to_test) ** 2).mean()), 2)
  MAPE_naive =  round(np.abs(((naive_temp_sr - y_to_test)/y_to_test).mean())*100, 2)
  MAE_naive =  round(np.abs((naive_temp_sr - y_to_test).mean()), 2)

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_naive=round((((naive_temp_sr - y_to_test)/y_to_test).mean())*100,2)

  # removing the negative data 
  naive_temp_sr [naive_temp_sr <0]=0  

  return naive_temp_sr*(1+CC_dev_naive),rmse_naive_1,MAPE_naive,percentage_bias_naive,MAE_naive

def naive_forecast_method_lagged_daily(y_set_naive,y_to_test,epoch_number_naive,lagged_daily): #,CC_deviation_naive=None):
  #taking the maximum of last 2 epoch
  
  if lagged_daily >= epoch_number_naive*R:
  #creating a dataframe to include the naive forecast, primary reason for this command is to assign the appropriate index to the naive forecast
    naive_temp_df=y_to_test.to_frame()
    naive_temp_df['naive_forecast']=0
  else:
    y_to_train = y_set_naive[max(0,(epoch_number_naive-2)*R-lagged_daily):(epoch_number_naive*R-lagged_daily)]

    max_in_train=max(y_to_train)

  #creating a dataframe to include the naive forecast, primary reason for this command is to assign the appropriate index to the naive forecast
    naive_temp_df=y_to_test.to_frame()
    naive_temp_df['naive_forecast']=max_in_train

  #Turning the column of naive forecast in the dumy dataframe into a series
  naive_temp_sr=naive_temp_df['naive_forecast']

  #if CC_deviation_naive is None:
  #  CC_dev_naive = 0
  #else:
  #  CC_dev_naive = CC_deviation_naive/100
  CC_dev_naive=0

  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_naive_1 = round(np.sqrt(((naive_temp_sr - y_to_test) ** 2).mean()), 2)
  MAPE_naive =  round(np.abs(((naive_temp_sr - y_to_test)/y_to_test).mean())*100, 2)
  MAE_naive =  round(np.abs((naive_temp_sr - y_to_test).mean()), 2)

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_naive=round((((naive_temp_sr - y_to_test)/y_to_test).mean())*100,2)

  # removing the negative data 
  naive_temp_sr [naive_temp_sr <0]=0  

  return naive_temp_sr*(1+CC_dev_naive),rmse_naive_1,MAPE_naive,percentage_bias_naive,MAE_naive

def naive_forecast_method_replenished(y_to_train,y_to_test): #,CC_deviation_naive=None):
  #taking the maximum of last 2 epoch
  max_in_train=(max(y_to_train))/period_info

  #creating a dataframe to include the naive forecast, primary reason for this command is to assign the appropriate index to the naive forecast
  naive_temp_df=y_to_test.to_frame()
  naive_temp_df['naive_forecast']=max_in_train

  #Turning the column of naive forecast in the dumy dataframe into a series
  naive_temp_sr=naive_temp_df['naive_forecast']

  #if CC_deviation_naive is None:
  #  CC_dev_naive = 0
  #else:
  #  CC_dev_naive = CC_deviation_naive/100
  CC_dev_naive=0
  
  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_naive_1 = round(np.sqrt(((naive_temp_sr - y_to_test) ** 2).mean()), 2)
  MAPE_naive =  round(np.abs(((naive_temp_sr - y_to_test)/y_to_test).mean())*100, 2)
  MAE_naive =  round(np.abs((naive_temp_sr - y_to_test).mean()), 2)

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_naive=round((((naive_temp_sr - y_to_test)/y_to_test).mean())*100,2)

  # removing the negative data 
  naive_temp_sr [naive_temp_sr <0]=0  

  return naive_temp_sr*(1+CC_dev_naive),rmse_naive_1,MAPE_naive,percentage_bias_naive,MAE_naive

def naive_forecast_method_replenished_variable(y_to_train,y_to_test): #,CC_deviation_naive=None):
  #taking the maximum of last 2 epoch
  if len(y_to_train)==0:
    #creating a dataframe to include the naive forecast, primary reason for this command is to assign the appropriate index to the naive forecast
    naive_temp_df=y_to_test.to_frame()
    naive_temp_df['naive_forecast']=0

    #Turning the column of naive forecast in the dumy dataframe into a series
    naive_temp_sr=naive_temp_df['naive_forecast']
  else:
    max_in_train=max(y_to_train)

    #creating a dataframe to include the naive forecast, primary reason for this command is to assign the appropriate index to the naive forecast
    naive_temp_df=y_to_test.to_frame()
    naive_temp_df['naive_forecast']=max_in_train

    #Turning the column of naive forecast in the dumy dataframe into a series
    naive_temp_sr=naive_temp_df['naive_forecast']

  #if CC_deviation_naive is None:
  #  CC_dev_naive = 0
  #else:
  #  CC_dev_naive = CC_deviation_naive/100
  CC_dev_naive=0
  
  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_naive_1 = round(np.sqrt(((naive_temp_sr - y_to_test) ** 2).mean()), 2)
  MAPE_naive =  round(np.abs(((naive_temp_sr - y_to_test)/y_to_test).mean())*100, 2)
  MAE_naive =  round(np.abs((naive_temp_sr - y_to_test).mean()), 2)

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_naive=round((((naive_temp_sr - y_to_test)/y_to_test).mean())*100,2)

  # removing the negative data 
  naive_temp_sr [naive_temp_sr <0]=0  

  return naive_temp_sr*(1+CC_dev_naive),rmse_naive_1,MAPE_naive,percentage_bias_naive,MAE_naive

"""### SIR forecasting method"""

# Forecasting Hospitalization with SIR model
# IMPORTANT NOTE: USER INPUT
# ==============================================================================================

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
SIR_forecast_df=pd.read_csv('forecast SIR without cumulative-complete 18 epochs.csv')


#setting the date as the index of the dataframe
SIR_forecast_df=SIR_forecast_df.set_index(y.index).drop(columns='Date')
#rounding up the value of forecast in the data frame 
#SIR_forecast_df=np.ceil(SIR_forecast_df)

#this function select the desired interval from the data frame which includes all the forecast for the SIR model
#later if the user wants to include the dynamic coefficient of consumption, coeff can be be changed
def SIR_forecast_model(epoch_number,y_to_test,CC_deviation_SIR=None,epoch_number_2=None):
  if CC_deviation_SIR is None:
    CC = protocol_coef
  else:
    CC = protocol_coef*(1+CC_deviation_SIR/100)
  
  if epoch_number_2 is None:
    epoch_number_2 = epoch_number

# this condition is for the first period for those period_info that do not have info 
  if epoch_number==0:

    forecast_sir_sr=SIR_forecast_df.iloc[:,epoch_number]
    forecast_sir_sr.iloc[:]=0
    forecast_sir_sr=(forecast_sir_sr[(epoch_number_2)*R:(epoch_number_2)*R+(R+L)])* CC

  elif epoch_number>0:

    forecast_sir_sr=SIR_forecast_df.iloc[:,epoch_number]
    forecast_sir_sr=(forecast_sir_sr[(epoch_number_2)*R:(epoch_number_2)*R+(R+L)])* CC



  
  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_sir_1 = round(np.sqrt(((forecast_sir_sr - y_to_test) ** 2).mean()), 2)
  MAPE_SIR =  round(np.abs(((forecast_sir_sr - y_to_test)/y_to_test).mean())*100, 2)
  MAE_SIR =  round(np.abs((forecast_sir_sr - y_to_test).mean()), 2)

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_SIR=round((((forecast_sir_sr - y_to_test)/y_to_test).mean())*100,2)

  # removing the negative data 
  forecast_sir_sr [forecast_sir_sr <0]=0  
  
  return forecast_sir_sr,rmse_sir_1,MAPE_SIR,percentage_bias_SIR,MAE_SIR

#this function select the desired interval from the data frame which includes all the forecast for the SIR model
#later if the user wants to include the dynamic coefficient of consumption, coeff can be be changed
def SIR_forecast_model_devited(epoch_number,y_to_test,CC_deviation_SIR):
  
  CC = protocol_coef*(CC_deviation_SIR/100)
  epoch_number_2=epoch_number
# this condition is for the first period for those period_info that do not have info 
  if epoch_number==0:

    forecast_sir_sr=SIR_forecast_df.iloc[:,epoch_number]
    forecast_sir_sr.iloc[:]=0
    forecast_sir_sr=(forecast_sir_sr[(epoch_number_2)*R:(epoch_number_2)*R+(R+L)])* CC

  elif epoch_number>0:

    forecast_sir_sr=SIR_forecast_df.iloc[:,epoch_number]
    forecast_sir_sr=(forecast_sir_sr[(epoch_number_2)*R:(epoch_number_2)*R+(R+L)])* CC



  
  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_sir_1 = round(np.sqrt(((forecast_sir_sr - y_to_test) ** 2).mean()), 2)
  MAPE_SIR =  round(np.abs(((forecast_sir_sr - y_to_test)/y_to_test).mean())*100, 2)
  MAE_SIR =  round(np.abs((forecast_sir_sr - y_to_test).mean()), 2)

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_SIR=round((((forecast_sir_sr - y_to_test)/y_to_test).mean())*100,2)

  # removing the negative data 
  forecast_sir_sr [forecast_sir_sr <0]=0  
  
  return forecast_sir_sr,rmse_sir_1,MAPE_SIR,percentage_bias_SIR,MAE_SIR

#this function select the desired interval from the data frame which includes all the forecast for the SIR model
#later if the user wants to include the dynamic coefficient of consumption, coeff can be be changed
def SIR_forecast_model_lagged(epoch_number,y_to_test,lagged_epoch,CC_deviation_SIR=None):
  if CC_deviation_SIR is None:
    CC = protocol_coef
  else:
    CC = protocol_coef*(1+CC_deviation_SIR/100)
  #CC=protocol_coef*(1+CC_deviation/100)
  
  if lag >= epoch_number : 
  #creating a dataframe to include the sir forecast, primary reason for this command is to assign the appropriate index to the naive forecast
    SIR_temp_df=y_to_test.to_frame()
    SIR_temp_df['SIR_forecast']=0
    forecast_sir_sr= SIR_temp_df['SIR_forecast']
  else:
    forecast_sir_sr=SIR_forecast_df.iloc[:,epoch_number-lagged_epoch]
    forecast_sir_sr=(forecast_sir_sr[(epoch_number)*R:(epoch_number)*R+(R+L)])* CC
  
  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_sir_1 = round(np.sqrt(((forecast_sir_sr - y_to_test) ** 2).mean()), 2)
  MAPE_SIR =  round(np.abs(((forecast_sir_sr - y_to_test)/y_to_test).mean())*100, 2)
  MAE_SIR =  round(np.abs((forecast_sir_sr - y_to_test).mean()), 2)

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_SIR=round((((forecast_sir_sr - y_to_test)/y_to_test).mean())*100,2)

  # removing the negative data 
  forecast_sir_sr [forecast_sir_sr <0]=0  
  
  return forecast_sir_sr,rmse_sir_1,MAPE_SIR,percentage_bias_SIR,MAE_SIR

#this function select the desired interval from the data frame which includes all the forecast for the SIR model
#later if the user wants to include the dynamic coefficient of consumption, coeff can be be changed
def SIR_forecast_model_lagged_daily(epoch_number,y_to_test,lagged_daily,CC_deviation_SIR=None):
  if CC_deviation_SIR is None:
    CC = protocol_coef
  else:
    CC = protocol_coef*(1+CC_deviation_SIR/100)
  #CC=protocol_coef*(1+CC_deviation/100)
  
  if lagged_daily >= epoch_number*R : 
  #creating a dataframe to include the sir forecast, primary reason for this command is to assign the appropriate index to the naive forecast
    SIR_temp_df=y_to_test.to_frame()
    SIR_temp_df['SIR_forecast']=0
    forecast_sir_sr= SIR_temp_df['SIR_forecast']
  else:
    # this code is to evaluate the last epoch,
    if epoch_number == math.ceil(len(y)/R):
      forecast_sir_sr = SIR_forecast_df.iloc[:,-1-lagged_daily]
      forecast_sir_sr =(forecast_sir_sr[(epoch_number)*R:(epoch_number)*R+(R+L)])* CC
    else:
      forecast_sir_sr = SIR_forecast_df.iloc[:,epoch_number*R-lagged_daily]
      forecast_sir_sr = (forecast_sir_sr[(epoch_number)*R:(epoch_number)*R+(R+L)])* CC
    
  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_sir_1 = round(np.sqrt(((forecast_sir_sr - y_to_test) ** 2).mean()), 2)
  MAPE_SIR =  round(np.abs(((forecast_sir_sr - y_to_test)/y_to_test).mean())*100, 2)
  MAE_SIR =  round(np.abs((forecast_sir_sr - y_to_test).mean()), 2)

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_SIR=round((((forecast_sir_sr - y_to_test)/y_to_test).mean())*100,2)

  # removing the negative data 
  forecast_sir_sr [forecast_sir_sr <0]=0  
  
  return forecast_sir_sr,rmse_sir_1,MAPE_SIR,percentage_bias_SIR,MAE_SIR

#this function select the desired interval from the data frame which includes all the forecast for the SIR model
#later if the user wants to include the dynamic coefficient of consumption, coeff can be be changed
def SIR_forecast_model_variable_period_info(period_info_var,y_to_test,epoch_number):


# this condition is for the first period for those period_info that do not have info 
  if period_info_var==0:

    forecast_sir_sr=y_to_test.copy()
    forecast_sir_sr.iloc[:]=0

  elif period_info_var>0:

    forecast_sir_sr=SIR_forecast_df.iloc[:,period_info_var]
    forecast_sir_sr=(forecast_sir_sr[(epoch_number)*R:(epoch_number)*R+(R+L)])

  forecast_sir_sr.index=y_to_test.index

  
  #computing the RSME of the forecast
  #Note: this error at the current epoch has not been realized yet, hence it should be used in the next epoch calculation
  rmse_sir_1 = round(np.sqrt(((forecast_sir_sr - y_to_test) ** 2).mean()), 2)
  MAPE_SIR =  round(np.abs(((forecast_sir_sr - y_to_test)/y_to_test).mean())*100, 2)
  MAE_SIR =  round(np.abs((forecast_sir_sr - y_to_test).mean()), 2)

  # Caluclating the percentage bias only for the interval within each epoch
  # becuase this is the interval the inventory function is based on 
  percentage_bias_SIR=round((((forecast_sir_sr - y_to_test)/y_to_test).mean())*100,2)

  # removing the negative data 
  forecast_sir_sr [forecast_sir_sr <0]=0  
  
  return forecast_sir_sr,rmse_sir_1,MAPE_SIR,percentage_bias_SIR,MAE_SIR

"""## Periodic Review Function"""

#PERIODIC REVIEW SYSTEM
#==============================================================================================================
def peridoic_review(y_previous_epoch,inventory_data_pre,prediction_data,epoch_number):


   
  #INVENTORY LEVEL AND SHORTAGE
  #========================================================================================
  #This section compute the inventory level at the end of last epoch and shortage during the the last epoch
  #The order that is placed at [math.ceil(L/R)] ago is to be arrive between this epoch_number and the last one
  
  # this code should be re-written as:
  #if epoch_number-math.ceil(L/R) => 0 :
  #  order_arrival=inventory_data_pre['order'][epoch_number-math.ceil(L/R)]
  #else:
  #  order_arrival=0
    
  order_arrival=inventory_data_pre['order'][max(0,epoch_number-math.ceil(L/R))]

  #To include the arrival of the orders that arrives between the two epochs, we break the inventory level into two parts: before and after
  # (L-math.floor(L/R)*R) gives us the number of days after the previous epoch that the inventory system has recieved the order which was placed at 
  # the order which was placed at [math.ceil(L/R)] ago.
  inventory_before_arrival = max(0,(inventory_data_pre['current'][epoch_number-1]-y_previous_epoch[:(L-math.floor(L/R)*R)].sum()))
  inventory_after_arrival  = max(0,((inventory_before_arrival+order_arrival)-y_previous_epoch[(L-math.floor(L/R)*R):].sum()))

  shortage_before_arrival  = max(0,(y_previous_epoch[:(L-math.floor(L/R)*R)].sum()-inventory_data_pre['current'][epoch_number-1]))
  shortage_after_arrival   = max(0,(y_previous_epoch[(L-math.floor(L/R)*R):].sum()-(inventory_before_arrival+order_arrival)))

  shortage = math.ceil(shortage_before_arrival  + shortage_after_arrival)
  current = inventory_after_arrival
  
  #SAFETY STOCK
  #========================================================================================
  z_service_level = st.norm.ppf(service_level)
  #the way that model was coded, the std deviation of each row represents the std of the previous epoch which is what we want
  #Use one of these two for standard deviation:
  #      std of previous epoch
  #      std from beginning
  
  try:
    type_std_system=type_std_system
  except NameError:
    type_std_system='std of previous epoch'
  
  std_system = inventory_data_pre[type_std_system][epoch_number-1]
  
  safety_stock = z_service_level * std_system * np.sqrt(R+L)

  #Actual order
  #========================================================================================
  # This section will store the actual quantity that needs to be ordered without any of the above constraints:
  actual_order=math.ceil(max(prediction_data[0].sum() + safety_stock - current,0))

  #ORDER PLACEMENT
  #========================================================================================
  # This condition makes sure the system will not place an order that will be recieved outside of the time horizon. 
  if len(y)-(epoch_number*R) > L:
    if actual_order > 0:
      #supplier's constraint
      order = min(suppliers_capacity_max, max(suppliers_capacity_min,actual_order ))
      
      # orders have to be in terms of pallet quantity
      order = (math.ceil(order/item_in_pallet)) * item_in_pallet
    else:
      order = 0
    #inventory capacity constraint
    #if (order+current-prediction_data[0][:L].sum())>inventory_capactiy:
    #  order=0
  else:
    order = 0 
  
  
  #COSTS:
  #========================================================================================
  # holding cost:
  holding_temp_df=y_previous_epoch.to_frame()
  holding_temp_df['inventory at the end of day']=0
  holding_temp_df['inventory at the end of day'][0]=max(0,inventory_data_pre['current'][epoch_number-1]-holding_temp_df['daily_consumption'][0])
  for k in range(1,len(y_previous_epoch)):
    if k != ((L-math.floor(L/R)*R)-1):
      holding_temp_df['inventory at the end of day'][k]=max(0,holding_temp_df['inventory at the end of day'][k-1]-holding_temp_df['daily_consumption'][k])
    else:
      holding_temp_df['inventory at the end of day'][k]=max(0,holding_temp_df['inventory at the end of day'][k-1]
                                                            +order_arrival-holding_temp_df['daily_consumption'][k])

  holding_temp_df['daily holding cost']=holding_temp_df['inventory at the end of day']*c_h
  holding_cost_previous_epoch=holding_temp_df['daily holding cost'].sum()
  #====================================================================
  # ordering cost
  if order > 0:
    ordering_cost_previous_epoch = c_o
  else:
    ordering_cost_previous_epoch = 0
  #====================================================================
  # unit cost:
  unit_cost_previous_epoch = order * c_u
  #====================================================================
  #shortage cost:
  shortage_cost_previous_epoch = shortage * c_s
  #====================================================================
  #total cost
  real_cost_previous_epoch = (holding_cost_previous_epoch + 
                               ordering_cost_previous_epoch + 
                               unit_cost_previous_epoch)
  #====================================================================
  #total cost
  total_cost_previous_epoch = (real_cost_previous_epoch +  
                               shortage_cost_previous_epoch)
  #===========================================================================================
  #computing the standard dev of system from the beginning 
  std_from_begin=(inventory_data_pre['std of previous epoch'].sum()+prediction_data[1])/epoch_number

  inventory_data_pre.loc[epoch_number]=[y_previous_epoch.sum(),
                                        round(current,2),
                                        round(order,2),
                                        round(shortage,2),
                                        round(prediction_data[1],2),
                                        round(std_from_begin,2),
                                        holding_cost_previous_epoch,
                                        ordering_cost_previous_epoch,
                                        unit_cost_previous_epoch,
                                        shortage_cost_previous_epoch,
                                        total_cost_previous_epoch,
                                        real_cost_previous_epoch,
                                        prediction_data[4],
                                        prediction_data[2],
                                        prediction_data[3],
                                        actual_order]


  return inventory_data_pre

"""## Inventory Simulation

### Inventory Sim with Naive
"""

# PERIODIC REVIEW SYSTEM with NAIVE forecasting
#====================================================================================================================
# creating a loop to through each epoch and make a forecast calculate the inventory level and move to the next epoch 
# Note, we will not make any forecast or decision for epoch=0

# in order to find how many epoch we have in each data-set, devide the number of days in your data set by the interval between eac epoch
# at this point we have igonre the remaining days of [len(y)-(math.floor((len(y)-(R+L))/R))]
# It is possible to include those remaining days by simple if statement
inventory_data_periodic_naive = inventory_data.copy()
inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)

#choosing the round down (floor) will exclude the last epoch which another epoch has been added; 
# This way even if the the devision gives the rounded number the last epoch still will be included.

for i in range(1,math.ceil(len(y)/R)+1):
  y_to_train = y[max(0,i-2)*R:(i)*R]

  y_to_val = y[(i)*R:(i)*R+(R+L)]
  #y_set=y[0:(i+1)*R+(R+L)]
  prediction = naive_forecast_method(y_to_train,y_to_val)
  y_previous_epoch=y[(i-1)*R:(i)*R]
  epoch_no=i
  inventory_data_periodic_naive=peridoic_review(y_previous_epoch,inventory_data_periodic_naive,prediction,epoch_no)
inventory_data_periodic_naive

"""### Inventory Sim with Holt"""

# PERIODIC REVIEW SYSTEM with HOLT forecasting
#====================================================================================================================
# creating a loop to through each epoch and make a forecast calculate the inventory level and move to the next epoch 
# Note, we will not make any forecast or decision for epoch=0

# in order to find how many epoch we have in each data-set, devide the number of days in your data set by the interval between eac epoch
# at this point we have igonre the remaining days of [len(y)-(math.floor((len(y)-(R+L))/R))]
# It is possible to include those remaining days by simple if statement
inventory_data_periodic_Holt = inventory_data
inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

#hospitalization_data_type='daily_consumption'

try:
  hospitalization_data_type=hospitalization_data_type
except NameError:
  hospitalization_data_type='daily_consumption'

if hospitalization_data_type=='daily_consumption':
  y_holt=y
else:
  y_holt=y_cumulative


for i in range(1,math.ceil(len(y)/R)+1):
  y_to_train = y_holt[0:(i)*R]
  y_to_val = y_holt[(i)*R:(i)*R+(R+L)]
  y_set=y_holt[0:(i)*R+(R+L)]
  y_previous_epoch_holt=y_holt[(i-1)*R:(i)*R]
  prediction = holt(y_set, y_to_train,y_to_val,i,y_previous_epoch_holt)
  y_previous_epoch=y[(i-1)*R:(i)*R]
  epoch_no=i
  inventory_data_periodic_Holt=peridoic_review(y_previous_epoch,inventory_data_periodic_Holt,prediction,epoch_no)
inventory_data_periodic_Holt

i=2
y_to_train = y_holt[0:(i)*R]
y_to_val = y_holt[(i)*R:(i)*R+(R+L)]
y_set=y_holt[0:(i)*R+(R+L)]
y_previous_epoch_holt=y_holt[(i-1)*R:(i)*R]
prediction = holt(y_set, y_to_train,y_to_val,i,y_previous_epoch_holt)
y_previous_epoch=y[(i-1)*R:(i)*R]
epoch_no=i

"""### Inventory Sim with SIR"""

# PERIODIC REVIEW SYSTEM with SIR forecasting model
#====================================================================================================================
#for this model to work, the csv file needs the follwoing conditions:
#     1.the index should be the desired date of studies
#     2.each column should contain the forecast data for the entire horizen 
#     3.the columns should represent the epochs in the incremental fashion, (ex: column1==> epoch n, column2==> epoch n+1, column3==> epoch n+3, .... )

# creating a loop to through each epoch and make a forecast calculate the inventory level and move to the next epoch 
# Note, we will not make any forecast or decision for epoch=0

# in order to find how many epoch we have in each data-set, devide the number of days in your data set by the interval between eac epoch
# at this point we have igonre the remaining days of [len(y)-(math.floor((len(y)-(R+L))/R))]
# It is possible to include those remaining days by simple if statement
inventory_data_periodic_SIR = inventory_data.copy()
inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)




for i in range(1,math.ceil(len(y)/R)+1):
  y_to_train = y[0:(i)*R]
  y_to_val = y[(i)*R:(i)*R+(R+L)]
  y_set=y[0:(i)*R+(R+L)]
  prediction = SIR_forecast_model(i,y_to_val)
  y_previous_epoch=y[(i-1)*R:(i)*R]
  epoch_no=i
  inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction,epoch_no)
inventory_data_periodic_SIR

y_to_val

SIR_forecast_df.iloc[:,3]

prediction[0]

"""## Noise creator functions

### Noise creator on the overall consumption
"""

def noise_creator_on_overall_consumption(data_without_noise,mean_type):
  
  #Creating an empty series to put the data with noise in
  data_with_noise=data_without_noise[0:0]

  #computing the max and mean of the data once to not to compute it everytime
  max_of_data_without_noise=data_without_noise.max()
  mean_of_data_without_noise=data_without_noise.mean()

  #depending on the type of mean in the system, different noise is added to the data
  if mean_type=='mean_each_epoch':
    
    #dividing the length of data by the epoch period will provide with the number of epochs in the horizon
    #we are rounding it to accomodate for days after the last epoch
    for i in range(math.ceil(len(data_without_noise)/R)):
      #temproray seires to place data with noise in
      temp_data=data_without_noise[i*R:(i+1)*R]
     
      #taking the mean of each epoch
      mean_val=temp_data.mean()
     
      #computing the sigma value based on the specified mean 
      sigma_noise=mean_val*percentage_of_variation
      
      #creating noise with normal distribution
      noise=np.random.normal(deviation_percentage_from_mean * mean_val , sigma_noise, temp_data.shape) 
      temp_data=temp_data+noise
     
      #Since the demand cannot be negative, we need to replace them with zero 
      temp_data[temp_data<0]=0
     
      #appending the new set into the previous epochs
      data_with_noise=data_with_noise.append(temp_data)
  
  elif mean_type=='max_each_epoch':
    #in this method we choose the maximum of each epoch as the mean for the noise
    #this method is more conservative than the preivous methods
    for i in range(math.ceil(len(data_without_noise)/R)):
      temp_data=data_without_noise[i*R:(i+1)*R]
     
      #taking the max of each epoch
      max_val=temp_data.max()
      
      sigma_noise=max_val*percentage_of_variation
      noise=np.random.normal(deviation_percentage_from_mean * max_val, sigma_noise, temp_data.shape) 
      temp_data=temp_data+noise
      temp_data[temp_data<0]=0

      data_with_noise=data_with_noise.append(temp_data)

  elif mean_type=='max_all_epoch':

    #in this method we choose the overall maximum of data as the mean for the noise
    #this method is the most conservative among all the preivous methods
        
    data_with_noise = data_without_noise[0:0]
    sigma_noise = max_of_data_without_noise * percentage_of_variation
    noise=np.random.normal(deviation_percentage_from_mean * max_of_data_without_noise, sigma_noise, data_without_noise.shape)
    data_with_noise=data_without_noise+noise
    data_with_noise[data_with_noise<0]=0
  
  elif mean_type=='mean_all_epoch':

    #in this method we choose the overall mean of data as the mean for the noise
        
    data_with_noise = data_without_noise[0:0]
    sigma_noise = mean_of_data_without_noise * percentage_of_variation
    noise=np.random.normal(deviation_percentage_from_mean * mean_of_data_without_noise, sigma_noise, data_without_noise.shape)
    data_with_noise=data_without_noise+noise
    data_with_noise[data_with_noise<0]=0
  
  
  return data_with_noise.apply(np.ceil)

"""### Noise creator on CC with SNR:"""

def noise_creator_with_SNR(daily_hospitalization_for_noise_addition,specified_CC):
  
  #Creating an empty series to put the data with noise in
  simulated_data=daily_hospitalization_for_noise_addition[0:0]

  #depending on the type of mean in the system, different noise is added to the data
  if noise_addition=='no':
    
    simulated_data = daily_hospitalization_for_noise_addition

  elif noise_addition == 'yes':

    simulated_data = daily_hospitalization_for_noise_addition[0:0]
    
    # calcualting the standard deviation for noise creation based on the specified signal to noise ratio
    sigma_noise=specified_CC/SNR
    
    # based on the std, generate the noise for coefficient of consumption for each day in our data set
    noise_for_CC=np.random.normal(0, sigma_noise, daily_hospitalization_for_noise_addition.shape)

    # adding the noise to CC of each day 
    daily_CC_with_noise=noise_for_CC+specified_CC

    #daily_CC_with_noise_devi=noise_for_CC+specified_CC*(1+dev)

    # simulating data with the CC that has added noise
    simulated_data=daily_hospitalization_for_noise_addition * daily_CC_with_noise

    # removing the negative data 
    simulated_data [simulated_data <0]=0  

    simulated_data=simulated_data.replace(0,1)
  
  return simulated_data.apply(np.ceil)

"""### Noise creator on CC with SNR and CC deviation:"""

def noise_creator_with_SNR_CCdev_1(daily_hospitalization_for_noise_addition,simulated_SNR,specified_CC):
  
  #Creating an empty series to put the data with noise in
  simulated_data=daily_hospitalization_for_noise_addition[0:0]

  #depending on the type of mean in the system, different noise is added to the data
  if noise_addition=='no':
    
    simulated_data = daily_hospitalization_for_noise_addition

  elif noise_addition == 'yes':

    simulated_data = daily_hospitalization_for_noise_addition[0:0]
    
    # calcualting the standard deviation for noise creation based on the specified signal to noise ratio
    sigma_noise=specified_CC/SNR
    
    # based on the std, generate the noise for coefficient of consumption for each day in our data set
    noise_for_CC=np.random.normal(0, sigma_noise, daily_hospitalization_for_noise_addition.shape)
  
  return noise_for_CC

def noise_creator_with_SNR_CCdev_2(daily_hospitalization_for_noise_addition,specified_CC,CCdev,simulated_noise):
  
  #Creating an empty series to put the data with noise in
  simulated_data=daily_hospitalization_for_noise_addition[0:0]

  #depending on the type of mean in the system, different noise is added to the data
  if noise_addition=='no':
    
    simulated_data = daily_hospitalization_for_noise_addition

  elif noise_addition == 'yes':

    simulated_data = daily_hospitalization_for_noise_addition[0:0]
    
    # UN-deviated simulated data
    #==========================================
    # using previously simulated noise, adding the noise to CC of each day 
    daily_CC_with_noise = simulated_noise + specified_CC

    # simulating data with the CC that has added noise
    simulated_data = daily_hospitalization_for_noise_addition * daily_CC_with_noise
    
    # deviated simulated data
    #==========================================
    # adding the computed noise to the devaited CC
    # Method I:
    #daily_CC_with_noise_CCdev = simulated_noise + specified_CC*(1+CCdev/100)
    # Method II:
    daily_CC_with_noise_CCdev = (simulated_noise + specified_CC)*(1+CCdev/100)

    # 0 to 200 % 
    daily_CC_with_noise_CCdev = (simulated_noise + specified_CC)*(CCdev/100)

    #simulating adjusted data with devaited CC that has added noise
    simulated_data_dev = daily_hospitalization_for_noise_addition * daily_CC_with_noise_CCdev

    # removing the negative data 
    simulated_data [simulated_data <0]=0  
    simulated_data_dev [simulated_data_dev <0]=0  
    # replacing the 0 demand with 1 to calcualted the relative error
    simulated_data=simulated_data.replace(0,1)
    #simulated_data_dev=simulated_data_dev.replace(0,1)
  
  return simulated_data.apply(np.ceil),simulated_data_dev.apply(np.ceil)

def noise_creator_for_basic_scenario(daily_hospitalization_for_noise_addition,simulated_SNR,specified_CC):
  
  #Creating an empty series to put the data with noise in
  simulated_data=daily_hospitalization_for_noise_addition[0:0]

  #depending on the type of mean in the system, different noise is added to the data
  if noise_addition=='no':
    
    simulated_data = daily_hospitalization_for_noise_addition

  elif noise_addition == 'yes':

    simulated_data = daily_hospitalization_for_noise_addition[0:0]
    
    # calcualting the standard deviation for noise creation based on the specified signal to noise ratio
    sigma_noise=specified_CC/simulated_SNR
    
    # based on the std, generate the noise for coefficient of consumption for each day in our data set
    noise_for_CC=np.random.normal(0, sigma_noise, daily_hospitalization_for_noise_addition.shape)
    
    simulated_data=daily_hospitalization_for_noise_addition *(noise_for_CC+specified_CC)

    # removing the negative data 
    simulated_data [simulated_data <0]=0  

    simulated_data=simulated_data.replace(0,1)


  return simulated_data.apply(np.ceil), noise_for_CC

"""
## Plots Comarison function"""

#protocol_coef_variation=[3,3.5,4,4.5,5]
#service_level_variation

def comparison_plots(data_for_comparison):

  if len(service_level_variation) ==1:
    df_comparison_coeff_chart_shortage=data_for_comparison[['Forecast method','Shortage','Protocol coefficient']]
    df_comparison_coeff_real_cost=data_for_comparison[['Forecast method','Real cost','Protocol coefficient']]

    plt.figure(figsize=(14,7))
    

    plt.subplot(1, 2, 1)
    #this part plots different Protocol coefficient and their related shortages
    #plt.figure(figsize=(17,8.5))
    chart=sns.barplot(x="Protocol coefficient", 
                y="Shortage", 
                hue="Forecast method", 
                data=df_comparison_coeff_chart_shortage,palette=["C0", "C1", "C2"])

    chart.set_xticklabels(chart.get_xmajorticklabels(), fontsize = 14)
    #chart.set_yticklabels(chart.get_ymajorticklabels(), fontsize = 12)
    plt.ylabel("Shortage", size=16)
    plt.xlabel("Protocol coefficient", size=16)
    plt.title("Total number of shortage with different protocol coefficient", size=18)
    plt.legend(fontsize=13)

    plt.subplot(1, 2, 2)
    #this part plots different Protocol coefficient and their related real costs
    #plt.figure(figsize=(17,8.5))
    chart=sns.barplot(x="Protocol coefficient", 
                y="Real cost", 
                hue="Forecast method", 
                data=df_comparison_coeff_real_cost,palette=["C0", "C1", "C2"])

    chart.set_xticklabels(chart.get_xmajorticklabels(), fontsize = 14)
    #chart.set_yticklabels(chart.get_ymajorticklabels(), fontsize = 12)
    plt.ylabel("Real cost", size=16)
    plt.xlabel("Protocol coefficient", size=16)
    plt.title("Real cost with different protocol coefficient", size=18)
    plt.legend(fontsize=13)

  elif len(protocol_coef_variation) ==1:
    df_comparison_SC_chart_shortage=data_for_comparison[['Forecast method','Shortage','Service level - specified']]
    df_comparison_SC_real_cost=data_for_comparison[['Forecast method','Real cost','Service level - specified']]

    plt.figure(figsize=(14,7))
    

    plt.subplot(1, 2, 1)
    #this part plots different Protocol coefficient and their related shortages
    #plt.figure(figsize=(17,8.5))
    chart=sns.barplot(x="Service level - specified", 
              y="Shortage", 
              hue="Forecast method", 
              data=df_comparison_SC_chart_shortage,palette=["C0", "C1", "C2"])

    chart.set_xticklabels(chart.get_xmajorticklabels(), fontsize = 14)
    #chart.set_yticklabels(chart.get_ymajorticklabels(), fontsize = 12)
    plt.ylabel("Shortage", size=16)
    plt.xlabel("Service level", size=16)
    plt.title("Total number of shortage with different Service levels", size=18)
    plt.legend(fontsize=13)
    

    plt.subplot(1, 2, 2)
    #this part plots different Protocol coefficient and their related real costs
    #plt.figure(figsize=(17,8.5))
    #plt.figure(figsize=(17,8.5))
    chart=sns.barplot(x="Service level - specified", 
                y="Real cost", 
                hue="Forecast method", 
                data=df_comparison_SC_real_cost,palette=["C0", "C1", "C2"])

    chart.set_xticklabels(chart.get_xmajorticklabels(), fontsize = 14)

    plt.ylabel("Real cost", size=16)
    plt.xlabel("Service level", size=16)
    plt.title("Real cost with different Service levels", size=16)
    plt.legend(fontsize=13)

"""# Scenario simulation

## Dataframe creation
"""

#Creating a tuple to put all the results in
#results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
forecast_names = pd.DataFrame(np.array([['SIR'],['Holt'],['Naive']]), columns=['Forecast Method'])

#Creating a data frame to put the filtered resutls in

inventory_data_result = pd.DataFrame(data=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], columns=['Forecast method',
                                                                          'Shortage',
                                                                          'Total cost',
                                                                          'Real cost',
                                                                          'Number of orders',
                                                                          'Inventory level at the last epoch',
                                                                          'Total demand',
                                                                          'Protocol coefficient',
                                                                          'Service level - specified',
                                                                          'Service level - achieved',
                                                                          'Number of epochs with shortage',
                                                                          'Total holding cost',
                                                                          'Total unit cost',
                                                                          'Total ordering cost',
                                                                          'Total shortage cost',
                                                                          'Total number of units ordered'])
inventory_data_result

#Creating a data frame to put the filtered resutls in

inventory_data_result_scenarios_empty = pd.DataFrame( data=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                                             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                                             0,0,0,0,0,0,0,0,0,0,0]],columns=['Lead time',
                                                                          'SNR',
                                                                          'Protocol coefficient',
                                                                          'Suppliers minimum capacity',
                                                                          'Suppliers maximum capacity',
                                                                          'Inventory capacity',
                                                                          'Specified service level',
                                                                          'SIR forecast: Shortage',
                                                                          'SIR forecast: Inventory level at the end',
                                                                          'SIR forecast: Achieved service level',
                                                                          'SIR forecast: Number of epochs with shortage',
                                                                          'SIR forecast: Total number of units ordered',
                                                                          'SIR forecast: Number of orders',
                                                                          'SIR forecast: Real cost',
                                                                          'SIR forecast: Total cost',
                                                                          'SIR forecast: Average effective percentage bias',
                                                                          'SIR forecast: Effective no. of epochs with under-forecast',
                                                                          'SIR forecast: Effective no. of epochs with over-forecast',
                                                                          'SIR forecast: Percentage of Effective no. of epochs with over-forecast',
                                                                          'SIR forecast: Effective RMSE',
                                                                          'SIR forecast: RMSE',
                                                                          'SIR forecast: Effective MAPE',
                                                                          'SIR forecast: MAPE',
                                                                          'SIR forecast: Effective MAE',
                                                                          'SIR forecast: MAE',
                                                                          'Holt forecast: Shortage',
                                                                          'Holt forecast: Inventory level at the end',
                                                                          'Holt forecast: Achieved service level',
                                                                          'Holt forecast: Number of epochs with shortage',
                                                                          'Holt forecast: Total number of units ordered',
                                                                          'Holt forecast: Number of orders',
                                                                          'Holt forecast: Real cost',
                                                                          'Holt forecast: Total cost',
                                                                          'Holt forecast: Average effective percentage bias',
                                                                          'Holt forecast: Effective no. of epochs with under-forecast',
                                                                          'Holt forecast: Effective no. of epochs with over-forecast',
                                                                          'Holt forecast: Percentage of Effective no. of epochs with over-forecast',
                                                                          'Holt forecast: Effective RMSE',
                                                                          'Holt forecast: RMSE',
                                                                          'Holt forecast: Effective MAPE',
                                                                          'Holt forecast: MAPE',
                                                                          'Holt forecast: Effective MAE',
                                                                          'Holt forecast: MAE',
                                                                          'Naive forecast: Shortage',
                                                                          'Naive forecast: Inventory level at the end',
                                                                          'Naive forecast: Achieved service level',
                                                                          'Naive forecast: Number of epochs with shortage',
                                                                          'Naive forecast: Total number of units ordered',
                                                                          'Naive forecast: Number of orders',
                                                                          'Naive forecast: Real cost',
                                                                          'Naive forecast: Total cost',
                                                                          'Naive forecast: Average effective percentage bias',
                                                                          'Naive forecast: Effective no. of epochs with under-forecast',
                                                                          'Naive forecast: Effective no. of epochs with over-forecast',
                                                                          'Naive forecast: Percentage of Effective no. of epochs with over-forecast',
                                                                          'Naive forecast: Effective RMSE',
                                                                          'Naive forecast: RMSE',
                                                                          'Naive forecast: Effective MAPE',
                                                                          'Naive forecast: MAPE',
                                                                          'Naive forecast: Effective MAE',
                                                                          'Naive forecast: MAE',
                                                                          'Total demand/consumption'])
inventory_data_result_scenarios_empty

#Creating a data frame to put the filtered resutls in

inventory_data_result_scenarios_combine_empty = pd.DataFrame( data=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                                             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                                             0,0,0,0,0,0,0,0,0,0,0,0]],columns=['Lead time',
                                                                          'SNR',
                                                                          'Protocol coefficient',
                                                                          'Suppliers minimum capacity',
                                                                          'Suppliers maximum capacity',
                                                                          'Inventory capacity',
                                                                          'Specified service level',
                                                                          'SIR forecast: Shortage',
                                                                          'SIR forecast: Inventory level at the end',
                                                                          'SIR forecast: Achieved service level',
                                                                          'SIR forecast: Number of epochs with shortage',
                                                                          'SIR forecast: Total number of units ordered',
                                                                          'SIR forecast: Number of orders',
                                                                          'SIR forecast: Real cost',
                                                                          'SIR forecast: Total cost',
                                                                          'SIR forecast: Average effective percentage bias',
                                                                          'SIR forecast: Effective no. of epochs with under-forecast',
                                                                          'SIR forecast: Effective no. of epochs with over-forecast',
                                                                          'SIR forecast: Percentage of Effective no. of epochs with over-forecast',
                                                                          'SIR forecast: Effective RMSE',
                                                                          'SIR forecast: EMSE',
                                                                          'SIR forecast: Effective MAPE',
                                                                          'SIR forecast: MAPE',
                                                                          'SIR forecast: Effective MAE',
                                                                          'SIR forecast: MAE',
                                                                          'Holt forecast: Shortage',
                                                                          'Holt forecast: Inventory level at the end',
                                                                          'Holt forecast: Achieved service level',
                                                                          'Holt forecast: Number of epochs with shortage',
                                                                          'Holt forecast: Total number of units ordered',
                                                                          'Holt forecast: Number of orders',
                                                                          'Holt forecast: Real cost',
                                                                          'Holt forecast: Total cost',
                                                                          'Holt forecast: Average effective percentage bias',
                                                                          'Holt forecast: Effective no. of epochs with under-forecast',
                                                                          'Holt forecast: Effective no. of epochs with over-forecast',
                                                                          'Holt forecast: Percentage of Effective no. of epochs with over-forecast',
                                                                          'Holt forecast: Effective RMSE',
                                                                          'Holt forecast: EMSE',
                                                                          'Holt forecast: Effective MAPE',
                                                                          'Holt forecast: MAPE',
                                                                          'Holt forecast: Effective MAE',
                                                                          'Holt forecast: MAE',
                                                                          'Naive forecast: Shortage',
                                                                          'Naive forecast: Inventory level at the end',
                                                                          'Naive forecast: Achieved service level',
                                                                          'Naive forecast: Number of epochs with shortage',
                                                                          'Naive forecast: Total number of units ordered',
                                                                          'Naive forecast: Number of orders',
                                                                          'Naive forecast: Real cost',
                                                                          'Naive forecast: Total cost',
                                                                          'Naive forecast: Average effective percentage bias',
                                                                          'Naive forecast: Effective no. of epochs with under-forecast',
                                                                          'Naive forecast: Effective no. of epochs with over-forecast',
                                                                          'Naive forecast: Percentage of Effective no. of epochs with over-forecast',
                                                                          'Naive forecast: Effective RMSE',
                                                                          'Naive forecast: EMSE',
                                                                          'Naive forecast: Effective MAPE',
                                                                          'Naive forecast: MAPE',
                                                                          'Naive forecast: Effective MAE',
                                                                          'Naive forecast: MAE',
                                                                          'Total demand/consumption',
                                                                          'CC deviation'])
inventory_data_result_scenarios_combine_empty

#Creating a data frame to put the filtered resutls in

inventory_data_result_scenarios_CC_dev_empty = pd.DataFrame(data=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],columns=['CC deviation',
                                                                          'Protocol coefficient',
                                                                          'Specified service level',
                                                                          'SIR forecast: Shortage',
                                                                          'SIR forecast: Inventory level at the end',
                                                                          'SIR forecast: Achieved service level',
                                                                          'SIR forecast: Number of epochs with shortage',
                                                                          'SIR forecast: Total number of units ordered',
                                                                          'SIR forecast: Number of orders',
                                                                          'SIR forecast: Real cost',
                                                                          'SIR forecast: Total cost',
                                                                          'Holt forecast: Shortage',
                                                                          'Holt forecast: Inventory level at the end',
                                                                          'Holt forecast: Achieved service level',
                                                                          'Holt forecast: Number of epochs with shortage',
                                                                          'Holt forecast: Total number of units ordered',
                                                                          'Holt forecast: Number of orders',
                                                                          'Holt forecast: Real cost',
                                                                          'Holt forecast: Total cost',
                                                                          'Naive forecast: Shortage',
                                                                          'Naive forecast: Inventory level at the end',
                                                                          'Naive forecast: Achieved service level',
                                                                          'Naive forecast: Number of epochs with shortage',
                                                                          'Naive forecast: Total number of units ordered',
                                                                          'Naive forecast: Number of orders',
                                                                          'Naive forecast: Real cost',
                                                                          'Naive forecast: Total cost',
                                                                          'Total demand/consumption'])
inventory_data_result_scenarios_CC_dev_empty

#Creating a data frame to put the filtered resutls in

sim_scenarios_empty = pd.DataFrame( data=[[0,0,0,0,0,0,0,0]],columns=['Min pallet',
                                                                          'Max pallet',
                                                                          'Protocol coefficient',
                                                                          'Lead time',
                                                                          'Inventory capacity (in pallet)',
                                                                          'Specified service level',
                                                                          'SNR',
                                                                          'Lag'])
sim_scenarios_empty

"""### Plotting functions for simulated with noise"""

def plot_raw_data (data_to_plot):

  real_data_scenario=data_to_plot.copy()
  day_start_scenarios=0
  day_end_scenarios=len(real_data_scenario)


  #PLOTTING FUNCTION
  #===============================================================================
  #t_scenarios = np.linspace(day_start_scenarios, day_end_scenarios-1, (day_end_scenarios-day_start_scenarios))
  t_scenarios = np.linspace(day_start_scenarios, day_end_scenarios-1, (day_end_scenarios-day_start_scenarios))

  #Choose one of the following methods for the noise calculation
  #       mean_each_epoch
  #       max_each_epoch
  #       max_all_epoch
  #       mean_all_epoch
  if mean_type_for_noise=='mean_each_epoch':
    mean_used_for_noise='mean of each epoch'
  elif mean_type_for_noise=='max_each_epoch':
    mean_used_for_noise='max of each epoch'
  elif mean_type_for_noise=='max_all_epoch':
    mean_used_for_noise='max value among all days'
  elif mean_type_for_noise=='mean_all_epoch':
    mean_used_for_noise='mean value of all days'

  fig = plt.figure(figsize=(17,8.5))
  ax = fig.add_subplot(111)

      #ax.set_xlim(xmin=0)
      #ax.set_ylim(ymax=200)

  #Start_day_training = 0
  #end_day_training = 50

  #plotting the training set of data
  ax.plot(t_scenarios[day_start_scenarios:day_end_scenarios], real_data_scenario, 'orange',linewidth=4)

  ax.set_ylim(ymin=0,ymax=800)
  plt.title('Hospitalization with noise of %d%% variation from mean (%s) in British Columbia'%((percentage_of_variation*100),mean_used_for_noise),fontsize=16)
  plt.xlabel('Time (days)',fontsize=16)
  plt.ylabel('Hospitalization',fontsize=16)
  plt.legend(fontsize=13)

  #plt.savefig('Hospitalization with noise in British Columbia in BC',dpi=300)

def plot_raw_data_SNR (data_to_plot):

  real_data_scenario=data_to_plot.copy()
  day_start_scenarios=0
  day_end_scenarios=len(real_data_scenario)


  #PLOTTING FUNCTION
  #===============================================================================
  #t_scenarios = np.linspace(day_start_scenarios, day_end_scenarios-1, (day_end_scenarios-day_start_scenarios))
  t_scenarios = np.linspace(day_start_scenarios, day_end_scenarios-1, (day_end_scenarios-day_start_scenarios))

  #Choose one of the following methods for the noise calculation
  #       mean_each_epoch
  #       max_each_epoch
  #       max_all_epoch
  #       mean_all_epoch


  fig = plt.figure(figsize=(17,8.5))
  ax = fig.add_subplot(111)

      #ax.set_xlim(xmin=0)
      #ax.set_ylim(ymax=200)

  #Start_day_training = 0
  #end_day_training = 50

  #plotting the training set of data
  ax.plot(t_scenarios[day_start_scenarios:day_end_scenarios], real_data_scenario, 'orange',linewidth=4)

  ax.set_ylim(ymin=0,ymax=800)
  plt.title('Simulated consumption data in British Columbia(SNR of %s and CC of %s)'%(SNR,protocol_coef),fontsize=16)
  plt.xlabel('Time (days)',fontsize=16)
  plt.ylabel('Consumption',fontsize=16)
  plt.legend(fontsize=13)

  #plt.savefig('Hospitalization with noise in British Columbia in BC',dpi=300)

def plot_raw_data_without_noise (data_to_plot,coef_Cons):

  real_data_scenario=data_to_plot.copy()
  day_start_scenarios=0
  day_end_scenarios=len(real_data_scenario)


  #PLOTTING FUNCTION
  #===============================================================================
  #t_scenarios = np.linspace(day_start_scenarios, day_end_scenarios-1, (day_end_scenarios-day_start_scenarios))
  t_scenarios = np.linspace(day_start_scenarios, day_end_scenarios-1, (day_end_scenarios-day_start_scenarios))

  #Choose one of the following methods for the noise calculation
  #       mean_each_epoch
  #       max_each_epoch
  #       max_all_epoch
  #       mean_all_epoch

  fig = plt.figure(figsize=(17,8.5))
  ax = fig.add_subplot(111)

      #ax.set_xlim(xmin=0)
      #ax.set_ylim(ymax=200)

  #Start_day_training = 0
  #end_day_training = 50

  #plotting the training set of data
  ax.plot(t_scenarios[day_start_scenarios:day_end_scenarios], real_data_scenario, 'orange',linewidth=4)

  ax.set_ylim(ymin=0,ymax=800)
  plt.title('Simulated consumption of PPE in British Columbia (without noise and CC of %d)'%(coef_Cons),fontsize=16)
  plt.xlabel('Time (days)',fontsize=16)
  plt.ylabel('Consumption',fontsize=16)
  plt.legend(fontsize=13)

  #plt.savefig('Hospitalization with noise in British Columbia in BC',dpi=300)

empty_inventory_data_result=inventory_data_result.iloc[0:0].copy()

empty_inventory_data_result

"""## Basic Scenarios

### Basic Scenario 1: with random noise on overall consumption

#### Scenario selection - random noise on overall consumption
"""

#Scenario range
#=======================================================================================
protocol_coef_variation=[3,3.5,4,4.5,5]
service_level_variation=[0.90]#,0.950,0.970,0.9900,0.99900]


#STANDARD DEVIATION TYPE
#=======================================================================================
#Choose one of the following method for the standard deviation

  #      std of previous epoch
  #      std from beginning
type_std_system='std of previous epoch'


#suppliers_capacity_max = 5000

#NOSIE SELECTION
#=======================================================================================
percentage_of_variation=0.2
mu_noise=0
#Choose one of the following methods for the noise calculation
#       mean_each_epoch
#       max_each_epoch
#       max_all_epoch
#       mean_all_epoch
mean_type_for_noise='max_each_epoch'

# Choose a percentage that simulated data should deviate from the mean, 
#for lower limit enter the negative number
deviation_percentage_from_mean=0

"""#### Scenario creation"""

#provincial_data_CIHI['daily_Hospitalizations']= noise_creator_on_overall_consumption(provincial_data_CIHI['daily_Hospitalizations'],mean_type_for_noise)
#provincial_data_CIHI['daily_Hospitalizations']

#this code will pass a clean table to the loop by removing all the rows of the dataframes
complete_inventory_results=inventory_data_result.iloc[0:0].copy()

#we add the noise to the number of hospitalization because the noise level should be the same for different protocols
raw_data_with_noise= noise_creator_on_overall_consumption(provincial_data_CIHI['daily_Hospitalizations'],
                                                              mean_type_for_noise)

for k in range(len(protocol_coef_variation)):

  #DATA SIMULATION
  #==============================================================================
  #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
  #noise recalculation should only be applied if we have have different coefficient of protocol, 
  #otherwise noise should stya the same, thats why it is outside of the loop
  
  protocol_coef = protocol_coef_variation[k]
  provincial_data_CIHI['daily_consumption']=protocol_coef * raw_data_with_noise
  provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()


  for j in range(len(service_level_variation)):
    
    service_level = service_level_variation[j]


    # >>>USER INPUT<<<
    # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
    # ======================================================================================================
    forecast_data_1 = provincial_data_CIHI.copy()

    # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
    consumption_date_type='daily_consumption'

    y = forecast_data_1[consumption_date_type]
    y_cumulative = forecast_data_1['cumulative_daily_consumption']

    # PERIODIC REVIEW SYSTEM with NAIVE forecasting
    #====================================================================================================================

    inventory_data_periodic_naive = inventory_data.copy()
    inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)


    for i in range(1,math.ceil(len(y)/R)+1):
      y_to_train = y[max(0,i-2)*R:(i)*R]

      y_to_val = y[(i)*R:(i)*R+(R+L)]
      #y_set=y[0:(i+1)*R+(R+L)]
      prediction = naive_forecast_method(y_to_train,y_to_val)
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_naive=peridoic_review(y_previous_epoch,inventory_data_periodic_naive,prediction,epoch_no)

    # PERIODIC REVIEW SYSTEM with HOLT forecasting
    #====================================================================================================================

    inventory_data_periodic_Holt = inventory_data
    inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

    #hospitalization_data_type='daily_consumption'

    try:
      hospitalization_data_type=hospitalization_data_type
    except NameError:
      hospitalization_data_type='daily_consumption'

    if hospitalization_data_type=='daily_consumption':
      y_holt=y
    else:
      y_holt=y_cumulative


    for i in range(1,math.ceil(len(y)/R)+1):
      y_to_train = y_holt[0:(i)*R]
      y_to_val = y_holt[(i)*R:(i)*R+(R+L)]
      y_set=y_holt[0:(i)*R+(R+L)]
      y_previous_epoch_holt=y_holt[(i-1)*R:(i)*R]
      prediction = holt(y_set, y_to_train,y_to_val,i,y_previous_epoch_holt)
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_Holt=peridoic_review(y_previous_epoch,inventory_data_periodic_Holt,prediction,epoch_no)

    # PERIODIC REVIEW SYSTEM with SIR forecasting model
    #====================================================================================================================

    inventory_data_periodic_SIR = inventory_data.copy()
    inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)


    for i in range(1,math.ceil(len(y)/R)+1):
      y_to_train = y[0:(i)*R]
      y_to_val = y[(i)*R:(i)*R+(R+L)]
      y_set=y[0:(i)*R+(R+L)]
      prediction = SIR_forecast_model(i,y_set)
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction,epoch_no)

    results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
    for i in range(len(results)):
        inventory_data_result.loc[i]=[forecast_names['Forecast Method'][i],
                                      results[i]['shortage'].sum(),
                                      results[i]['total cost'].sum(),
                                      results[i]['real cost'].sum(),
                                      (results[i]['order'] != 0).sum(),
                                      results[i]['current'].iloc[-1],
                                      results[i]['consumption in last epoch'].sum(),
                                      protocol_coef,
                                      service_level,
                                      round((1-(results[i]['shortage'] != 0).sum()/len(results[i]['shortage']-1)),2),
                                      (results[i]['shortage'] != 0).sum(),
                                      results[i]['holding cost'].sum(),
                                      results[i]['unit cost'].sum(),
                                      results[i]['ordering cost'].sum(),
                                      results[i]['shortage cost'].sum(),
                                      results[i]['order'].sum()]

    complete_inventory_results=complete_inventory_results.append([inventory_data_result])


    print(inventory_data_periodic_Holt['holding cost'].sum(),
      inventory_data_periodic_Holt['unit cost'].sum(),
      inventory_data_periodic_Holt['ordering cost'].sum())

"""### Basic Scenarios 2: with random noise with SNR on CC

#### Scenario selection - random noise with SNR on CC
"""

#Scenario range
#=======================================================================================
protocol_coef_variation=[4] #,3.5,4,4.5,5]
service_level_variation=[0.90]#,0.950,0.970,0.9900,0.99900]


#STANDARD DEVIATION TYPE
#=======================================================================================
#Choose one of the following method for the standard deviation

  #      std of previous epoch
  #      std from beginning
type_std_system='std of previous epoch'


#suppliers_capacity_max = 5000

#NOSIE SELECTION
#=======================================================================================
#Do you want to have noise added to the data?
#     'yes'     if you want to add noise to the data 
#     'no'      if you do NOT want to add noise
noise_addition='yes'

#specify the desired SNR 

SNR= 2

#Re-active Coefficient of consumption (CC) in SIR forecasting
#======================================================================================= 
# Do you want to increase\decrease CC in antipication of consumption deviation from protocols?
# NOTE: 
#   1. The values are in percentage; e.g. 20 => 20% 
#   2. For reduction simply put the negative sign

SIR_forecast_CC_dev=[100,50,0,-200]


L=5

"""#### Scenario creation"""

#this code will pass a clean table to the loop by removing all the rows of the dataframes
complete_inventory_results=inventory_data_result.iloc[0:0].copy()
inventory_data_result = inventory_data_result.iloc[0:0].copy()

for k in range(len(protocol_coef_variation)):

  #DATA SIMULATION
  #==============================================================================
  #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
  #noise recalculation should only be applied if we have have different coefficient of protocol, 
  #otherwise noise should stya the same, thats why it is outside of the loop
  
  protocol_coef = protocol_coef_variation[k]
  provincial_data_CIHI['daily_consumption']=noise_creator_with_SNR(provincial_data_CIHI['daily_Hospitalizations'],
                                                                   protocol_coef)
  provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()


  for j in range(len(service_level_variation)):
    
    service_level = service_level_variation[j]


    # >>>USER INPUT<<<
    # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
    # ======================================================================================================
    forecast_data_1 = provincial_data_CIHI.copy()

    # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
    consumption_date_type='daily_consumption'

    y = forecast_data_1[consumption_date_type]
    y_cumulative = forecast_data_1['cumulative_daily_consumption']

    # PERIODIC REVIEW SYSTEM with NAIVE forecasting
    #====================================================================================================================

    inventory_data_periodic_naive = inventory_data.copy()
    inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)


    for i in range(1,math.ceil(len(y)/R)+1):
      y_to_train = y[max(0,i-2)*R:(i)*R]

      y_to_val = y[(i)*R:(i)*R+(R+L)]
      #y_set=y[0:(i+1)*R+(R+L)]
      prediction = naive_forecast_method(y_to_train,y_to_val)
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_naive=peridoic_review(y_previous_epoch,inventory_data_periodic_naive,prediction,epoch_no)

    # PERIODIC REVIEW SYSTEM with HOLT forecasting
    #====================================================================================================================

    inventory_data_periodic_Holt = inventory_data
    inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

    #hospitalization_data_type='daily_consumption'

    try:
      hospitalization_data_type=hospitalization_data_type
    except NameError:
      hospitalization_data_type='daily_consumption'

    if hospitalization_data_type=='daily_consumption':
      y_holt=y
    else:
      y_holt=y_cumulative


    for i in range(1,math.ceil(len(y)/R)+1):
      y_to_train = y_holt[0:(i)*R]
      y_to_val = y_holt[(i)*R:(i)*R+(R+L)]
      y_set=y_holt[0:(i)*R+(R+L)]
      y_previous_epoch_holt=y_holt[(i-1)*R:(i)*R]
      prediction = holt(y_set, y_to_train,y_to_val,i,y_previous_epoch_holt)
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_Holt=peridoic_review(y_previous_epoch,inventory_data_periodic_Holt,prediction,epoch_no)

    # PERIODIC REVIEW SYSTEM with SIR forecasting model
    #====================================================================================================================

    #inventory_data_periodic_SIR = inventory_data.copy()
    #inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)

    #Creating a dict comprehension to place the dataframes into
    SIR_inventory_dict= {'SIR_dev_{}'.format(dev): pd.DataFrame() for dev in SIR_forecast_CC_dev}

    for d in range(len(SIR_forecast_CC_dev)):

      inventory_data_periodic_SIR = inventory_data.copy()
      inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)
      
      for i in range(1,math.ceil(len(y)/R)+1):
        y_to_train = y[0:(i)*R]
        y_to_val = y[(i)*R:(i)*R+(R+L)]
        y_set=y[0:(i)*R+(R+L)]
        prediction = SIR_forecast_model(i,y_set,SIR_forecast_CC_dev[d])
        y_previous_epoch=y[(i-1)*R:(i)*R]
        epoch_no=i
        inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction,epoch_no)
    
      SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[d])]=inventory_data_periodic_SIR

    #creating a data frame for the names of the methods used
    forecast_names = pd.DataFrame(np.array([['Holt'],['Naive']]), columns=['Forecast Method'])
    #adding different variation of SIR methods
    for i in range(len(SIR_forecast_CC_dev)):
      forecast_names.loc[i+2]=['SIR with deviation of {}%'.format(SIR_forecast_CC_dev[i])]

    results=(inventory_data_periodic_Holt,inventory_data_periodic_naive)
    
    for i in range(len(forecast_names)):
        if i <=1:
          inventory_data_result.loc[i]=[forecast_names['Forecast Method'][i],
                                        results[i]['shortage'].sum(),
                                        results[i]['total cost'].sum(),
                                        results[i]['real cost'].sum(),
                                        (results[i]['order'] != 0).sum(),
                                        results[i]['current'].iloc[-1],
                                        results[i]['consumption in last epoch'].sum(),
                                        protocol_coef,
                                        service_level,
                                        round((1-(results[i]['shortage'] != 0).sum()/len(results[i]['shortage']-1)),2),
                                        (results[i]['shortage'] != 0).sum(),
                                        results[i]['holding cost'].sum(),
                                        results[i]['unit cost'].sum(),
                                        results[i]['ordering cost'].sum(),
                                        results[i]['shortage cost'].sum(),
                                        results[i]['order'].sum()]
        elif i >=1:
          inventory_data_result.loc[i]=[forecast_names['Forecast Method'][i],
                                        SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['shortage'].sum(),
                                        SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['total cost'].sum(),
                                        SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['real cost'].sum(),
                                        (SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['order'] != 0).sum(),
                                        SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['current'].iloc[-1],
                                        SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['consumption in last epoch'].sum(),
                                        protocol_coef,
                                        service_level,
                                        round((1-(SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['shortage'] != 0).sum()/len(SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['shortage']-1)),2),
                                        (SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['shortage'] != 0).sum(),
                                        SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['holding cost'].sum(),
                                        SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['unit cost'].sum(),
                                        SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['ordering cost'].sum(),
                                        SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['shortage cost'].sum(),
                                        SIR_inventory_dict['SIR_dev_{}'.format(SIR_forecast_CC_dev[i-2])]['order'].sum()]


                  

    complete_inventory_results=complete_inventory_results.append([inventory_data_result])

"""### Basic Scenarios 3: with complete ranges on % of devaition

#### Parameter seclection
"""

#Scenario range
#=======================================================================================
protocol_coef_variation=[4,5,6]       #,3.5,4,4.5,5]
service_level_variation=[0.95]    #,0.950,0.970,0.9900,0.99900]


#STANDARD DEVIATION TYPE
#=======================================================================================
#Choose one of the following method for the standard deviation

  #      std of previous epoch
  #      std from beginning
type_std_system='std of previous epoch'


# SUPPLIER'S CONSTRAINTS:
#====================================
# How many items are in a box:
item_in_box=20
# How many items are in a pallet:
box_in_pallet=40
# What is the minimum order pallet:
minimum_pallet= 1
# What is the maximum order pallet:
maximum_pallet = 10
# What is the maximum capacity of inventory space?
inventory_capactiy_in_pallet=50

# LEAD TIME:
#====================================
L=20 #days

#NOSIE SELECTION
#=======================================================================================
#Do you want to have noise added to the data?
#     'yes'     if you want to add noise to the data 
#     'no'      if you do NOT want to add noise
noise_addition='yes'

#specify the desired Signal to noise ratio (SNR):

SNR= 2

#Re-active Coefficient of consumption (CC) in SIR forecasting
#======================================================================================= 
# Do you want to increase\decrease CC in antipication of consumption deviation from protocols?
# NOTE: 
#   1. The values are in percentage; e.g. 20 => 20% 
#   2. These values indicates what we assume the deviation should from the prediction
     # for example if the prediction is at 20 consumption for a single, we think the consumption is 
     # 20% lower (or higher); therefore the planning is based on our assumed consumption and not the exact prediction
#   3. For lower assumption of consumption simply put the negative sign

# What is the upper limit of deviation?
CC_dev_upper_limit=100
# What is the lower limit of devation?
CC_dev_lower_limit=-100
# what is the steps?
CC_dev_step=25

#SIR_forecast_CC_dev=[100,50,0,-200]

"""#### Scenario creation"""

# These varibales are being used by the periodic review function:
item_in_pallet= item_in_box * box_in_pallet
suppliers_capacity_min= item_in_pallet * minimum_pallet
suppliers_capacity_max= item_in_pallet * maximum_pallet
inventory_capactiy= inventory_capactiy_in_pallet * item_in_pallet

# Creating a list of desired steps for different C deviations
List_CC_dev_step=np.arange(CC_dev_lower_limit,(CC_dev_upper_limit+CC_dev_step), CC_dev_step).tolist()

inventory_data_result_scenarios_CC_dev=inventory_data_result_scenarios_CC_dev_empty.copy()
temp_df_inventory_scenarios_CC_dev=inventory_data_result_scenarios_CC_dev.iloc[0:0].copy()
#Creating an empty dataframe to place the final data in
complete_inventory_results_scenarios_CC_dev=pd.DataFrame()

for k in range(len(protocol_coef_variation)):
  # Clearing the temporary data
  temp_df_inventory_scenarios_CC_dev=inventory_data_result_scenarios_CC_dev.iloc[0:0].copy()

  #DATA SIMULATION
  #==============================================================================
  #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
  #noise recalculation should only be applied if we have have different coefficient of protocol, 
  #otherwise noise should stya the same, thats why it is outside of the loop
  
  # Assigining the selected CC into its variable:
  protocol_coef = protocol_coef_variation[k]

  # Simulating the data based on the specified CC:
  provincial_data_CIHI['daily_consumption']=noise_creator_with_SNR(provincial_data_CIHI['daily_Hospitalizations'],protocol_coef)
  provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()

  for j in range(len(service_level_variation)):
    
    # Assigining the selected service level into its variable:
    service_level = service_level_variation[j]

    for CC_dev in range(len(List_CC_dev_step)):
      

      # >>>USER INPUT<<<
      # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
      # ======================================================================================================
      forecast_data_1 = provincial_data_CIHI.copy()

      # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
      consumption_date_type='daily_consumption'

      y = forecast_data_1[consumption_date_type]
      y_cumulative = forecast_data_1['cumulative_daily_consumption']

      # PERIODIC REVIEW SYSTEM with NAIVE forecasting
      #====================================================================================================================

      inventory_data_periodic_naive = inventory_data.copy()
      inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)


      for i in range(1,math.ceil(len(y)/R)+1):
        y_to_train = y[max(0,i-2)*R:(i)*R]

        y_to_val = y[(i)*R:(i)*R+(R+L)]
        #y_set=y[0:(i+1)*R+(R+L)]
        prediction = naive_forecast_method(y_to_train,y_to_val,List_CC_dev_step[CC_dev])
        y_previous_epoch=y[(i-1)*R:(i)*R]
        epoch_no=i
        inventory_data_periodic_naive=peridoic_review(y_previous_epoch,inventory_data_periodic_naive,prediction,epoch_no)

      # PERIODIC REVIEW SYSTEM with HOLT forecasting
      #====================================================================================================================

      inventory_data_periodic_Holt = inventory_data
      inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

      #hospitalization_data_type='daily_consumption'

      try:
        hospitalization_data_type=hospitalization_data_type
      except NameError:
        hospitalization_data_type='daily_consumption'

      if hospitalization_data_type=='daily_consumption':
        y_holt=y
      else:
        y_holt=y_cumulative


      for i in range(1,math.ceil(len(y)/R)+1):
        y_to_train = y_holt[0:(i)*R]
        y_to_val = y_holt[(i)*R:(i)*R+(R+L)]
        y_set=y_holt[0:(i)*R+(R+L)]
        y_previous_epoch_holt=y_holt[(i-1)*R:(i)*R]
        prediction = holt(y_set, y_to_train,y_to_val,i,y_previous_epoch_holt,List_CC_dev_step[CC_dev])
        y_previous_epoch=y[(i-1)*R:(i)*R]
        epoch_no=i
        inventory_data_periodic_Holt=peridoic_review(y_previous_epoch,inventory_data_periodic_Holt,prediction,epoch_no)

      # PERIODIC REVIEW SYSTEM with SIR forecasting model
      #====================================================================================================================


      inventory_data_periodic_SIR = inventory_data.copy()
      inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)
      
      for i in range(1,math.ceil(len(y)/R)+1):
        y_to_train = y[0:(i)*R]
        y_to_val = y[(i)*R:(i)*R+(R+L)]
        y_set=y[0:(i)*R+(R+L)]
        prediction = SIR_forecast_model(i,y_set,List_CC_dev_step[CC_dev])
        y_previous_epoch=y[(i-1)*R:(i)*R]
        epoch_no=i
        inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction,epoch_no)

      results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
      
      # Putting everything into a dataframe
      #=========================================================================================================
      # putting in the main parameters of the scenarios in the data frames
      inventory_data_result_scenarios_CC_dev['CC deviation']=List_CC_dev_step[CC_dev]
      inventory_data_result_scenarios_CC_dev['Protocol coefficient']=protocol_coef
      inventory_data_result_scenarios_CC_dev['Specified service level']=service_level
      
      # The total demand/consumption is the same for all methods, so we can pick any of them, in this case, SIR was picked
      inventory_data_result_scenarios_CC_dev['Total demand/consumption']=inventory_data_periodic_SIR['consumption in last epoch'].sum()

      # Puttin the analysis data into the data frame:
      #=========================================================================================================
      results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
      
      for i in range(len(results)):
        inventory_data_result_scenarios_CC_dev.iloc[[0],[(i*8)+3]]=results[i]['shortage'].sum()
        inventory_data_result_scenarios_CC_dev.iloc[[0],[(i*8)+4]]=results[i]['current'].iloc[-1],
        inventory_data_result_scenarios_CC_dev.iloc[[0],[(i*8)+5]]=round((1-(results[i]['shortage'] != 0).sum()/len(results[i]['shortage']-1)),2)
        inventory_data_result_scenarios_CC_dev.iloc[[0],[(i*8)+6]]=(results[i]['shortage'] != 0).sum()
        inventory_data_result_scenarios_CC_dev.iloc[[0],[(i*8)+7]]=results[i]['order'].sum()
        inventory_data_result_scenarios_CC_dev.iloc[[0],[(i*8)+8]]=(results[i]['order'] != 0).sum()
        inventory_data_result_scenarios_CC_dev.iloc[[0],[(i*8)+9]]=results[i]['real cost'].sum()
        inventory_data_result_scenarios_CC_dev.iloc[[0],[(i*8)+10]]=results[i]['total cost'].sum()


      #inventory_data_result_scenarios_CC_dev  
      temp_df_inventory_scenarios_CC_dev=temp_df_inventory_scenarios_CC_dev.append([inventory_data_result_scenarios_CC_dev],ignore_index=True)

  temp_df_inventory_scenarios_CC_dev=temp_df_inventory_scenarios_CC_dev.add_suffix(' with CC of %d'%(protocol_coef))  
  complete_inventory_results_scenarios_CC_dev=pd.concat([complete_inventory_results_scenarios_CC_dev,temp_df_inventory_scenarios_CC_dev],axis=1)

#Exporting the data into the a csv file which will be saved into the drive
complete_inventory_results_scenarios_CC_dev.to_csv('Inventory management results with %s different SNR scenarios and %s different CC.csv'%(((CC_dev_upper_limit-CC_dev_lower_limit)/CC_dev_step),len(protocol_coef_variation)))

"""#### Results

##### Importing Data
"""

#complete_inventory_results_scenarios_CC_dev

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
complete_inventory_results_scenarios_CC_dev=pd.read_csv('Inventory management results with 400.0 different SNR scenarios and 3 different CC.csv')
#setting the date as the index of the dataframe
complete_inventory_results_scenarios_CC_dev=complete_inventory_results_scenarios_CC_dev.drop(columns='Unnamed: 0')
#rounding up the value of forecast in the data frame 
#SIR_forecast_df=np.ceil(SIR_forecast_df)
complete_inventory_results_scenarios_CC_dev=(complete_inventory_results_scenarios_CC_dev.copy().reset_index()).drop(columns='index')
#df_data_analysis = df_data_analysis.apply(pd.to_numeric)



complete_inventory_results_scenarios_CC_dev.head()

"""##### Preliminary results analysis"""

complete_inventory_results_scenarios_CC_dev.head()

x1=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[0])]
y1_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of %s'%(protocol_coef_variation[0])]
y1_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of %s'%(protocol_coef_variation[0])]
y1_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of %s'%(protocol_coef_variation[0])]

plt.plot(x1, y1_SIR)
plt.plot(x1, y1_Holt)
plt.plot(x1, y1_Naive)

plt.legend(["SIR with CC of %s"%(protocol_coef_variation[0]),
            "Holt with CC of %s"%(protocol_coef_variation[0]),
            "Naive with CC of %s"%(protocol_coef_variation[0])])

# Maximum 4 pallet for supplier capacity

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)


x1=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[0])]
y1_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of %s'%(protocol_coef_variation[0])]
y1_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of %s'%(protocol_coef_variation[0])]
y1_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of %s'%(protocol_coef_variation[0])]

plt.plot(x1, y1_SIR, color = 'green',label="SIR with CC of %s"%(protocol_coef_variation[0]),marker='^')
plt.plot(x1, y1_Holt, color = 'blue',label="Holt with CC of %s"%(protocol_coef_variation[0]),marker='^')
plt.plot(x1, y1_Naive, color = 'red',label="Naive with CC of %s"%(protocol_coef_variation[0]),marker='^')



x2=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[1])]
y2_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of %s'%(protocol_coef_variation[1])]
y2_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of %s'%(protocol_coef_variation[1])]
y2_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of %s'%(protocol_coef_variation[1])]

plt.plot(x2, y2_SIR, color = 'green',label="SIR with CC of %s"%(protocol_coef_variation[1]),marker='o')
plt.plot(x2, y2_Holt, color = 'blue',label="Holt with CC of %s"%(protocol_coef_variation[1]),marker='o')
plt.plot(x2, y2_Naive, color = 'red',label="Naive with CC of %s"%(protocol_coef_variation[1]),marker='o')




x3=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[2])]
y3_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of %s'%(protocol_coef_variation[2])]
y3_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of %s'%(protocol_coef_variation[2])]
y3_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of %s'%(protocol_coef_variation[2])]

plt.plot(x3, y3_SIR, color = 'green',label="SIR with CC of %s"%(protocol_coef_variation[0]),marker='.')
plt.plot(x3, y3_Holt, color = 'blue',label="Holt with CC of %s"%(protocol_coef_variation[0]),marker='.')
plt.plot(x3, y3_Naive, color = 'red',label="Naive with CC of %s"%(protocol_coef_variation[0]),marker='.')



plt.legend(["SIR with CC of %s"%(protocol_coef_variation[0]),
            "Holt with CC of %s"%(protocol_coef_variation[0]),
            "Naive with CC of %s"%(protocol_coef_variation[0]),
            "SIR with CC of %s"%(protocol_coef_variation[1]),
            "Holt with CC of %s"%(protocol_coef_variation[1]),
            "Naive with CC of %s"%(protocol_coef_variation[1]),
            "SIR with CC of %s"%(protocol_coef_variation[2]),
            "Holt with CC of %s"%(protocol_coef_variation[2]),
            "Naive with CC of %s"%(protocol_coef_variation[2])])

plt.xlabel('deviation % from CC',fontsize=16)
plt.ylabel('Shortage',fontsize=16)

complete_inventory_results_scenarios_CC_dev.head()

# Maximum 10 pallet for supplier capacity


fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)


x1=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[0])]
y1_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of %s'%(protocol_coef_variation[0])]
y1_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of %s'%(protocol_coef_variation[0])]
y1_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of %s'%(protocol_coef_variation[0])]

plt.plot(x1, y1_SIR, color = 'green',label="SIR with CC of %s"%(protocol_coef_variation[0]),marker='^')
plt.plot(x1, y1_Holt, color = 'blue',label="Holt with CC of %s"%(protocol_coef_variation[0]),marker='^')
plt.plot(x1, y1_Naive, color = 'red',label="Naive with CC of %s"%(protocol_coef_variation[0]),marker='^')



x2=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[1])]
y2_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of %s'%(protocol_coef_variation[1])]
y2_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of %s'%(protocol_coef_variation[1])]
y2_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of %s'%(protocol_coef_variation[1])]

plt.plot(x2, y2_SIR, color = 'green',label="SIR with CC of %s"%(protocol_coef_variation[1]),marker='o')
plt.plot(x2, y2_Holt, color = 'blue',label="Holt with CC of %s"%(protocol_coef_variation[1]),marker='o')
plt.plot(x2, y2_Naive, color = 'red',label="Naive with CC of %s"%(protocol_coef_variation[1]),marker='o')




x3=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[2])]
y3_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of %s'%(protocol_coef_variation[2])]
y3_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of %s'%(protocol_coef_variation[2])]
y3_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of %s'%(protocol_coef_variation[2])]

plt.plot(x3, y3_SIR, color = 'green',label="SIR with CC of %s"%(protocol_coef_variation[0]),marker='.')
plt.plot(x3, y3_Holt, color = 'blue',label="Holt with CC of %s"%(protocol_coef_variation[0]),marker='.')
plt.plot(x3, y3_Naive, color = 'red',label="Naive with CC of %s"%(protocol_coef_variation[0]),marker='.')



plt.legend(["SIR with CC of %s"%(protocol_coef_variation[0]),
            "Holt with CC of %s"%(protocol_coef_variation[0]),
            "Naive with CC of %s"%(protocol_coef_variation[0]),
            "SIR with CC of %s"%(protocol_coef_variation[1]),
            "Holt with CC of %s"%(protocol_coef_variation[1]),
            "Naive with CC of %s"%(protocol_coef_variation[1]),
            "SIR with CC of %s"%(protocol_coef_variation[2]),
            "Holt with CC of %s"%(protocol_coef_variation[2]),
            "Naive with CC of %s"%(protocol_coef_variation[2])])

plt.xlabel('deviation % from CC',fontsize=16)
plt.ylabel('Shortage',fontsize=16)

#Inventory level at the end

# Maximum 10 pallet for supplier capacity


fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)


x1=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[0])]
y1_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Inventory level at the end with CC of %s'%(protocol_coef_variation[0])]
y1_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Inventory level at the end with CC of %s'%(protocol_coef_variation[0])]
y1_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Inventory level at the end with CC of %s'%(protocol_coef_variation[0])]

plt.plot(x1, y1_SIR, color = 'green',label="SIR with CC of %s"%(protocol_coef_variation[0]),marker='^')
plt.plot(x1, y1_Holt, color = 'blue',label="Holt with CC of %s"%(protocol_coef_variation[0]),marker='^')
plt.plot(x1, y1_Naive, color = 'red',label="Naive with CC of %s"%(protocol_coef_variation[0]),marker='^')



#x2=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[1])]
#y2_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Inventory level at the end with CC of %s'%(protocol_coef_variation[1])]
#y2_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Inventory level at the end with CC of %s'%(protocol_coef_variation[1])]
#y2_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Inventory level at the end with CC of %s'%(protocol_coef_variation[1])]

plt.plot(x2, y2_SIR, color = 'green',label="SIR with CC of %s"%(protocol_coef_variation[1]),marker='o')
plt.plot(x2, y2_Holt, color = 'blue',label="Holt with CC of %s"%(protocol_coef_variation[1]),marker='o')
plt.plot(x2, y2_Naive, color = 'red',label="Naive with CC of %s"%(protocol_coef_variation[1]),marker='o')




x3=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[2])]
y3_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Inventory level at the end with CC of %s'%(protocol_coef_variation[2])]
y3_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Inventory level at the end with CC of %s'%(protocol_coef_variation[2])]
y3_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Inventory level at the end with CC of %s'%(protocol_coef_variation[2])]

plt.plot(x3, y3_SIR, color = 'green',label="SIR with CC of %s"%(protocol_coef_variation[0]),marker='.')
plt.plot(x3, y3_Holt, color = 'blue',label="Holt with CC of %s"%(protocol_coef_variation[0]),marker='.')
plt.plot(x3, y3_Naive, color = 'red',label="Naive with CC of %s"%(protocol_coef_variation[0]),marker='.')



plt.legend(["SIR with CC of %s"%(protocol_coef_variation[0]),
            "Holt with CC of %s"%(protocol_coef_variation[0]),
            "Naive with CC of %s"%(protocol_coef_variation[0]),
            "SIR with CC of %s"%(protocol_coef_variation[1]),
            "Holt with CC of %s"%(protocol_coef_variation[1]),
            "Naive with CC of %s"%(protocol_coef_variation[1]),
            "SIR with CC of %s"%(protocol_coef_variation[2]),
            "Holt with CC of %s"%(protocol_coef_variation[2]),
            "Naive with CC of %s"%(protocol_coef_variation[2])])

plt.xlabel('deviation % from CC',fontsize=16)
plt.ylabel('Inventory level at the end',fontsize=16)

x3=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of %s'%(protocol_coef_variation[2])]
y3_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of %s'%(protocol_coef_variation[2])]
y3_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of %s'%(protocol_coef_variation[2])]
y3_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of %s'%(protocol_coef_variation[2])]

plt.plot(x3, y3_SIR)
plt.plot(x3, y3_Holt)
plt.plot(x3, y3_Naive)

plt.legend(["SIR with CC of %s"%(protocol_coef_variation[1]),
            "Holt with CC of %s"%(protocol_coef_variation[1]),
            "Naive with CC of %s"%(protocol_coef_variation[1])])

x1=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of 4']
y1_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of 4']
y1_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of 4']
y1_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of 4']

plt.plot(x1, y1_SIR)
plt.plot(x1, y1_Holt)
plt.plot(x1, y1_Naive)

plt.legend(["SIR", "Holt","Naive"])

x2=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of 5']
y2_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of 5']
y2_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of 5']
y2_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of 5']

plt.plot(x2, y2_SIR)
plt.plot(x2, y2_Holt)
plt.plot(x2, y2_Naive)

plt.legend(["SIR", "Holt","Naive"])

x3=complete_inventory_results_scenarios_CC_dev['CC deviation with CC of 6']
y3_SIR=complete_inventory_results_scenarios_CC_dev['SIR forecast: Shortage with CC of 6']
y3_Holt=complete_inventory_results_scenarios_CC_dev['Holt forecast: Shortage with CC of 6']
y3_Naive=complete_inventory_results_scenarios_CC_dev['Naive forecast: Shortage with CC of 6']

plt.plot(x3, y3_SIR)
plt.plot(x3, y3_Holt)
plt.plot(x3, y3_Naive)

plt.legend(["SIR", "Holt","Naive"])

"""## Demand simulations

### parameters
"""

# In this section the parameters for ranges of scenarios are defined
#==============================================================================

# How many scenarios do you want?
no_scenarios=10000

# SUPPLIER'S CONSTRAINTS:
#====================================
# How many items are in a box:
item_in_box=12
# How many items are in a pallet:
box_in_pallet=1

item_in_pallet= item_in_box * box_in_pallet

# Minimum order
# what is the lower limit for minimum pallet order:
lower_limit_minimum_pallet=1 
# what is the upper limit for minimum pallet order:
upper_limit_minimum_pallet=10 

# Maximum order
# what is the lower limit for maximum pallet order:
lower_limit_maximum_pallet=200 
# what is the upper limit for maximum pallet order:
upper_limit_maximum_pallet=400

# Lead time
# what is the lower limit for lead time of the order:
lower_limit_L=5
# what is the upper limit for lead time of the order:
upper_limit_L=30

# CONSUMPTION'S CONSTRAINTS:
#====================================
# What is the lower limit of coefficient of consumption:
lower_limit_CC=3
# What is the upper limit of coefficient of consumption:
upper_limit_CC=7


# INVENTORY MANAGEMENT CONSTRAINTS:
#====================================
# Capacity of inventory
# What is the lower limit of invernoty capacity (in terms of number of pallet):
lower_limit_invernoty_capacity= 3000
# What is the upper limit of invernoty capacity (in terms of number of pallet):
upper_limit_invernoty_capacity= 6000

# Service level
# What is the lower limit of serivce level (in percent (%)):
lower_limit_service_level= 95
# What is the upper limit of serivce level (in percent (%))
upper_limit_service_level= 99.9


#  DATA SIMULATION CONSTRAINTS:
#====================================

#Do you want to have noise added to the data?
#     'yes'     if you want to add noise to the data 
#     'no'      if you do NOT want to add noise
noise_addition='yes'


# Noise generation 
# What is the lower limit of SNR:
lower_limit_SNR= 2
# What is the upper limit of SNR:
upper_limit_SNR= 10

# Deviation from CC
#SIR_forecast_CC_dev=[0]

#  Lag info 
#====================================

# how many epochs are there in the 'lag'?
lag=2

# daily lag limit 
# What is the lower limit of daily lag:
lower_limit_lag= 0
# What is the upper limit of daily lag:
upper_limit_lag= 14

#  Replenishment info 
#====================================

# how many days should be between the arrival of info (replenishement mode)
period_info=7


# epoch Period
R=7

import numpy as np
import pandas as pd

service_level = round(np.random.uniform(lower_limit_service_level, upper_limit_service_level),1)/100
service_level

"""### Scenario creator"""

sim_scenario=sim_scenarios_empty.copy()
complete_sim_scenarios=sim_scenario.iloc[0:0].copy()
#inventory_data_result_scenarios = inventory_data_result_scenarios.iloc[0:0].copy()
sim_demand=provincial_data_CIHI['daily_Hospitalizations'].to_frame().copy()
sim_noise=provincial_data_CIHI['daily_Hospitalizations'].to_frame().copy()


for scenarios in range(no_scenarios):
  # Using uniform distribution to randomly choose a value for the following parameters
  sim_scenario=sim_scenarios_empty.copy()

  minimum_pallet = round(np.random.uniform(lower_limit_minimum_pallet, upper_limit_minimum_pallet))
  maximum_pallet = round(np.random.uniform(lower_limit_maximum_pallet, upper_limit_maximum_pallet))
  protocol_coef = (round((np.random.uniform(lower_limit_CC, upper_limit_CC))*2))/2
  L = round(np.random.uniform(lower_limit_L, upper_limit_L))
  inventory_capactiy_in_pallet = round(np.random.uniform(lower_limit_invernoty_capacity, upper_limit_invernoty_capacity))
  service_level = round(np.random.uniform(lower_limit_service_level, upper_limit_service_level),1)/100
  SNR = round(np.random.uniform(lower_limit_SNR, upper_limit_SNR))
  lag = round(np.random.uniform(lower_limit_lag, upper_limit_lag))

  # Calculating the following varibales based on the above randomly chosen varibales
  item_in_pallet= item_in_box * box_in_pallet
  suppliers_capacity_min= item_in_pallet * minimum_pallet
  suppliers_capacity_max= item_in_pallet * maximum_pallet
  inventory_capactiy= inventory_capactiy_in_pallet * item_in_pallet


    #DATA SIMULATION
    #==============================================================================
    #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
    #noise recalculation should only be applied if we have have different coefficient of protocol, 
    #otherwise noise should stya the same, thats why it is outside of the loop
    
  
  sim_data=noise_creator_for_basic_scenario(provincial_data_CIHI['daily_Hospitalizations'],
                                                                             SNR,
                                                                             protocol_coef)
  
  provincial_data_CIHI['daily_consumption']=sim_data[0]
  provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()


      # >>>USER INPUT<<<
      # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
      # ======================================================================================================
  forecast_data_1 = provincial_data_CIHI.copy()

      # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
  consumption_date_type='daily_consumption'

  y = forecast_data_1[consumption_date_type]
  y_cumulative = forecast_data_1['cumulative_daily_consumption']

  sim_scenario['Min pallet']=minimum_pallet
  sim_scenario['Max pallet']=maximum_pallet
  sim_scenario['Protocol coefficient']=protocol_coef
  sim_scenario['Lead time']=L
  sim_scenario['Inventory capacity (in pallet)']=inventory_capactiy_in_pallet
  sim_scenario['Specified service level']=service_level
  sim_scenario['SNR']=SNR
  sim_scenario['Lag']=lag
  
  #adding the parameters of a new scenario to previous ones
  complete_sim_scenarios=complete_sim_scenarios.append([sim_scenario],ignore_index=True)
  
  # converting the demand series into a dataframe for demand so that it can be added to the dataframe
  y=y.to_frame()
  y.rename(columns={ y.columns[0]: 'sim demand no.%d'%(scenarios)}, inplace = True) 
  
  
  # converting the noise array into a dataframe for demand so that it can be added to the dataframe
  # we also have to set the index into the day format 
  temp_sim_noise = pd.DataFrame(sim_data[1],columns=['sim demand no.%d'%(scenarios)]).set_index(y.index)
 
  # adding the simulated demand and noise to the previous ones

  sim_demand=pd.concat([sim_demand,y], axis=1, join="inner")
  
  sim_noise=pd.concat([sim_noise,temp_sim_noise], axis=1, join="inner")


sim_demand.drop(sim_demand.columns[0], axis=1, inplace=True)
sim_noise.drop(sim_noise.columns[0], axis=1, inplace=True)

complete_sim_scenarios.to_csv('Simulated parameters, %d scenarios.csv'%(no_scenarios))
sim_demand.to_csv('Simulated demand, %d scenarios.csv'%(no_scenarios))
sim_noise.to_csv('Simulated noise, %d scenarios.csv'%(no_scenarios))

"""### Importing data

"""

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
complete_sim_scenarios=pd.read_csv('Simulated parameters, 10000 scenarios.csv')
#setting the date as the index of the dataframe
complete_sim_scenarios=complete_sim_scenarios.drop(columns='Unnamed: 0')

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
sim_demand=pd.read_csv('Simulated demand, 10000 scenarios.csv')

sim_demand=sim_demand.rename(columns={'Unnamed: 0':'Date'})
#setting the date as the index of the dataframe
sim_demand=sim_demand.set_index(sim_demand['Date'])

sim_demand.drop(sim_demand.columns[0], axis=1, inplace=True)

sim_demand=sim_demand.reset_index()
sim_demand=sim_demand.set_index(y.index).drop(columns='Date')

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
sim_noise=pd.read_csv('Simulated noise, 10000 scenarios.csv')
#setting the date as the index of the dataframe

sim_noise=sim_noise.rename(columns={'Unnamed: 0':'Date'})
#setting the date as the index of the dataframe
sim_noise=sim_noise.set_index(sim_noise['Date'])

sim_noise.drop(sim_noise.columns[0], axis=1, inplace=True)


sim_noise=sim_noise.reset_index()
sim_noise=sim_noise.set_index(y.index).drop(columns='Date')

"""## Myopic Scenario

#### Scenario creator
"""

#this code will pass a clean table to the loop by removing all the rows of the dataframes

inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()
complete_inventory_results_scenarios_myopic=inventory_data_result_scenarios.iloc[0:0].copy()
#inventory_data_result_scenarios = inventory_data_result_scenarios.iloc[0:0].copy()



for scenarios in range(no_scenarios):
  # Using uniform distribution to randomly choose a value for the following parameters
  inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()

  minimum_pallet = complete_sim_scenarios['Min pallet'][scenarios]
  maximum_pallet = complete_sim_scenarios['Max pallet'][scenarios]
  protocol_coef = complete_sim_scenarios['Protocol coefficient'][scenarios]
  L = complete_sim_scenarios['Lead time'][scenarios]
  inventory_capactiy_in_pallet = complete_sim_scenarios['Inventory capacity (in pallet)'][scenarios]
  service_level = complete_sim_scenarios['Specified service level'][scenarios]
  SNR = complete_sim_scenarios['SNR'][scenarios]
  

  # Calculating the following varibales
  item_in_pallet= item_in_box * box_in_pallet
  suppliers_capacity_min= item_in_pallet * minimum_pallet
  suppliers_capacity_max= item_in_pallet * maximum_pallet
  inventory_capactiy= inventory_capactiy_in_pallet * item_in_pallet


  #this code will pass a clean table to the loop by removing all the rows of the dataframes
  complete_inventory_results=inventory_data_result.iloc[0:0].copy()
  inventory_data_result = inventory_data_result.iloc[0:0].copy()



    #DATA SIMULATION
    #==============================================================================
    #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
    #noise recalculation should only be applied if we have have different coefficient of protocol, 
    #otherwise noise should stya the same, thats why it is outside of the loop
    
  provincial_data_CIHI['daily_consumption']=sim_demand.iloc[:,scenarios]
  provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()


      # >>>USER INPUT<<<
      # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
      # ======================================================================================================
  forecast_data_1 = provincial_data_CIHI.copy()

      # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
  consumption_date_type='daily_consumption'

  y = forecast_data_1[consumption_date_type]
  y_cumulative = forecast_data_1['cumulative_daily_consumption']

      # PERIODIC REVIEW SYSTEM with MYOPIC forecasting
      #====================================================================================================================

  inventory_data_periodic_myopic = inventory_data.copy()
  inventory_data_periodic_myopic.drop(inventory_data_periodic_myopic.index[1:],0,inplace=True)


  for i in range(1,math.ceil(len(y)/R)+1):
    

    y_to_val = y[(i)*R:(i)*R+(R+L)]
        #y_set=y[0:(i+1)*R+(R+L)]
    prediction = myopic_forecast_method(y_to_val)
    y_previous_epoch=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_myopic=peridoic_review(y_previous_epoch,inventory_data_periodic_myopic,prediction,epoch_no)

  # Putting everything into a dataframe
  #=========================================================================================================
  #putting in the main parameters of the scenarios in the data frames
  inventory_data_result_scenarios['Lead time']=L
  inventory_data_result_scenarios['SNR']=SNR
  inventory_data_result_scenarios['Protocol coefficient']=protocol_coef
  inventory_data_result_scenarios['Suppliers minimum capacity']=suppliers_capacity_min
  inventory_data_result_scenarios['Suppliers maximum capacity']=suppliers_capacity_max
  inventory_data_result_scenarios['Inventory capacity']=inventory_capactiy
  inventory_data_result_scenarios['Specified service level']=service_level
    # The total demand/consumption is the same for all methods, so we can pick any of them, in this case, Myopic was picked
  inventory_data_result_scenarios['Total demand/consumption']=inventory_data_periodic_myopic['consumption in last epoch'].sum()

  # Puttin the analysis data into the data frame:
  #=========================================================================================================
  results=(inventory_data_periodic_myopic)
  i=0
  
  #inventory_data_result_scenarios.iloc[[0],[(i*18)+7]]=results['shortage'].sum()
  inventory_data_result_scenarios['Myopic forecast: Shortage']=results['shortage'].sum()
  inventory_data_result_scenarios['Myopic forecast: Inventory level at the end']=results['current'].iloc[-1]
  inventory_data_result_scenarios['Myopic forecast: Achieved service level']=round((1-(results['shortage'] != 0).sum()/len(results['shortage']-1)),2)
  inventory_data_result_scenarios['Myopic forecast: Number of epochs with shortage']=(results['shortage'] != 0).sum()
  inventory_data_result_scenarios['Myopic forecast: Total number of units ordered']=results['order'].sum()
  inventory_data_result_scenarios['Myopic forecast: Number of orders']=(results['order'] != 0).sum()
  inventory_data_result_scenarios['Myopic forecast: Real cost']=results['real cost'].sum()
  inventory_data_result_scenarios['Myopic forecast: Total cost']=results['total cost'].sum()
  # the below command compute average effective percentage bias, by average we specify those epochs that orders can be made
  # if becuase of the lead time, we cannot place an order in a specific epoch (the end epochs) then the percentage bias
  # is not relevent to our study 
  inventory_data_result_scenarios['Myopic forecast: Average effective percentage bias']=results['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean()
  
  # Counting the effective number of epochs with under-forecast
  inventory_data_result_scenarios['Myopic forecast: Average effective percentage bias']=(results['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] < 0).sum()
  
  # Counting the effective number of epochs with over-forecast
  inventory_data_result_scenarios['Myopic forecast: Effective no. of epochs with over-forecast']=(results['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] > 0).sum()
  
  # Calulating the percentage of effective number of epochs with under-forecast
  inventory_data_result_scenarios['Myopic forecast: Effective no. of epochs with over-forecast']=round((inventory_data_result_scenarios.iloc[0][(i*12)+17])*100/((inventory_data_result_scenarios.iloc[0][(i*12)+16])+(inventory_data_result_scenarios.iloc[0][(i*12)+17])),2)

  #Effective RMSE
  inventory_data_result_scenarios['Myopic forecast: Effective no. of epochs with over-forecast']=results['std of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean() 
  #RMSE
  inventory_data_result_scenarios['Myopic forecast: RMSE']=results['std of previous epoch'][1:-1].mean()
  #Effective MAPE
  inventory_data_result_scenarios['Myopic forecast: Effective MAPE']=results['MAPE'][1:math.floor((len(y)-L)/R)+1].mean() 
  #MAPE
  inventory_data_result_scenarios['Myopic forecast: MAPE']=results['MAPE'][1:-1].mean()
  #Effective MAE
  inventory_data_result_scenarios['Myopic forecast: Effective MAE']=results['MAE'][1:math.floor((len(y)-L)/R)+1].mean() 
  #MAE
  inventory_data_result_scenarios['Myopic forecast: MAE']=results['MAE'][1:-1].mean()

  Myopic_epochs_bias=((results.iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Myopic percentage bias of epoch: ')
  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             Myopic_epochs_bias],axis=1)

  Myopic_epochs_shortage=(results.iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Myopic shortage of epoch: ')
  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             Myopic_epochs_shortage],axis=1)

  Myopic_epochs_leftover_in=(results.iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Myopic left-over inventory of epoch: ')
  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             Myopic_epochs_leftover_in],axis=1)
  
  Myopic_order=(results.iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Myopic order of epoch: ')
  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              Myopic_order],axis=1)
  
  Myopic_order_actual=(results.iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Myopic order actual of epoch: ')
  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              Myopic_order_actual],axis=1)

  #inventory_data_result_scenarios  
  complete_inventory_results_scenarios_myopic=complete_inventory_results_scenarios_myopic.append([inventory_data_result_scenarios],ignore_index=True)

#Exporting the data into the a csv file which will be saved into the drive
complete_inventory_results_scenarios_myopic.to_csv('Inventory management results with %s scenarios Myopic.csv'%(no_scenarios))

"""#### Results

###### Importing Data
"""

complete_inventory_results_scenarios_Myopic

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
complete_inventory_results_scenarios_Myopic=pd.read_csv('Inventory management results with 10000 scenarios Myopic.csv')
#setting the date as the index of the dataframe
complete_inventory_results_scenarios_Myopic=complete_inventory_results_scenarios_Myopic.drop(columns='Unnamed: 0')

"""###### Data filtering"""

#creating a separet daraframe in order to not touch the original copy
df_temp_present=complete_inventory_results_scenarios_Myopic.copy()

# SHORTAGE
#============================================
# filtering the data for sepcified category
shortage_epoch_df_Myopic=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('Myopic shortage of epoch')]]

# Left-over inventory
#============================================
# filtering the data for sepcified category
leftover_epoch_df_Myopic=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('Myopic left-over inventory of epoch')]]
# Bais-over inventory
#============================================
# filtering the data for sepcified category
bias_epoch_df_Myopic=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('Myopic percentage bias of epoch')]]

grouped_demand=sim_demand.reset_index().rename(columns = {'index':'date'}, inplace = False)
grouped_demand['Date'] = pd.to_datetime(grouped_demand['Date'],format='%Y-%m-%d')
grouped_demand=(grouped_demand.groupby(grouped_demand.index // R).sum())
# adding a row of zero for time zero
grouped_demand.loc[len(grouped_demand)] = 0
grouped_demand = round(grouped_demand.shift()).apply(np.int64)
grouped_demand.loc[0] = 0
grouped_demand=grouped_demand.T.replace(0,1)
grouped_demand

"""###### Shortage vs epoch"""

def graph_generator_for_epochs_Myopic_1(data1,data2,data3,data4,y_axis_name,data_type):
  fig = plt.figure(figsize=(15,8))
  ax = fig.add_subplot(111)

  #taking the mean of each column and removing the epochs that order cannot be placed (based on the lead time)
  y_bias_SIR=data1.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Holt=data2.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Naive=data3.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Myopic=data4.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  x=range(len(y_bias_SIR))

  #spine placement data centered
  ax.spines['left'].set_position(('data', 0.0))
  ax.spines['bottom'].set_position(('data', 0.0))
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')

  plt.plot(x,y_bias_SIR, color = 'blue',label='SEIRHD',marker='^')
  plt.plot(x,y_bias_Holt, color = 'green',label='Holt',marker='^')
  plt.plot(x,y_bias_Naive, color = 'red',label='Naive',marker='^')
  plt.plot(x,y_bias_Myopic, color = 'black',label='Myopic',marker='^')



  plt.legend(['SEIRHD','Holt','Naive','Myopic'])

  plt.xlabel('Epoch',fontsize=16)
  plt.ylabel(y_axis_name,fontsize=16)
  #plt.title(title,fontsize=20)
  #plt.savefig('%s%% data,%s '%(data_type,y_axis_name))

  return

def graph_generator_for_epochs_Myopic(data1,y_axis_name,data_type):
  fig = plt.figure(figsize=(15,8))
  ax = fig.add_subplot(111)

  #taking the mean of each column and removing the epochs that order cannot be placed (based on the lead time)
  y_bias_Myopic=data1.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]

  x=range(len(y_bias_Myopic))

  #spine placement data centered
  ax.spines['left'].set_position(('data', 0.0))
  ax.spines['bottom'].set_position(('data', 0.0))
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')

  plt.plot(x,y_bias_Myopic, color = 'blue',label='SEIRHD',marker='^')
  #plt.plot(x,y_bias_Holt, color = 'green',label='Holt',marker='^')
  #plt.plot(x,y_bias_Naive, color = 'red',label='Naive',marker='^')



  plt.legend(['Myopic'])

  plt.xlabel('Epoch',fontsize=16)
  plt.ylabel(y_axis_name,fontsize=16)
  #plt.title(title,fontsize=20)
  #plt.savefig('%s%% data,%s '%(data_type,y_axis_name))

  return

shortage_epoch_df_Myopic_relative_to_demand=100*shortage_epoch_df_Myopic.div(grouped_demand.values)

#10,000



graph_generator_for_epochs_Myopic(shortage_epoch_df_Myopic_relative_to_demand,'Shortage','Normal')

"""###### Left-over inventory vs epoch"""

leftover_epoch_df_Myopic_relative_to_demand=100*leftover_epoch_df_Myopic.div(grouped_demand.values)

#10,000

leftover_epoch_df_Myopic_relative_to_demand.iloc[:,0]=0

graph_generator_for_epochs_Myopic(leftover_epoch_df_Myopic_relative_to_demand,'Shortage','Normal')

"""## Scenario I: with random ranges of parameters

#### Parameter selection
"""

# In this section the parameters for ranges of scenarios are defined
#==============================================================================

# How many scenarios do you want?
no_scenarios=100

# SUPPLIER'S CONSTRAINTS:
#====================================
# How many items are in a box:
item_in_box=12
# How many items are in a pallet:
box_in_pallet=1

item_in_pallet= item_in_box * box_in_pallet

# Minimum order
# what is the lower limit for minimum pallet order:
lower_limit_minimum_pallet=1 
# what is the upper limit for minimum pallet order:
upper_limit_minimum_pallet=10 

# Maximum order
# what is the lower limit for maximum pallet order:
lower_limit_maximum_pallet=200 
# what is the upper limit for maximum pallet order:
upper_limit_maximum_pallet=400

# Lead time
# what is the lower limit for lead time of the order:
lower_limit_L=5
# what is the upper limit for lead time of the order:
upper_limit_L=30

# CONSUMPTION'S CONSTRAINTS:
#====================================
# What is the lower limit of coefficient of consumption:
lower_limit_CC=3
# What is the upper limit of coefficient of consumption:
upper_limit_CC=7


# INVENTORY MANAGEMENT CONSTRAINTS:
#====================================
# Capacity of inventory
# What is the lower limit of invernoty capacity (in terms of number of pallet):
lower_limit_invernoty_capacity= 3000
# What is the upper limit of invernoty capacity (in terms of number of pallet):
upper_limit_invernoty_capacity= 6000

# Service level
# What is the lower limit of serivce level (in percent (%)):
lower_limit_service_level= 95
# What is the upper limit of serivce level (in percent (%))
upper_limit_service_level= 99.9


#  DATA SIMULATION CONSTRAINTS:
#====================================

#Do you want to have noise added to the data?
#     'yes'     if you want to add noise to the data 
#     'no'      if you do NOT want to add noise
noise_addition='yes'


# Noise generation 
# What is the lower limit of SNR:
lower_limit_SNR= 2
# What is the upper limit of SNR:
upper_limit_SNR= 10

# Deviation from CC
#SIR_forecast_CC_dev=[0]

"""#### Scenario creator"""

# Forecasting Hospitalization with SIR model
# IMPORTANT NOTE: USER INPUT
# ==============================================================================================

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
SIR_forecast_df=pd.read_csv('forecast SIR without cumulative-complete 18 epochs.csv')
#setting the date as the index of the dataframe
SIR_forecast_df=SIR_forecast_df.set_index(y.index).drop(columns='Date')
#rounding up the value of forecast in the data frame 
#SIR_forecast_df=np.ceil(SIR_forecast_df)

#this code will pass a clean table to the loop by removing all the rows of the dataframes

inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()
complete_inventory_results_scenarios=inventory_data_result_scenarios.iloc[0:0].copy()
#inventory_data_result_scenarios = inventory_data_result_scenarios.iloc[0:0].copy()



for scenarios in range(1):#no_scenarios):
  # Using uniform distribution to randomly choose a value for the following parameters
  inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()

  minimum_pallet = complete_sim_scenarios['Min pallet'][scenarios]
  maximum_pallet = complete_sim_scenarios['Max pallet'][scenarios]
  protocol_coef = complete_sim_scenarios['Protocol coefficient'][scenarios]
  L = complete_sim_scenarios['Lead time'][scenarios]
  inventory_capactiy_in_pallet = complete_sim_scenarios['Inventory capacity (in pallet)'][scenarios]
  service_level = complete_sim_scenarios['Specified service level'][scenarios]
  SNR = complete_sim_scenarios['SNR'][scenarios]
  

  # Calculating the following varibales
  item_in_pallet= item_in_box * box_in_pallet
  suppliers_capacity_min= item_in_pallet * minimum_pallet
  suppliers_capacity_max= item_in_pallet * maximum_pallet
  inventory_capactiy= inventory_capactiy_in_pallet * item_in_pallet


  #this code will pass a clean table to the loop by removing all the rows of the dataframes
  complete_inventory_results=inventory_data_result.iloc[0:0].copy()
  inventory_data_result = inventory_data_result.iloc[0:0].copy()



    #DATA SIMULATION
    #==============================================================================
    #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
    #noise recalculation should only be applied if we have have different coefficient of protocol, 
    #otherwise noise should stya the same, thats why it is outside of the loop
    
  provincial_data_CIHI['daily_consumption']=sim_demand.iloc[:,scenarios]
  provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()


      # >>>USER INPUT<<<
      # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
      # ======================================================================================================
  forecast_data_1 = provincial_data_CIHI.copy()

      # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
  consumption_date_type='daily_consumption'

  y = forecast_data_1[consumption_date_type]
  y_cumulative = forecast_data_1['cumulative_daily_consumption']

      # PERIODIC REVIEW SYSTEM with NAIVE forecasting
      #====================================================================================================================

  inventory_data_periodic_naive = inventory_data.copy()
  inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)


  for i in range(1,math.ceil(len(y)/R)+1):
    y_to_train = y[max(0,i-2)*R:(i)*R]

    y_to_val = y[(i)*R:(i)*R+(R+L)]
        #y_set=y[0:(i+1)*R+(R+L)]
    prediction = naive_forecast_method(y_to_train,y_to_val)
    y_previous_epoch=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_naive=peridoic_review(y_previous_epoch,inventory_data_periodic_naive,prediction,epoch_no)

      # PERIODIC REVIEW SYSTEM with HOLT forecasting
      #====================================================================================================================

  inventory_data_periodic_Holt = inventory_data
  inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

      #hospitalization_data_type='daily_consumption'

  try:
    hospitalization_data_type=hospitalization_data_type
  except NameError:
    hospitalization_data_type='daily_consumption'

  if hospitalization_data_type=='daily_consumption':
    y_holt=y
  else:
    y_holt=y_cumulative


  for i in range(1,math.ceil(len(y)/R)+1):
    y_to_train = y_holt[0:(i)*R]
    y_to_val = y_holt[(i)*R:(i)*R+(R+L)]
    y_set=y_holt[0:(i)*R+(R+L)]
    y_previous_epoch_holt=y_holt[(i-1)*R:(i)*R]
    prediction = holt(y_set, y_to_train,y_to_val,i,y_previous_epoch_holt)
    y_previous_epoch=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_Holt=peridoic_review(y_previous_epoch,inventory_data_periodic_Holt,prediction,epoch_no)

    # PERIODIC REVIEW SYSTEM with SIR forecasting model
    #=========================================================================================================

  inventory_data_periodic_SIR = inventory_data.copy()
  inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)


  for i in range(1,math.ceil(len(y)/R)+1):
    y_to_train = y[0:(i)*R]
    y_to_val = y[(i)*R:(i)*R+(R+L)]
    y_set=y[0:(i)*R+(R+L)]
    prediction = SIR_forecast_model(i,y_to_val)
    y_previous_epoch=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction,epoch_no)

  # Putting everything into a dataframe
  #=========================================================================================================
  #putting in the main parameters of the scenarios in the data frames
  inventory_data_result_scenarios['Lead time']=L
  inventory_data_result_scenarios['SNR']=SNR
  inventory_data_result_scenarios['Protocol coefficient']=protocol_coef
  inventory_data_result_scenarios['Suppliers minimum capacity']=suppliers_capacity_min
  inventory_data_result_scenarios['Suppliers maximum capacity']=suppliers_capacity_max
  inventory_data_result_scenarios['Inventory capacity']=inventory_capactiy
  inventory_data_result_scenarios['Specified service level']=service_level
    # The total demand/consumption is the same for all methods, so we can pick any of them, in this case, SIR was picked
  inventory_data_result_scenarios['Total demand/consumption']=inventory_data_periodic_SIR['consumption in last epoch'].sum()

  # Puttin the analysis data into the data frame:
  #=========================================================================================================
  results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
  
  for i in range(len(results)):
    inventory_data_result_scenarios.iloc[[0],[(i*18)+7]]=results[i]['shortage'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+8]]=results[i]['current'].iloc[-1],
    inventory_data_result_scenarios.iloc[[0],[(i*18)+9]]=round((1-(results[i]['shortage'] != 0).sum()/len(results[i]['shortage']-1)),2)
    inventory_data_result_scenarios.iloc[[0],[(i*18)+10]]=(results[i]['shortage'] != 0).sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+11]]=results[i]['order'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+12]]=(results[i]['order'] != 0).sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+13]]=results[i]['real cost'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+14]]=results[i]['total cost'].sum()
    # the below command compute average effective percentage bias, by average we specify those epochs that orders can be made
    # if becuase of the lead time, we cannot place an order in a specific epoch (the end epochs) then the percentage bias
    # is not relevent to our study 
    inventory_data_result_scenarios.iloc[[0],[(i*18)+15]]=results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean()
    
    # Counting the effective number of epochs with under-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+16]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] < 0).sum()
    
    # Counting the effective number of epochs with over-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+17]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] > 0).sum()
    
    # Calulating the percentage of effective number of epochs with under-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+18]]=round((inventory_data_result_scenarios.iloc[0][(i*12)+17])*100/((inventory_data_result_scenarios.iloc[0][(i*12)+16])+(inventory_data_result_scenarios.iloc[0][(i*12)+17])),2)

    #Effective RMSE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+19]]=results[i]['std of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean() 
    #RMSE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+20]]=results[i]['std of previous epoch'][1:-1].mean()
    #Effective MAPE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+21]]=results[i]['MAPE'][1:math.floor((len(y)-L)/R)+1].mean() 
    #MAPE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+22]]=results[i]['MAPE'][1:-1].mean()
    #Effective MAE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+23]]=results[i]['MAE'][1:math.floor((len(y)-L)/R)+1].mean() 
    #MAE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+24]]=results[i]['MAE'][1:-1].mean()

  SIR_epochs_bias=((results[0].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('SIR percentage bias of epoch: ')
  Holt_epochs_bias=((results[1].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Holt percentage bias of epoch: ')
  Naive_epochs_bias=((results[2].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Naive percentage bias of epoch: ')
  
  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_bias,
                                             Holt_epochs_bias,
                                             Naive_epochs_bias],axis=1)

  SIR_epochs_shortage=(results[0].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('SIR shortage of epoch: ')
  Holt_epochs_shortage=(results[1].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Holt shortage of epoch: ')
  Naive_epochs_shortage=(results[2].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Naive shortage of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_shortage,
                                             Holt_epochs_shortage,
                                             Naive_epochs_shortage],axis=1)

  SIR_epochs_leftover_in=(results[0].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('SIR left-over inventory of epoch: ')
  Holt_epochs_leftover_in=(results[1].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Holt left-over inventory of epoch: ')
  Naive_epochs_leftover_in=(results[2].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Naive left-over inventory of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_leftover_in,
                                             Holt_epochs_leftover_in,
                                             Naive_epochs_leftover_in],axis=1)
  SIR_order=(results[0].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('SIR order of epoch: ')
  Holt_order=(results[1].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Holt order of epoch: ')
  Naive_order=(results[2].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Naive order of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_order,
                                              Holt_order,
                                              Naive_order],axis=1)
  
  SIR_order_actual=(results[0].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('SIR order actual of epoch: ')
  Holt_order_actual=(results[1].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Holt order actual of epoch: ')
  Naive_order_actual=(results[2].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Naive order actual of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_order_actual,
                                              Holt_order_actual,
                                              Naive_order_actual],axis=1)

  #inventory_data_result_scenarios  
  complete_inventory_results_scenarios=complete_inventory_results_scenarios.append([inventory_data_result_scenarios],ignore_index=True)

#Exporting the data into the a csv file which will be saved into the drive
complete_inventory_results_scenarios.to_csv('Inventory management results with %s scenarios(un-touched data).csv'%(no_scenarios))

"""#### Results

##### Import Data
"""

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
complete_inventory_results_scenarios=pd.read_csv('Inventory management results with 10000 scenarios(un-touched data).csv')
#setting the date as the index of the dataframe
complete_inventory_results_scenarios=complete_inventory_results_scenarios.drop(columns='Unnamed: 0')
complete_inventory_results_scenarios=complete_inventory_results_scenarios.iloc[:1000,:]

"""##### Presentation of results

###### Data filtering
"""

#creating a separet daraframe in order to not touch the original copy
df_temp_present=complete_inventory_results_scenarios.copy()

# SHORTAGE
#============================================
# filtering the data for sepcified category
shortage_epoch_df_SIR=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('SIR shortage of epoch')]]
shortage_epoch_df_Holt=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('Holt shortage of epoch')]]
shortage_epoch_df_Naive=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('Naive shortage of epoch')]]


# Left-over inventory
#============================================
# filtering the data for sepcified category
leftover_epoch_df_SIR=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('SIR left-over inventory of epoch')]]
leftover_epoch_df_Holt=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('Holt left-over inventory of epoch')]]
leftover_epoch_df_Naive=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('Naive left-over inventory of epoch')]]

# Bais-over inventory
#============================================
# filtering the data for sepcified category
bias_epoch_df_SIR=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('SIR percentage bias of epoch')]]
bias_epoch_df_Holt=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('Holt percentage bias of epoch')]]
bias_epoch_df_Naive=df_temp_present[df_temp_present.columns[pd.Series(df_temp_present.columns).str.startswith('Naive percentage bias of epoch')]]

grouped_demand=sim_demand.reset_index().rename(columns = {'index':'date'}, inplace = False)
grouped_demand['Date'] = pd.to_datetime(grouped_demand['Date'],format='%Y-%m-%d')
grouped_demand=(grouped_demand.groupby(grouped_demand.index // R).sum())
# adding a row of zero for time zero
grouped_demand.loc[len(grouped_demand)] = 0
grouped_demand = round(grouped_demand.shift()).apply(np.int64)
grouped_demand.loc[0] = 0
grouped_demand=grouped_demand.T.replace(0,1)
grouped_demand=grouped_demand.iloc[:1000,:]

"""###### Shortage vs epoch"""

shortage_epoch_df_SIR_relative_to_demand=100*shortage_epoch_df_SIR.div(grouped_demand.values)
shortage_epoch_df_Holt_relative_to_demand=100*shortage_epoch_df_Holt.div(grouped_demand.values)
shortage_epoch_df_Naive_relative_to_demand=100*shortage_epoch_df_Naive.div(grouped_demand.values)

graph_generator_for_epochs_Myopic_1(shortage_epoch_df_SIR_relative_to_demand,
                                    shortage_epoch_df_Holt_relative_to_demand,
                                    shortage_epoch_df_Naive_relative_to_demand,
                                    shortage_epoch_df_Myopic_relative_to_demand,'Shortage','Normal')

#10,000

shortage_epoch_df_SIR_relative_to_demand=100*shortage_epoch_df_SIR.div(grouped_demand.values)
shortage_epoch_df_Holt_relative_to_demand=100*shortage_epoch_df_Holt.div(grouped_demand.values)
shortage_epoch_df_Naive_relative_to_demand=100*shortage_epoch_df_Naive.div(grouped_demand.values)


graph_generator_for_epochs(shortage_epoch_df_SIR_relative_to_demand,
                           shortage_epoch_df_Holt_relative_to_demand,
                           shortage_epoch_df_Naive_relative_to_demand,'Shortage','Normal')

#100
graph_generator_for_epochs(shortage_epoch_df_SIR,shortage_epoch_df_Holt,shortage_epoch_df_Naive,'Shortage','normal')

"""###### Left-over inventory vs epoch"""

leftover_epoch_df_SIR_relative_to_demand=100*leftover_epoch_df_SIR.div(grouped_demand.values)
leftover_epoch_df_Holt_relative_to_demand=100*leftover_epoch_df_Holt.div(grouped_demand.values)
leftover_epoch_df_Naive_relative_to_demand=100*leftover_epoch_df_Naive.div(grouped_demand.values)

leftover_epoch_df_SIR_relative_to_demand.iloc[:,0]=0
leftover_epoch_df_Holt_relative_to_demand.iloc[:,0]=0
leftover_epoch_df_Naive_relative_to_demand.iloc[:,0]=0

#10,000
graph_generator_for_epochs_Myopic_1(leftover_epoch_df_SIR_relative_to_demand,
                                    leftover_epoch_df_Holt_relative_to_demand,
                                    leftover_epoch_df_Naive_relative_to_demand,
                                    leftover_epoch_df_Myopic_relative_to_demand,'Inventory level','normal')

#10,000
graph_generator_for_epochs(leftover_epoch_df_SIR_relative_to_demand,leftover_epoch_df_Holt_relative_to_demand,leftover_epoch_df_Naive_relative_to_demand,'Inventory level','normal')

#n100
graph_generator_for_epochs(leftover_epoch_df_SIR,leftover_epoch_df_Holt,leftover_epoch_df_Naive,'Inventory level','normal')

"""###### Percentage bias vs epoch"""

#10,000
graph_generator_for_epochs(bias_epoch_df_SIR,bias_epoch_df_Holt,bias_epoch_df_Naive,'Percentage bias')

#100
graph_generator_for_epochs(bias_epoch_df_SIR,bias_epoch_df_Holt,bias_epoch_df_Naive,'Percentage bias')

"""###### Shortage vs. final inventory"""

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)

x_plot=complete_inventory_results_scenarios['SIR forecast: Shortage']
y_plot=complete_inventory_results_scenarios['SIR forecast: Inventory level at the end']
plt.scatter(x_plot, y_plot, color = 'red',label='SIR forecast',marker='o')

x_plot=complete_inventory_results_scenarios['Holt forecast: Shortage']
y_plot=complete_inventory_results_scenarios['Holt forecast: Inventory level at the end']
plt.scatter(x_plot, y_plot, color = 'blue',label='Holt forecast',marker='x')

x_plot=complete_inventory_results_scenarios['Naive forecast: Shortage']
y_plot=complete_inventory_results_scenarios['Naive forecast: Inventory level at the end']
plt.scatter(x_plot, y_plot, color = 'green',label='Naive forecast',marker='^')

plt.title('%s scnearios of inventory management in British Columbia'%(no_scenarios),fontsize=16)
plt.xlabel('Shortage',fontsize=16)
plt.ylabel('Final inventory',fontsize=16)
plt.legend(fontsize=13)

plt.show()

"""## Scenario II: Simulation process "Replenished-Data"; Granularity; Variable Period_info

### Scenario creator
"""

# we will run this just once
variable_period_info_df=pd.read_csv('variable 10000 period info.csv')
variable_period_info_df=variable_period_info_df.drop(columns='Unnamed: 0')
variable_period_info_df=variable_period_info_df.reset_index()
variable_period_info_df['index']=0
variable_period_info_df.columns=list(range(len(variable_period_info_df.columns)))
#variable_period_info_df

# temperoray code:
#================================
#iteration_0=pd.read_csv('result of SIR forecast iteration 0.csv')
#iteration_1=pd.read_csv('result of SIR forecast iteration 1.csv')
#df_SIR_forecast_results_variable=iteration_0.append(iteration_1,sort=False)

# Activate this code:
#================================
#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
df_SIR_forecast_results_variable=pd.read_csv('SIR forecast variable 1000.csv')
#setting the date as the index of the dataframe


#df_SIR_forecast_results_variable=df_SIR_forecast_results_variable.rename(columns={'Unnamed: 0':'Date'})
#setting the date as the index of the dataframe
df_SIR_forecast_results_variable=df_SIR_forecast_results_variable.set_index(df_SIR_forecast_results_variable['date'])

df_SIR_forecast_results_variable.drop(df_SIR_forecast_results_variable.columns[0], axis=1, inplace=True)

start_rep_array=890
end_rep_array=900
end_rep_array

no_scenarios=1000
#this code will pass a clean table to the loop by removing all the rows of the dataframes

#inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()
#complete_inventory_results_scenarios_replenished=inventory_data_result_scenarios.iloc[0:0].copy()
#inventory_data_result_scenarios = inventory_data_result_scenarios.iloc[0:0].copy()

inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()
complete_inventory_results_scenarios_replenished_variable=inventory_data_result_scenarios.iloc[0:0].copy()



for va_rep_no in range(start_rep_array,end_rep_array):

  complete_inventory_results_scenarios_replenished=inventory_data_result_scenarios.iloc[0:0].copy()


  # PERIOD INFO SELECTION
  #=============================================================
  # Selecting period info of the iteration 'scenarios' 
  # Finding the correct period_info for the desired iteration:
  variable_period_info_it_i=(variable_period_info_df.iloc[va_rep_no,:].to_frame()).dropna()
  # cumsum the day to get the day when the data is arriving 
  variable_period_info_it_i['day of data arrival']=(variable_period_info_it_i.iloc[:,0]).cumsum()
  
  # SIR forecast selection
  #=============================================================
  SIR_forecast_df=df_SIR_forecast_results_variable[va_rep_no*len(y):(va_rep_no+1)*len(y)].dropna(axis=1)


  #for scenarios in range(no_scenarios):
  for scenarios in range(no_scenarios):
    # Using uniform distribution to randomly choose a value for the following parameters
    inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()

    minimum_pallet = complete_sim_scenarios['Min pallet'][scenarios]
    maximum_pallet = complete_sim_scenarios['Max pallet'][scenarios]
    protocol_coef = complete_sim_scenarios['Protocol coefficient'][scenarios]
    L = complete_sim_scenarios['Lead time'][scenarios]
    inventory_capactiy_in_pallet = complete_sim_scenarios['Inventory capacity (in pallet)'][scenarios]
    service_level = complete_sim_scenarios['Specified service level'][scenarios]
    SNR = complete_sim_scenarios['SNR'][scenarios]
    

    # Calculating the following varibales
    item_in_pallet= item_in_box * box_in_pallet
    suppliers_capacity_min= item_in_pallet * minimum_pallet
    suppliers_capacity_max= item_in_pallet * maximum_pallet
    inventory_capactiy= inventory_capactiy_in_pallet * item_in_pallet


    #this code will pass a clean table to the loop by removing all the rows of the dataframes
    complete_inventory_results=inventory_data_result.iloc[0:0].copy()
    inventory_data_result = inventory_data_result.iloc[0:0].copy()



      #DATA SIMULATION
      #==============================================================================
      #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
      #noise recalculation should only be applied if we have have different coefficient of protocol, 
      #otherwise noise should stya the same, thats why it is outside of the loop
      
    provincial_data_CIHI['daily_consumption']=sim_demand.iloc[:,scenarios]
    provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()


    # >>>USER INPUT<<<
    # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
    # ======================================================================================================
    forecast_data_1 = provincial_data_CIHI.copy()

        # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
    consumption_date_type='daily_consumption'

    y = forecast_data_1[consumption_date_type]

    # this code will grouped the data based on the period info
    y_replenishement=replenished_func_variable_period_info(y,va_rep_no)  





    # PERIODIC REVIEW SYSTEM with NAIVE forecasting
    #====================================================================================================================

    inventory_data_periodic_naive = inventory_data.copy()
    inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)

    #number of periods that needs to be reviewed to find max value based on the specified period_info
    #no_period_for_naive_max=math.ceil(2*R/period_info)

    for i in range(1,math.ceil(len(y)/R)+1):

      days_in_epoch_i=len(y[0:(i)*R])

      # this line finds how many period_info have passed for the days in the epoch 'i'
      # using the between function, we check the number of days in each period_info to see if they are between (0,days_in_epoch_i)
      # it will return the bolyan value, we will then get the index of those which are 'True' and the max of these values
      # is the last period_info in the epoch i 
      upper_limit_period_info=variable_period_info_it_i['day of data arrival'].between(0,days_in_epoch_i)
      upper_limit_period_info=int((upper_limit_period_info[upper_limit_period_info].index.values ).max())

      lower_limit_period_info=variable_period_info_it_i['day of data arrival'].between(max(0,variable_period_info_it_i['day of data arrival'][upper_limit_period_info]-2*R),days_in_epoch_i)
      lower_limit_period_info=int((lower_limit_period_info[lower_limit_period_info].index.values ).min())

      upper_limit_period_info
      lower_limit_period_info

      upper_day=variable_period_info_it_i['day of data arrival'][upper_limit_period_info]


      if (upper_day-variable_period_info_it_i['day of data arrival'][lower_limit_period_info]) <2*R:
        lower_day=variable_period_info_it_i['day of data arrival'][max(0,lower_limit_period_info-1)]
      else:
        lower_day=variable_period_info_it_i['day of data arrival'][lower_limit_period_info]


      y_to_train = y_replenishement[int(lower_day):int(upper_day)]
      y_to_val = y[(i)*R:(i)*R+(R+L)]
      y_set=y[0:(i)*R+(R+L)]

      prediction_naive = naive_forecast_method_replenished_variable(y_to_train,y_to_val)


      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_naive=peridoic_review(y_previous_epoch,
                                                    inventory_data_periodic_naive,
                                                    prediction_naive,
                                                    epoch_no)

    # PERIODIC REVIEW SYSTEM with HOLT forecasting
    #====================================================================================================================

    inventory_data_periodic_Holt = inventory_data
    inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

        #hospitalization_data_type='daily_consumption'

    try:
      holt_data=holt_data_type
    except NameError:
      holt_data='normal'


    for i in range(1,math.ceil(len(y)/R)+1):
      
      days_in_epoch_i=len(y[0:(i)*R])

      # this line finds how many period_info have passed for the days in the epoch 'i'
      # using the between function, we check the number of days in each period_info to see if they are between (0,days_in_epoch_i)
      # it will return the bolyan value, we will then get the index of those which are 'True' and the max of these values
      # is the last period_info in the epoch i 
      upper_limit_period_info=variable_period_info_it_i['day of data arrival'].between(0,days_in_epoch_i)
      upper_limit_period_info=int((upper_limit_period_info[upper_limit_period_info].index.values ).max())

      upper_day=variable_period_info_it_i['day of data arrival'][upper_limit_period_info]

      y_to_train = y_replenishement[0:int(upper_day)]
      y_to_val = y[(i)*R:(i)*R+(R+L)]
      y_set=y[0:(i)*R+(R+L)]

      prediction_holt = holt_4(y_set, y_to_train,y_to_val)

      y_previous_epoch_holt=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_Holt=peridoic_review(y_previous_epoch_holt,
                                                    inventory_data_periodic_Holt,
                                                    prediction_holt,
                                                    epoch_no)

      # PERIODIC REVIEW SYSTEM with SIR forecasting model
      #=========================================================================================================

    inventory_data_periodic_SIR = inventory_data.copy()
    inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)


    for i in range(1,math.ceil(len(y)/R)+1):
      #y_to_train = y[0:(i)*R]
      #y_to_val = y[(i)*R:(i)*R+(R+L)]
      #y_set=y[0:(i)*R+(R+L)]
      #prediction = SIR_forecast_model(i,y_to_val)

      # number of days in epoch i:
      days_in_epoch_i=len(y[0:(i)*R])

      upper_limit_period_info=variable_period_info_it_i['day of data arrival'].between(0,days_in_epoch_i)
      # vpi: variable period info
      # this is the column where the code has to pick from the 'SIR_forecast_df' of this specific iteration
      upper_epoch_vpi=int((upper_limit_period_info[upper_limit_period_info].index.values ).max())

      y_to_val = y[(i)*R:(i)*R+(R+L)]
      prediction = SIR_forecast_model_variable_period_info(upper_epoch_vpi,y_to_val,i)
      
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction,epoch_no)

    # Putting everything into a dataframe
    #=========================================================================================================
    #putting in the main parameters of the scenarios in the data frames
    inventory_data_result_scenarios['Lead time']=L
    inventory_data_result_scenarios['SNR']=SNR
    inventory_data_result_scenarios['Protocol coefficient']=protocol_coef
    inventory_data_result_scenarios['Suppliers minimum capacity']=suppliers_capacity_min
    inventory_data_result_scenarios['Suppliers maximum capacity']=suppliers_capacity_max
    inventory_data_result_scenarios['Inventory capacity']=inventory_capactiy
    inventory_data_result_scenarios['Specified service level']=service_level
      # The total demand/consumption is the same for all methods, so we can pick any of them, in this case, SIR was picked
    inventory_data_result_scenarios['Total demand/consumption']=inventory_data_periodic_SIR['consumption in last epoch'].sum()

    # Puttin the analysis data into the data frame:
    #=========================================================================================================
    results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
    
    for i in range(len(results)):
      inventory_data_result_scenarios.iloc[[0],[(i*18)+7]]=results[i]['shortage'].sum()
      inventory_data_result_scenarios.iloc[[0],[(i*18)+8]]=results[i]['current'].iloc[-1]
      inventory_data_result_scenarios.iloc[[0],[(i*18)+9]]=round((1-(results[i]['shortage'] != 0).sum()/len(results[i]['shortage']-1)),2)
      inventory_data_result_scenarios.iloc[[0],[(i*18)+10]]=(results[i]['shortage'] != 0).sum()
      inventory_data_result_scenarios.iloc[[0],[(i*18)+11]]=results[i]['order'].sum()
      inventory_data_result_scenarios.iloc[[0],[(i*18)+12]]=(results[i]['order'] != 0).sum()
      inventory_data_result_scenarios.iloc[[0],[(i*18)+13]]=results[i]['real cost'].sum()
      inventory_data_result_scenarios.iloc[[0],[(i*18)+14]]=results[i]['total cost'].sum()
      # the below command compute average effective percentage bias, by average we specify those epochs that orders can be made
      # if becuase of the lead time, we cannot place an order in a specific epoch (the end epochs) then the percentage bias
      # is not relevent to our study 
      inventory_data_result_scenarios.iloc[[0],[(i*18)+15]]=results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean()
      
      # Counting the effective number of epochs with under-forecast
      inventory_data_result_scenarios.iloc[[0],[(i*18)+16]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] < 0).sum()
      
      # Counting the effective number of epochs with over-forecast
      inventory_data_result_scenarios.iloc[[0],[(i*18)+17]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] > 0).sum()
      
      # Calulating the percentage of effective number of epochs with under-forecast
      inventory_data_result_scenarios.iloc[[0],[(i*18)+18]]=round((inventory_data_result_scenarios.iloc[0][(i*18)+17])*100/((inventory_data_result_scenarios.iloc[0][(i*18)+16])+(inventory_data_result_scenarios.iloc[0][(i*18)+17])),2)

      #Effective RMSE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+19]]=results[i]['std of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean() 
      #RMSE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+20]]=results[i]['std of previous epoch'][1:-1].mean()
      #Effective MAPE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+21]]=results[i]['MAPE'][1:math.floor((len(y)-L)/R)+1].mean() 
      #MAPE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+22]]=results[i]['MAPE'][1:-1].mean()
      #Effective MAE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+23]]=results[i]['MAE'][1:math.floor((len(y)-L)/R)+1].mean() 
      #MAE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+24]]=results[i]['MAE'][1:-1].mean()

    SIR_epochs_bias=((results[0].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('SIR percentage bias of epoch: ')
    Holt_epochs_bias=((results[1].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Holt percentage bias of epoch: ')
    Naive_epochs_bias=((results[2].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Naive percentage bias of epoch: ')
    
    inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_epochs_bias,
                                              Holt_epochs_bias,
                                              Naive_epochs_bias],axis=1)

    SIR_epochs_shortage=(results[0].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('SIR shortage of epoch: ')
    Holt_epochs_shortage=(results[1].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Holt shortage of epoch: ')
    Naive_epochs_shortage=(results[2].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Naive shortage of epoch: ')

    inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_epochs_shortage,
                                              Holt_epochs_shortage,
                                              Naive_epochs_shortage],axis=1)

    SIR_epochs_leftover_in=(results[0].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('SIR left-over inventory of epoch: ')
    Holt_epochs_leftover_in=(results[1].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Holt left-over inventory of epoch: ')
    Naive_epochs_leftover_in=(results[2].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Naive left-over inventory of epoch: ')

    inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_epochs_leftover_in,
                                              Holt_epochs_leftover_in,
                                              Naive_epochs_leftover_in],axis=1)
    SIR_order=(results[0].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('SIR order of epoch: ')
    Holt_order=(results[1].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Holt order of epoch: ')
    Naive_order=(results[2].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Naive order of epoch: ')

    inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                                SIR_order,
                                                Holt_order,
                                                Naive_order],axis=1)
    
    SIR_order_actual=(results[0].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('SIR order actual of epoch: ')
    Holt_order_actual=(results[1].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Holt order actual of epoch: ')
    Naive_order_actual=(results[2].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Naive order actual of epoch: ')

    inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                                SIR_order_actual,
                                                Holt_order_actual,
                                                Naive_order_actual],axis=1)

    #inventory_data_result_scenarios  
    complete_inventory_results_scenarios_replenished=complete_inventory_results_scenarios_replenished.append([inventory_data_result_scenarios],ignore_index=True)
  
  complete_inventory_results_scenarios_replenished_variable=complete_inventory_results_scenarios_replenished_variable.append([complete_inventory_results_scenarios_replenished],ignore_index=True)




complete_inventory_results_scenarios_replenished_variable.to_csv('Results of var replenished %dto%d.csv'%(start_rep_array,end_rep_array))

complete_inventory_results_scenarios_replenished_variable.to_csv('Results of var replenished %dto%d.csv'%(start_rep_array,end_rep_array))

complete_inventory_results_scenarios_replenished_variable

complete_inventory_results_scenarios_replenished

"""### Imporing data"""

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
complete_inventory_results_scenarios_replenished=pd.read_csv('Inventory management results with 1000 scenarios replenished varibale.csv')
#setting the date as the index of the dataframe
complete_inventory_results_scenarios_replenished=complete_inventory_results_scenarios_replenished.drop(columns='Unnamed: 0')

"""### Results

#### Presentation of results

##### Data filtering
"""

#creating a separet daraframe in order to not touch the original copy
df_temp_present_replenished_var=complete_inventory_results_scenarios_replenished.copy()

# SHORTAGE
#============================================
# filtering the data for sepcified category
shortage_epoch_df_SIR_replenished_var=df_temp_present_replenished_var[df_temp_present_replenished_var.columns[pd.Series(df_temp_present_replenished_var.columns).str.startswith('SIR shortage of epoch')]]
shortage_epoch_df_Holt_replenished_var=df_temp_present_replenished_var[df_temp_present_replenished_var.columns[pd.Series(df_temp_present_replenished_var.columns).str.startswith('Holt shortage of epoch')]]
shortage_epoch_df_Naive_replenished_var=df_temp_present_replenished_var[df_temp_present_replenished_var.columns[pd.Series(df_temp_present_replenished_var.columns).str.startswith('Naive shortage of epoch')]]


# Left-over inventory
#============================================
# filtering the data for sepcified category
leftover_epoch_df_SIR_replenished_var=df_temp_present_replenished_var[df_temp_present_replenished_var.columns[pd.Series(df_temp_present_replenished_var.columns).str.startswith('SIR left-over inventory of epoch')]]
leftover_epoch_df_Holt_replenished_var=df_temp_present_replenished_var[df_temp_present_replenished_var.columns[pd.Series(df_temp_present_replenished_var.columns).str.startswith('Holt left-over inventory of epoch')]]
leftover_epoch_df_Naive_replenished_var=df_temp_present_replenished_var[df_temp_present_replenished_var.columns[pd.Series(df_temp_present_replenished_var.columns).str.startswith('Naive left-over inventory of epoch')]]

# Bais-over inventory
#============================================
# filtering the data for sepcified category
bias_epoch_df_SIR_replenished_var=df_temp_present_replenished_var[df_temp_present_replenished_var.columns[pd.Series(df_temp_present_replenished_var.columns).str.startswith('SIR percentage bias of epoch')]]
bias_epoch_df_Holt_replenished_var=df_temp_present_replenished_var[df_temp_present_replenished_var.columns[pd.Series(df_temp_present_replenished_var.columns).str.startswith('Holt percentage bias of epoch')]]
bias_epoch_df_Naive_replenished_var=df_temp_present_replenished_var[df_temp_present_replenished_var.columns[pd.Series(df_temp_present_replenished_var.columns).str.startswith('Naive percentage bias of epoch')]]

grouped_demand=sim_demand.reset_index().rename(columns = {'index':'date'}, inplace = False)
grouped_demand['Date'] = pd.to_datetime(grouped_demand['Date'],format='%Y-%m-%d')
grouped_demand=(grouped_demand.groupby(grouped_demand.index // R).sum())
# adding a row of zero for time zero
grouped_demand.loc[len(grouped_demand)] = 0
grouped_demand = round(grouped_demand.shift()).apply(np.int64)
grouped_demand.loc[0] = 0
grouped_demand=grouped_demand.T.replace(0,1)
grouped_demand=grouped_demand.iloc[:1000,:]

"""##### Shortage vs epoch"""

shortage_epoch_df_SIR_relative_to_demand_replenished_var=100*shortage_epoch_df_SIR_replenished_var.div(grouped_demand.values)
shortage_epoch_df_Holt_relative_to_demand_replenished_var=100*shortage_epoch_df_Holt_replenished_var.div(grouped_demand.values)
shortage_epoch_df_Naive_relative_to_demand_replenished_var=100*shortage_epoch_df_Naive_replenished_var.div(grouped_demand.values)

graph_generator_for_epochs(shortage_epoch_df_SIR_relative_to_demand_replenished_var,
                           shortage_epoch_df_Holt_relative_to_demand_replenished_var,
                           shortage_epoch_df_Naive_relative_to_demand_replenished_var,'Shortage','Relative')

graph_generator_for_epochs(shortage_epoch_df_SIR_replenished_var,
                           shortage_epoch_df_Holt_replenished_var,
                           shortage_epoch_df_Naive_replenished_var,'Shortage','Varibale period info')

"""##### Left-over inventory vs epoch"""

leftover_epoch_df_SIR_relative_to_demand_replenished_var=100*leftover_epoch_df_SIR_replenished_var.div(grouped_demand.values)
leftover_epoch_df_Holt_relative_to_demand_replenished_var=100*leftover_epoch_df_Holt_replenished_var.div(grouped_demand.values)
leftover_epoch_df_Naive_relative_to_demand_replenished_var=100*leftover_epoch_df_Naive_replenished_var.div(grouped_demand.values)

leftover_epoch_df_SIR_relative_to_demand_replenished_var.iloc[:,0]=0
leftover_epoch_df_Holt_relative_to_demand_replenished_var.iloc[:,0]=0
leftover_epoch_df_Naive_relative_to_demand_replenished_var.iloc[:,0]=0

# 10000
graph_generator_for_epochs(leftover_epoch_df_SIR_relative_to_demand_replenished_var,
                           leftover_epoch_df_Holt_relative_to_demand_replenished_var,
                           leftover_epoch_df_Naive_relative_to_demand_replenished_var,'Inventory level','Variable period info')

# 10000
graph_generator_for_epochs(leftover_epoch_df_SIR_replenished_var,
                           leftover_epoch_df_Holt_replenished_var,
                           leftover_epoch_df_Naive_replenished_var,'Inventory level','Variable period info')

"""## Scenario II: Simulation process "Replenished-Data"; Granularity; Fixed Period_info

#### Parameters
"""

# In this section the parameters for ranges of scenarios are defined
#==============================================================================

# Choose what type of data is to be used by holt method:
# normal
# cumulative
holt_data_type='normal'

# How many scenarios do you want?
no_scenarios=1

# SUPPLIER'S CONSTRAINTS:
#====================================
# How many items are in a box:
item_in_box=12
# How many box are in a pallet:
box_in_pallet=1

item_in_pallet= item_in_box * box_in_pallet

# Minimum order
# what is the lower limit for minimum pallet order:
lower_limit_minimum_pallet=1 
# what is the upper limit for minimum pallet order:
upper_limit_minimum_pallet=10 

# Maximum order
# what is the lower limit for maximum pallet order:
lower_limit_maximum_pallet=200 
# what is the upper limit for maximum pallet order:
upper_limit_maximum_pallet=400

# Lead time
# what is the lower limit for lead time of the order:
lower_limit_L=5
# what is the upper limit for lead time of the order:
upper_limit_L=30

# CONSUMPTION'S CONSTRAINTS:
#====================================
# What is the lower limit of coefficient of consumption:
lower_limit_CC=3
# What is the upper limit of coefficient of consumption:
upper_limit_CC=7


# INVENTORY MANAGEMENT CONSTRAINTS:
#====================================
# Capacity of inventory
# What is the lower limit of invernoty capacity (in terms of number of pallet):
lower_limit_invernoty_capacity= 3000
# What is the upper limit of invernoty capacity (in terms of number of pallet):
upper_limit_invernoty_capacity= 6000

# Service level
# What is the lower limit of serivce level (in percent (%)):
lower_limit_service_level= 95
# What is the upper limit of serivce level (in percent (%))
upper_limit_service_level= 99.9


#  DATA SIMULATION CONSTRAINTS:
#====================================

#Do you want to have noise added to the data?
#     'yes'     if you want to add noise to the data 
#     'no'      if you do NOT want to add noise
noise_addition='yes'


# Noise generation 
# What is the lower limit of SNR:
lower_limit_SNR= 2
# What is the upper limit of SNR:
upper_limit_SNR= 10

# Deviation from CC
#SIR_forecast_CC_dev=[0]

#  Replenishment info 
#====================================

# how mony days should be between the arrival of info (replenishement mode)
period_info=8

"""#### Scenario creator"""

period_info=4

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
SIR_forecast_df=pd.read_csv('forecast SIR with period info of %d.csv'%period_info)
#setting the date as the index of the dataframe


SIR_forecast_df=SIR_forecast_df.rename(columns={'Unnamed: 0':'Date'})
#setting the date as the index of the dataframe
SIR_forecast_df=SIR_forecast_df.set_index(SIR_forecast_df['Date'])

SIR_forecast_df.drop(SIR_forecast_df.columns[0], axis=1, inplace=True)

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
SIR_forecast_df=pd.read_csv('forecast SIR without cumulative-complete with period_info %d.csv'%period_info)
#setting the date as the index of the dataframe


SIR_forecast_df=SIR_forecast_df.rename(columns={'Unnamed: 0':'Date'})
#setting the date as the index of the dataframe
SIR_forecast_df=SIR_forecast_df.set_index(SIR_forecast_df['Date'])

SIR_forecast_df.drop(SIR_forecast_df.columns[0], axis=1, inplace=True)





#this code will pass a clean table to the loop by removing all the rows of the dataframes

inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()
complete_inventory_results_scenarios_replenished=inventory_data_result_scenarios.iloc[0:0].copy()
#inventory_data_result_scenarios = inventory_data_result_scenarios.iloc[0:0].copy()

no_scenarios=1000

#for scenarios in range(no_scenarios):
for scenarios in range(no_scenarios):
  # Using uniform distribution to randomly choose a value for the following parameters
  inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()

  minimum_pallet = complete_sim_scenarios['Min pallet'][scenarios]
  maximum_pallet = complete_sim_scenarios['Max pallet'][scenarios]
  protocol_coef = complete_sim_scenarios['Protocol coefficient'][scenarios]
  L = complete_sim_scenarios['Lead time'][scenarios]
  inventory_capactiy_in_pallet = complete_sim_scenarios['Inventory capacity (in pallet)'][scenarios]
  service_level = complete_sim_scenarios['Specified service level'][scenarios]
  SNR = complete_sim_scenarios['SNR'][scenarios]
  

  # Calculating the following varibales
  item_in_pallet= item_in_box * box_in_pallet
  suppliers_capacity_min= item_in_pallet * minimum_pallet
  suppliers_capacity_max= item_in_pallet * maximum_pallet
  inventory_capactiy= inventory_capactiy_in_pallet * item_in_pallet


  #this code will pass a clean table to the loop by removing all the rows of the dataframes
  complete_inventory_results=inventory_data_result.iloc[0:0].copy()
  inventory_data_result = inventory_data_result.iloc[0:0].copy()



    #DATA SIMULATION
    #==============================================================================
    #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
    #noise recalculation should only be applied if we have have different coefficient of protocol, 
    #otherwise noise should stya the same, thats why it is outside of the loop
    
  provincial_data_CIHI['daily_consumption']=sim_demand.iloc[:,scenarios]
  provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()


  # >>>USER INPUT<<<
  # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
  # ======================================================================================================
  forecast_data_1 = provincial_data_CIHI.copy()

      # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
  consumption_date_type='daily_consumption'

  y = forecast_data_1[consumption_date_type]

  # this code will grouped the data based on the period info
  y_replenishement=replenishment_func_2(y,period_info)
  





  # PERIODIC REVIEW SYSTEM with NAIVE forecasting
  #====================================================================================================================

  inventory_data_periodic_naive = inventory_data.copy()
  inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)

  #number of periods that needs to be reviewed to find max value based on the specified period_info
  no_period_for_naive_max=math.ceil(2*R/period_info)

  for i in range(1,math.ceil(len(y)/R)+1):




        #number of days in epoch i:
    days_in_epoch_i=len(y[0:(i)*R])

    # number of periods in the available data based on the period_info    
    no_period_for_period_info = math.floor(days_in_epoch_i/period_info)

    # based on the period_info we want to take the maximum value in the past 2 weeks (2*R). therefore we take the maximum
    #value in the past n*period_info that covers 2*R, that is why "no_period_for_naive_max" is the math.ceil.
    y_to_train = y_replenishement[max(0,no_period_for_period_info-no_period_for_naive_max+1):no_period_for_period_info+1]
    y_to_val = y[(i)*R:(i)*R+(R+L)]
    y_set=y[0:(i)*R+(R+L)]

    prediction_naive = naive_forecast_method_replenished(y_to_train,y_to_val)

    y_previous_epoch=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_naive=peridoic_review(y_previous_epoch,
                                                  inventory_data_periodic_naive,
                                                  prediction_naive,
                                                  epoch_no)

  # PERIODIC REVIEW SYSTEM with HOLT forecasting
  #====================================================================================================================

  inventory_data_periodic_Holt = inventory_data
  inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

      #hospitalization_data_type='daily_consumption'

  try:
    holt_data=holt_data_type
  except NameError:
    holt_data='normal'


  for i in range(1,math.ceil(len(y)/R)+1):
    
    days_in_epoch_i=len(y[0:(i)*R])

    no_period_for_period_info = math.floor(days_in_epoch_i/period_info)
    y_to_train = y_replenishement[0:no_period_for_period_info+1]
    y_to_val = y[(i)*R:(i)*R+(R+L)]
    y_set=y[0:(i)*R+(R+L)]

    prediction_holt = holt_3(y_set, y_to_train,y_to_val,no_period_for_period_info)

    y_previous_epoch_holt=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_Holt=peridoic_review(y_previous_epoch_holt,
                                                  inventory_data_periodic_Holt,
                                                  prediction_holt,
                                                  epoch_no)

    # PERIODIC REVIEW SYSTEM with SIR forecasting model
    #=========================================================================================================

  inventory_data_periodic_SIR = inventory_data.copy()
  inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)


  for i in range(1,math.ceil(len(y)/R)+1):
    #y_to_train = y[0:(i)*R]
    #y_to_val = y[(i)*R:(i)*R+(R+L)]
    #y_set=y[0:(i)*R+(R+L)]
    #prediction = SIR_forecast_model(i,y_to_val)

    # number of days in epoch i:
    days_in_epoch_i=len(y[0:(i)*R])

    #number of days in the given data based on the period info:
    no_period_for_period_info = math.floor(days_in_epoch_i/period_info)
    days_in_period_info=no_period_for_period_info*period_info

    y_to_train = y[0:days_in_period_info]
    y_to_val = y[(i)*R:(i)*R+(R+L)]
    #y[days_in_period_info:days_in_period_info+(days_in_epoch_i-days_in_period_info)+(R+L)]
    prediction = SIR_forecast_model(no_period_for_period_info,y_to_val,epoch_number_2=i)
    
    y_previous_epoch=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction,epoch_no)

  # Putting everything into a dataframe
  #=========================================================================================================
  #putting in the main parameters of the scenarios in the data frames
  inventory_data_result_scenarios['Lead time']=L
  inventory_data_result_scenarios['SNR']=SNR
  inventory_data_result_scenarios['Protocol coefficient']=protocol_coef
  inventory_data_result_scenarios['Suppliers minimum capacity']=suppliers_capacity_min
  inventory_data_result_scenarios['Suppliers maximum capacity']=suppliers_capacity_max
  inventory_data_result_scenarios['Inventory capacity']=inventory_capactiy
  inventory_data_result_scenarios['Specified service level']=service_level
    # The total demand/consumption is the same for all methods, so we can pick any of them, in this case, SIR was picked
  inventory_data_result_scenarios['Total demand/consumption']=inventory_data_periodic_SIR['consumption in last epoch'].sum()

  # Puttin the analysis data into the data frame:
  #=========================================================================================================
  results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
  
  for i in range(len(results)):
    inventory_data_result_scenarios.iloc[[0],[(i*18)+7]]=results[i]['shortage'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+8]]=results[i]['current'].iloc[-1]
    inventory_data_result_scenarios.iloc[[0],[(i*18)+9]]=round((1-(results[i]['shortage'] != 0).sum()/len(results[i]['shortage']-1)),2)
    inventory_data_result_scenarios.iloc[[0],[(i*18)+10]]=(results[i]['shortage'] != 0).sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+11]]=results[i]['order'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+12]]=(results[i]['order'] != 0).sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+13]]=results[i]['real cost'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+14]]=results[i]['total cost'].sum()
    # the below command compute average effective percentage bias, by average we specify those epochs that orders can be made
    # if becuase of the lead time, we cannot place an order in a specific epoch (the end epochs) then the percentage bias
    # is not relevent to our study 
    inventory_data_result_scenarios.iloc[[0],[(i*18)+15]]=results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean()
    
    # Counting the effective number of epochs with under-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+16]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] < 0).sum()
    
    # Counting the effective number of epochs with over-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+17]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] > 0).sum()
    
    # Calulating the percentage of effective number of epochs with under-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+18]]=round((inventory_data_result_scenarios.iloc[0][(i*18)+17])*100/((inventory_data_result_scenarios.iloc[0][(i*18)+16])+(inventory_data_result_scenarios.iloc[0][(i*18)+17])),2)

    #Effective RMSE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+19]]=results[i]['std of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean() 
    #RMSE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+20]]=results[i]['std of previous epoch'][1:-1].mean()
    #Effective MAPE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+21]]=results[i]['MAPE'][1:math.floor((len(y)-L)/R)+1].mean() 
    #MAPE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+22]]=results[i]['MAPE'][1:-1].mean()
    #Effective MAE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+23]]=results[i]['MAE'][1:math.floor((len(y)-L)/R)+1].mean() 
    #MAE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+24]]=results[i]['MAE'][1:-1].mean()

  SIR_epochs_bias=((results[0].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('SIR percentage bias of epoch: ')
  Holt_epochs_bias=((results[1].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Holt percentage bias of epoch: ')
  Naive_epochs_bias=((results[2].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Naive percentage bias of epoch: ')
  
  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_bias,
                                             Holt_epochs_bias,
                                             Naive_epochs_bias],axis=1)

  SIR_epochs_shortage=(results[0].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('SIR shortage of epoch: ')
  Holt_epochs_shortage=(results[1].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Holt shortage of epoch: ')
  Naive_epochs_shortage=(results[2].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Naive shortage of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_shortage,
                                             Holt_epochs_shortage,
                                             Naive_epochs_shortage],axis=1)

  SIR_epochs_leftover_in=(results[0].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('SIR left-over inventory of epoch: ')
  Holt_epochs_leftover_in=(results[1].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Holt left-over inventory of epoch: ')
  Naive_epochs_leftover_in=(results[2].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Naive left-over inventory of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_leftover_in,
                                             Holt_epochs_leftover_in,
                                             Naive_epochs_leftover_in],axis=1)
  SIR_order=(results[0].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('SIR order of epoch: ')
  Holt_order=(results[1].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Holt order of epoch: ')
  Naive_order=(results[2].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Naive order of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_order,
                                              Holt_order,
                                              Naive_order],axis=1)
  
  SIR_order_actual=(results[0].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('SIR order actual of epoch: ')
  Holt_order_actual=(results[1].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Holt order actual of epoch: ')
  Naive_order_actual=(results[2].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Naive order actual of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_order_actual,
                                              Holt_order_actual,
                                              Naive_order_actual],axis=1)

  #inventory_data_result_scenarios  
  complete_inventory_results_scenarios_replenished=complete_inventory_results_scenarios_replenished.append([inventory_data_result_scenarios],ignore_index=True)

#Exporting the data into the a csv file which will be saved into the drive
complete_inventory_results_scenarios_replenished.to_csv('Inventory management results with %s scenarios(replenished data) period info %s.csv'%(no_scenarios,period_info))

complete_inventory_results_scenarios_replenished

"""#### Results

##### Importing data
"""

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
complete_inventory_results_scenarios_replenished_6=pd.read_csv('Inventory management results with 10000 scenarios(replenished data) period_info 6.csv')
#setting the date as the index of the dataframe
complete_inventory_results_scenarios_replenished_6=complete_inventory_results_scenarios_replenished_6.drop(columns='Unnamed: 0')

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
complete_inventory_results_scenarios_replenished_7=pd.read_csv('Inventory management results with 10000 scenarios(replenished data) period_info 7.csv')
#setting the date as the index of the dataframe
complete_inventory_results_scenarios_replenished_7=complete_inventory_results_scenarios_replenished_7.drop(columns='Unnamed: 0')

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
complete_inventory_results_scenarios_replenished_8=pd.read_csv('Inventory management results with 10000 scenarios(replenished data) period_info 8.csv')
#setting the date as the index of the dataframe
complete_inventory_results_scenarios_replenished_8=complete_inventory_results_scenarios_replenished_8.drop(columns='Unnamed: 0')

"""##### Presentation of results

###### Data filtering
"""

#creating a separet daraframe in order to not touch the original copy
df_temp_present_replenished_6=complete_inventory_results_scenarios_replenished_6.copy()

# SHORTAGE
#============================================
# filtering the data for sepcified category
shortage_epoch_df_SIR_replenished_6=df_temp_present_replenished_6[df_temp_present_replenished_6.columns[pd.Series(df_temp_present_replenished_6.columns).str.startswith('SIR shortage of epoch')]]
shortage_epoch_df_Holt_replenished_6=df_temp_present_replenished_6[df_temp_present_replenished_6.columns[pd.Series(df_temp_present_replenished_6.columns).str.startswith('Holt shortage of epoch')]]
shortage_epoch_df_Naive_replenished_6=df_temp_present_replenished_6[df_temp_present_replenished_6.columns[pd.Series(df_temp_present_replenished_6.columns).str.startswith('Naive shortage of epoch')]]


# Left-over inventory
#============================================
# filtering the data for sepcified category
leftover_epoch_df_SIR_replenished_6=df_temp_present_replenished_6[df_temp_present_replenished_6.columns[pd.Series(df_temp_present_replenished_6.columns).str.startswith('SIR left-over inventory of epoch')]]
leftover_epoch_df_Holt_replenished_6=df_temp_present_replenished_6[df_temp_present_replenished_6.columns[pd.Series(df_temp_present_replenished_6.columns).str.startswith('Holt left-over inventory of epoch')]]
leftover_epoch_df_Naive_replenished_6=df_temp_present_replenished_6[df_temp_present_replenished_6.columns[pd.Series(df_temp_present_replenished_6.columns).str.startswith('Naive left-over inventory of epoch')]]

# Bais-over inventory
#============================================
# filtering the data for sepcified category
bias_epoch_df_SIR_replenished_6=df_temp_present_replenished_6[df_temp_present_replenished_6.columns[pd.Series(df_temp_present_replenished_6.columns).str.startswith('SIR percentage bias of epoch')]]
bias_epoch_df_Holt_replenished_6=df_temp_present_replenished_6[df_temp_present_replenished_6.columns[pd.Series(df_temp_present_replenished_6.columns).str.startswith('Holt percentage bias of epoch')]]
bias_epoch_df_Naive_replenished_6=df_temp_present_replenished_6[df_temp_present_replenished_6.columns[pd.Series(df_temp_present_replenished_6.columns).str.startswith('Naive percentage bias of epoch')]]

#creating a separet daraframe in order to not touch the original copy
df_temp_present_replenished_7=complete_inventory_results_scenarios_replenished_7.copy()

# SHORTAGE
#============================================
# filtering the data for sepcified category
shortage_epoch_df_SIR_replenished_7=df_temp_present_replenished_7[df_temp_present_replenished_7.columns[pd.Series(df_temp_present_replenished_7.columns).str.startswith('SIR shortage of epoch')]]
shortage_epoch_df_Holt_replenished_7=df_temp_present_replenished_7[df_temp_present_replenished_7.columns[pd.Series(df_temp_present_replenished_7.columns).str.startswith('Holt shortage of epoch')]]
shortage_epoch_df_Naive_replenished_7=df_temp_present_replenished_7[df_temp_present_replenished_7.columns[pd.Series(df_temp_present_replenished_7.columns).str.startswith('Naive shortage of epoch')]]


# Left-over inventory
#============================================
# filtering the data for sepcified category
leftover_epoch_df_SIR_replenished_7=df_temp_present_replenished_7[df_temp_present_replenished_7.columns[pd.Series(df_temp_present_replenished_7.columns).str.startswith('SIR left-over inventory of epoch')]]
leftover_epoch_df_Holt_replenished_7=df_temp_present_replenished_7[df_temp_present_replenished_7.columns[pd.Series(df_temp_present_replenished_7.columns).str.startswith('Holt left-over inventory of epoch')]]
leftover_epoch_df_Naive_replenished_7=df_temp_present_replenished_7[df_temp_present_replenished_7.columns[pd.Series(df_temp_present_replenished_7.columns).str.startswith('Naive left-over inventory of epoch')]]

# Bais-over inventory
#============================================
# filtering the data for sepcified category
bias_epoch_df_SIR_replenished_7=df_temp_present_replenished_7[df_temp_present_replenished_7.columns[pd.Series(df_temp_present_replenished_7.columns).str.startswith('SIR percentage bias of epoch')]]
bias_epoch_df_Holt_replenished_7=df_temp_present_replenished_7[df_temp_present_replenished_7.columns[pd.Series(df_temp_present_replenished_7.columns).str.startswith('Holt percentage bias of epoch')]]
bias_epoch_df_Naive_replenished_7=df_temp_present_replenished_7[df_temp_present_replenished_7.columns[pd.Series(df_temp_present_replenished_7.columns).str.startswith('Naive percentage bias of epoch')]]

#creating a separet daraframe in order to not touch the original copy
df_temp_present_replenished_8=complete_inventory_results_scenarios_replenished_8.copy()

# SHORTAGE
#============================================
# filtering the data for sepcified category
shortage_epoch_df_SIR_replenished_8=df_temp_present_replenished_8[df_temp_present_replenished_8.columns[pd.Series(df_temp_present_replenished_8.columns).str.startswith('SIR shortage of epoch')]]
shortage_epoch_df_Holt_replenished_8=df_temp_present_replenished_8[df_temp_present_replenished_8.columns[pd.Series(df_temp_present_replenished_8.columns).str.startswith('Holt shortage of epoch')]]
shortage_epoch_df_Naive_replenished_8=df_temp_present_replenished_8[df_temp_present_replenished_8.columns[pd.Series(df_temp_present_replenished_8.columns).str.startswith('Naive shortage of epoch')]]


# Left-over inventory
#============================================
# filtering the data for sepcified category
leftover_epoch_df_SIR_replenished_8=df_temp_present_replenished_8[df_temp_present_replenished_8.columns[pd.Series(df_temp_present_replenished_8.columns).str.startswith('SIR left-over inventory of epoch')]]
leftover_epoch_df_Holt_replenished_8=df_temp_present_replenished_8[df_temp_present_replenished_8.columns[pd.Series(df_temp_present_replenished_8.columns).str.startswith('Holt left-over inventory of epoch')]]
leftover_epoch_df_Naive_replenished_8=df_temp_present_replenished_8[df_temp_present_replenished_8.columns[pd.Series(df_temp_present_replenished_8.columns).str.startswith('Naive left-over inventory of epoch')]]

# Bais-over inventory
#============================================
# filtering the data for sepcified category
bias_epoch_df_SIR_replenished_8=df_temp_present_replenished_8[df_temp_present_replenished_8.columns[pd.Series(df_temp_present_replenished_8.columns).str.startswith('SIR percentage bias of epoch')]]
bias_epoch_df_Holt_replenished_8=df_temp_present_replenished_8[df_temp_present_replenished_8.columns[pd.Series(df_temp_present_replenished_8.columns).str.startswith('Holt percentage bias of epoch')]]
bias_epoch_df_Naive_replenished_8=df_temp_present_replenished_8[df_temp_present_replenished_8.columns[pd.Series(df_temp_present_replenished_8.columns).str.startswith('Naive percentage bias of epoch')]]

"""###### comparison graph"""

scenario_II_graph_comparison()

"""###### Comparison function graph"""

def scenario_II_graph_comparison ():
  #protocol_coef_variation=[3,3.5,4,4.5,5]
  #service_level_variation


  plt.figure(figsize=(23,10))

  plt.subplot(1, 2, 1)
  #this part plots different Protocol coefficient and their related shortages
  #plt.figure(figsize=(17,8.5))

  data1=shortage_epoch_df_SIR_replenished
  data2=shortage_epoch_df_Holt_replenished
  data3=shortage_epoch_df_Naive_replenished
  y_axis_name='Shortage'
  #taking the mean of each column and removing the epochs that order cannot be placed (based on the lead time)
  y_bias_SIR=data1.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Holt=data2.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Naive=data3.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  x=range(len(y_bias_SIR))

  #spine placement data centered
  ax.spines['left'].set_position(('data', 0.0))
  ax.spines['bottom'].set_position(('data', 0.0))
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')

  plt.plot(x,y_bias_SIR, color = 'blue',label='SEIRHD',marker='^')
  plt.plot(x,y_bias_Holt, color = 'green',label='Holt',marker='^')
  plt.plot(x,y_bias_Naive, color = 'red',label='Naive',marker='^')



  plt.legend(['SEIRHD','Holt','Naive'])

  plt.xlabel('Epoch',fontsize=16)
  plt.ylabel(y_axis_name,fontsize=16)




  plt.subplot(1, 2, 2)
  #10,000


  data1=leftover_epoch_df_SIR_replenished
  data2=leftover_epoch_df_Holt_replenished
  data3=leftover_epoch_df_Naive_replenished
  y_axis_name='inventory level'
  #taking the mean of each column and removing the epochs that order cannot be placed (based on the lead time)
  y_bias_SIR=data1.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Holt=data2.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  y_bias_Naive=data3.mean(axis=0)[:math.floor((len(y)-upper_limit_L)/R)+1]
  x=range(len(y_bias_SIR))

  #spine placement data centered
  ax.spines['left'].set_position(('data', 0.0))
  ax.spines['bottom'].set_position(('data', 0.0))
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')

  plt.plot(x,y_bias_SIR, color = 'blue',label='SEIRHD',marker='^')
  plt.plot(x,y_bias_Holt, color = 'green',label='Holt',marker='^')
  plt.plot(x,y_bias_Naive, color = 'red',label='Naive',marker='^')



  plt.legend(['SEIRHD','Holt','Naive'])

  plt.xlabel('Epoch',fontsize=16)
  plt.ylabel(y_axis_name,fontsize=16)

  return

df_comparison=inventory_data_periodic_Holt['consumption in last epoch']
type(df_comparison)
df_comparison=df_comparison.to_frame()
df_comparison['SIR current']=inventory_data_periodic_SIR['current']
df_comparison['SIR order']=inventory_data_periodic_SIR['order']
df_comparison['SIR shortage']=inventory_data_periodic_SIR['shortage']

df_comparison['naive current']=inventory_data_periodic_naive['current']
df_comparison['naive order']=inventory_data_periodic_naive['order']
df_comparison['naive shortage']=inventory_data_periodic_naive['shortage']


df_comparison['Holt current']=inventory_data_periodic_Holt['current']
df_comparison['Holt order']=inventory_data_periodic_Holt['order']
df_comparison['Holt shortage']=inventory_data_periodic_Holt['shortage']



df_comparison

df_maker(leftover_epoch_df_SIR_replenished,leftover_epoch_df_Holt_replenished,leftover_epoch_df_Naive_replenished)

df_maker(data_1=shortage_epoch_df_SIR_replenished,data_2=shortage_epoch_df_Holt_replenished,data_3=shortage_epoch_df_Naive_replenished)

"""###### Relative data"""

### SHORTAGE

#10,000

shortage_epoch_df_SIR_relative_to_demand_replenished_6=100*shortage_epoch_df_SIR_replenished_6.div(grouped_demand.values)
shortage_epoch_df_Holt_relative_to_demand_replenished_6=100*shortage_epoch_df_Holt_replenished_6.div(grouped_demand.values)
shortage_epoch_df_Naive_relative_to_demand_replenished_6=100*shortage_epoch_df_Naive_replenished_6.div(grouped_demand.values)

shortage_epoch_df_SIR_relative_to_demand_replenished_7=100*shortage_epoch_df_SIR_replenished_7.div(grouped_demand.values)
shortage_epoch_df_Holt_relative_to_demand_replenished_7=100*shortage_epoch_df_Holt_replenished_7.div(grouped_demand.values)
shortage_epoch_df_Naive_relative_to_demand_replenished_7=100*shortage_epoch_df_Naive_replenished_7.div(grouped_demand.values)

shortage_epoch_df_SIR_relative_to_demand_replenished_8=100*shortage_epoch_df_SIR_replenished_8.div(grouped_demand.values)
shortage_epoch_df_Holt_relative_to_demand_replenished_8=100*shortage_epoch_df_Holt_replenished_8.div(grouped_demand.values)
shortage_epoch_df_Naive_relative_to_demand_replenished_8=100*shortage_epoch_df_Naive_replenished_8.div(grouped_demand.values)

### LEFT-OVER INVENTORY

grouped_demand

leftover_epoch_df_SIR_relative_to_demand_replenished_6=100*leftover_epoch_df_SIR_replenished_6.div(grouped_demand.values)
leftover_epoch_df_Holt_relative_to_demand_replenished_6=100*leftover_epoch_df_Holt_replenished_6.div(grouped_demand.values)
leftover_epoch_df_Naive_relative_to_demand_replenished_6=100*leftover_epoch_df_Naive_replenished_6.div(grouped_demand.values)

leftover_epoch_df_SIR_relative_to_demand_replenished_6.iloc[:,0]=0
leftover_epoch_df_Holt_relative_to_demand_replenished_6.iloc[:,0]=0
leftover_epoch_df_Naive_relative_to_demand_replenished_6.iloc[:,0]=0

leftover_epoch_df_SIR_relative_to_demand_replenished_7=100*leftover_epoch_df_SIR_replenished_7.div(grouped_demand.values)
leftover_epoch_df_Holt_relative_to_demand_replenished_7=100*leftover_epoch_df_Holt_replenished_7.div(grouped_demand.values)
leftover_epoch_df_Naive_relative_to_demand_replenished_7=100*leftover_epoch_df_Naive_replenished_7.div(grouped_demand.values)

leftover_epoch_df_SIR_relative_to_demand_replenished_7.iloc[:,0]=0
leftover_epoch_df_Holt_relative_to_demand_replenished_7.iloc[:,0]=0
leftover_epoch_df_Naive_relative_to_demand_replenished_7.iloc[:,0]=0

leftover_epoch_df_SIR_relative_to_demand_replenished_8=100*leftover_epoch_df_SIR_replenished_8.div(grouped_demand.values)
leftover_epoch_df_Holt_relative_to_demand_replenished_8=100*leftover_epoch_df_Holt_replenished_8.div(grouped_demand.values)
leftover_epoch_df_Naive_relative_to_demand_replenished_8=100*leftover_epoch_df_Naive_replenished_8.div(grouped_demand.values)

leftover_epoch_df_SIR_relative_to_demand_replenished_8.iloc[:,0]=0
leftover_epoch_df_Holt_relative_to_demand_replenished_8.iloc[:,0]=0
leftover_epoch_df_Naive_relative_to_demand_replenished_8.iloc[:,0]=0

"""###### Shortage vs epochs"""

### PERIOD INFO ==> 6

graph_generator_for_epochs_2(shortage_epoch_df_SIR_relative_to_demand_replenished_6,
                           shortage_epoch_df_Holt_relative_to_demand_replenished_6,
                           shortage_epoch_df_Naive_relative_to_demand_replenished_6,'Shortage','Normal','Replenished: Relative data with Period info 6')

#new 10,000
graph_generator_for_epochs_2(shortage_epoch_df_SIR_replenished_6,
                           shortage_epoch_df_Holt_replenished_6,
                           shortage_epoch_df_Naive_replenished_6,'Shortage','Granularity','Replenished: Raw data with Period info 6')

### PERIOD INFO ==>7

graph_generator_for_epochs_2(shortage_epoch_df_SIR_relative_to_demand_replenished_7,
                           shortage_epoch_df_Holt_relative_to_demand_replenished_7,
                           shortage_epoch_df_Naive_relative_to_demand_replenished_7,'Shortage','Normal','Replenished: Relative data with Period info 7')

#new 10,000
graph_generator_for_epochs_2(shortage_epoch_df_SIR_replenished_7,
                           shortage_epoch_df_Holt_replenished_7,
                           shortage_epoch_df_Naive_replenished_7,'Shortage','Granularity','Replenished: Raw data with Period info 7')

### PERIOD INFO ==> 8

graph_generator_for_epochs_2(shortage_epoch_df_SIR_relative_to_demand_replenished_8,
                           shortage_epoch_df_Holt_relative_to_demand_replenished_8,
                           shortage_epoch_df_Naive_relative_to_demand_replenished_8,'Shortage','Normal','Replenished: Relative data with Period info 8')

#new 10,000
graph_generator_for_epochs_2(shortage_epoch_df_SIR_replenished_8,
                           shortage_epoch_df_Holt_replenished_8,
                           shortage_epoch_df_Naive_replenished_8,'Shortage','Granularity','Replenished: Raw data with Period info 8')

"""###### Left-over inventory vs epoch"""

# PERIOD INFO ==> 6

graph_generator_for_epochs_2(leftover_epoch_df_SIR_relative_to_demand_replenished_6,
                           leftover_epoch_df_Holt_relative_to_demand_replenished_6,
                           leftover_epoch_df_Naive_relative_to_demand_replenished_6,'leftover','Normal','Replenished: Relative data with Period info 6')

#100
graph_generator_for_epochs_2(leftover_epoch_df_SIR_replenished_6,
                           leftover_epoch_df_Holt_replenished_6,
                           leftover_epoch_df_Naive_replenished_6,'left-over inventory','Granularity','Replenished: Raw data with Period info 6')

# PERIOD INFO ==> 7

graph_generator_for_epochs_2(leftover_epoch_df_SIR_relative_to_demand_replenished_7,
                           leftover_epoch_df_Holt_relative_to_demand_replenished_7,
                           leftover_epoch_df_Naive_relative_to_demand_replenished_7,'leftover','Normal','Replenished: Relative data with Period info 7')

#100
graph_generator_for_epochs_2(leftover_epoch_df_SIR_replenished_7,
                           leftover_epoch_df_Holt_replenished_7,
                           leftover_epoch_df_Naive_replenished_7,'left-over inventory','Granularity','Replenished: Raw data with Period info 7')

# PERIOD INFO ==> 8

graph_generator_for_epochs_2(leftover_epoch_df_SIR_relative_to_demand_replenished_8,
                           leftover_epoch_df_Holt_relative_to_demand_replenished_8,
                           leftover_epoch_df_Naive_relative_to_demand_replenished_8,'leftover','Normal','Replenished: Relative data with Period info 8')

#100
graph_generator_for_epochs_2(leftover_epoch_df_SIR_replenished_8,
                           leftover_epoch_df_Holt_replenished_8,
                           leftover_epoch_df_Naive_replenished_8,'left-over inventory','Granularity','Replenished: Raw data with Period info 8')

"""###### Comparison of different period infos"""

plt.subplot(nrows=1, ncols=2, figsize=(5, 3))

graph_generator_for_epochs_2(shortage_epoch_df_SIR_relative_to_demand_replenished_6,
                           shortage_epoch_df_Holt_relative_to_demand_replenished_6,
                           shortage_epoch_df_Naive_relative_to_demand_replenished_6,'Shortage','Normal','Replenished: Relative data with Period info 6')

#new 10,000
graph_generator_for_epochs_2(shortage_epoch_df_SIR_replenished_6,
                           shortage_epoch_df_Holt_replenished_6,
                           shortage_epoch_df_Naive_replenished_6,'Shortage','Granularity','Replenished: Raw data with Period info 6')

plt.tight_layout()

"""###### Percentage bias vs epoch"""

#10,000
graph_generator_for_epochs(bias_epoch_df_SIR_replenished,
                           bias_epoch_df_Holt_replenished,
                           bias_epoch_df_Naive_replenished,'Percentage bias')

#100
graph_generator_for_epochs(bias_epoch_df_SIR_replenished,
                           bias_epoch_df_Holt_replenished,
                           bias_epoch_df_Naive_replenished,'Percentage bias')

"""###### Shortage vs. final inventory"""

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)

x_plot=df_temp_present_replenished['SIR forecast: Shortage']
y_plot=df_temp_present_replenished['SIR forecast: Inventory level at the end']
plt.scatter(x_plot, y_plot, color = 'red',label='SIR forecast',marker='o')

x_plot=df_temp_present_replenished['Holt forecast: Shortage']
y_plot=df_temp_present_replenished['Holt forecast: Inventory level at the end']
plt.scatter(x_plot, y_plot, color = 'blue',label='Holt forecast',marker='x')

x_plot=df_temp_present_replenished['Naive forecast: Shortage']
y_plot=df_temp_present_replenished['Naive forecast: Inventory level at the end']
plt.scatter(x_plot, y_plot, color = 'green',label='Naive forecast',marker='^')

plt.title('%s scnearios of inventory management in British Columbia'%(no_scenarios),fontsize=16)
plt.xlabel('Shortage',fontsize=16)
plt.ylabel('Final inventory',fontsize=16)
plt.legend(fontsize=13)

plt.show()

"""## Scenario III: Simulation process "Lagged"

#### Parameters
"""

# In this section the parameters for ranges of scenarios are defined
#==============================================================================

# How many scenarios do you want?
no_scenarios=1

# SUPPLIER'S CONSTRAINTS:
#====================================
# How many items are in a box:
item_in_box=12
# How many items are in a pallet:
box_in_pallet=1

item_in_pallet= item_in_box * box_in_pallet

# Minimum order
# what is the lower limit for minimum pallet order:
lower_limit_minimum_pallet=1 
# what is the upper limit for minimum pallet order:
upper_limit_minimum_pallet=10 

# Maximum order
# what is the lower limit for maximum pallet order:
lower_limit_maximum_pallet=200 
# what is the upper limit for maximum pallet order:
upper_limit_maximum_pallet=400

# Lead time
# what is the lower limit for lead time of the order:
lower_limit_L=5
# what is the upper limit for lead time of the order:
upper_limit_L=30

# CONSUMPTION'S CONSTRAINTS:
#====================================
# What is the lower limit of coefficient of consumption:
lower_limit_CC=3
# What is the upper limit of coefficient of consumption:
upper_limit_CC=7


# INVENTORY MANAGEMENT CONSTRAINTS:
#====================================
# Capacity of inventory
# What is the lower limit of invernoty capacity (in terms of number of pallet):
lower_limit_invernoty_capacity= 3000
# What is the upper limit of invernoty capacity (in terms of number of pallet):
upper_limit_invernoty_capacity= 6000

# Service level
# What is the lower limit of serivce level (in percent (%)):
lower_limit_service_level= 95
# What is the upper limit of serivce level (in percent (%))
upper_limit_service_level= 99.9


#  DATA SIMULATION CONSTRAINTS:
#====================================

#Do you want to have noise added to the data?
#     'yes'     if you want to add noise to the data 
#     'no'      if you do NOT want to add noise
noise_addition='yes'


# Noise generation 
# What is the lower limit of SNR:
lower_limit_SNR= 2
# What is the upper limit of SNR:
upper_limit_SNR= 10

# Deviation from CC
#SIR_forecast_CC_dev=[0]

#  Lag info 
#====================================

# how many epochs are there in the 'lag'?
lag=2

# daily lag limit 
# What is the lower limit of daily lag:
lower_limit_lag= 0
# What is the upper limit of daily lag:
upper_limit_lag= 14

#  Replenishment info 
#====================================

# how many days should be between the arrival of info (replenishement mode)
period_info=7

"""#### Scenario creator

##### version I (epoch):
"""

# Choose the type of data to be used in the SIR forecasting method:
# Normal
# Replenished
# ==============================================================================================
# This is the "NORMAL" data type


#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
SIR_forecast_df=pd.read_csv('forecast SIR without cumulative-complete 18 epochs.csv')
#setting the date as the index of the dataframe
SIR_forecast_df=SIR_forecast_df.set_index(SIR_forecast_df['Date']).drop(columns='Date')
#rounding up the value of forecast in the data frame 
#SIR_forecast_df=np.ceil(SIR_forecast_df)

lag=2

#this code will pass a clean table to the loop by removing all the rows of the dataframes

inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()
complete_inventory_results_scenarios_lagged=inventory_data_result_scenarios.iloc[0:0].copy()
#inventory_data_result_scenarios = inventory_data_result_scenarios.iloc[0:0].copy()



for scenarios in range(no_scenarios):
  # Using uniform distribution to randomly choose a value for the following parameters
  inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()

  minimum_pallet = round(np.random.uniform(lower_limit_minimum_pallet, upper_limit_minimum_pallet))
  maximum_pallet = round(np.random.uniform(lower_limit_maximum_pallet, upper_limit_maximum_pallet))
  protocol_coef = (round((np.random.uniform(lower_limit_CC, upper_limit_CC))*2))/2
  L = round(np.random.uniform(lower_limit_L, upper_limit_L))
  inventory_capactiy_in_pallet = round(np.random.uniform(lower_limit_invernoty_capacity, upper_limit_invernoty_capacity))
  service_level = round(np.random.uniform(lower_limit_service_level, upper_limit_service_level),1)/100
  SNR = round(np.random.uniform(lower_limit_SNR, upper_limit_SNR))

  # Calculating the following varibales
  item_in_pallet= item_in_box * box_in_pallet
  suppliers_capacity_min= item_in_pallet * minimum_pallet
  suppliers_capacity_max= item_in_pallet * maximum_pallet
  inventory_capactiy= inventory_capactiy_in_pallet * item_in_pallet


  #this code will pass a clean table to the loop by removing all the rows of the dataframes
  complete_inventory_results=inventory_data_result.iloc[0:0].copy()
  inventory_data_result = inventory_data_result.iloc[0:0].copy()



    #DATA SIMULATION
    #==============================================================================
    #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
    #noise recalculation should only be applied if we have have different coefficient of protocol, 
    #otherwise noise should stya the same, thats why it is outside of the loop
    
  provincial_data_CIHI['daily_consumption']=noise_creator_with_SNR(provincial_data_CIHI['daily_Hospitalizations'],
                                                                    protocol_coef)
  provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()


      # >>>USER INPUT<<<
      # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
      # ======================================================================================================
  forecast_data_1 = provincial_data_CIHI.copy()

      # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
  consumption_date_type='daily_consumption'

  y = forecast_data_1[consumption_date_type]
  y_cumulative = forecast_data_1['cumulative_daily_consumption']

      # PERIODIC REVIEW SYSTEM with NAIVE forecasting
      #====================================================================================================================

  inventory_data_periodic_naive = inventory_data.copy()
  inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)


  for i in range(1,math.ceil(len(y)/R)+1):

    y_to_val = y[(i)*R:(i)*R+(R+L)]
        #y_set=y[0:(i+1)*R+(R+L)]
    epoch_no=i
    prediction = naive_forecast_method_lagged(y,y_to_val,epoch_no,lag)
    y_previous_epoch=y[(i-1)*R:(i)*R]
    inventory_data_periodic_naive=peridoic_review(y_previous_epoch,inventory_data_periodic_naive,prediction,epoch_no)

      # PERIODIC REVIEW SYSTEM with HOLT forecasting
      #====================================================================================================================

  inventory_data_periodic_Holt = inventory_data
  inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

      #hospitalization_data_type='daily_consumption'

  try:
    hospitalization_data_type=hospitalization_data_type
  except NameError:
    hospitalization_data_type='daily_consumption'

  if hospitalization_data_type=='daily_consumption':
    y_holt=y
  else:
    y_holt=y_cumulative


  for i in range(1,math.ceil(len(y)/R)+1):
    
    y_to_val = y_holt[(i)*R:(i)*R+(R+L)]
    y_set=y_holt[0:(i)*R+(R+L)]
    y_previous_epoch_holt=y_holt[(i-1)*R:(i)*R]
    prediction = holt_lagged(y_set,y_to_val,i,y_previous_epoch_holt,lag)
    y_previous_epoch=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_Holt=peridoic_review(y_previous_epoch,inventory_data_periodic_Holt,prediction,epoch_no)

    # PERIODIC REVIEW SYSTEM with SIR forecasting model
    #=========================================================================================================

  inventory_data_periodic_SIR = inventory_data.copy()
  inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)


  for i in range(1,math.ceil(len(y)/R)+1):
    y_to_train = y[0:(i)*R]
    y_to_val = y[(i)*R:(i)*R+(R+L)]
    y_set=y[0:(i)*R+(R+L)]
    prediction = SIR_forecast_model_lagged(i,y_to_val,lag)
    y_previous_epoch=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction,epoch_no)

  # Putting everything into a dataframe
  #=========================================================================================================
  #putting in the main parameters of the scenarios in the data frames
  inventory_data_result_scenarios['Lead time']=L
  inventory_data_result_scenarios['SNR']=SNR
  inventory_data_result_scenarios['Protocol coefficient']=protocol_coef
  inventory_data_result_scenarios['Suppliers minimum capacity']=suppliers_capacity_min
  inventory_data_result_scenarios['Suppliers maximum capacity']=suppliers_capacity_max
  inventory_data_result_scenarios['Inventory capacity']=inventory_capactiy
  inventory_data_result_scenarios['Specified service level']=service_level
    # The total demand/consumption is the same for all methods, so we can pick any of them, in this case, SIR was picked
  inventory_data_result_scenarios['Total demand/consumption']=inventory_data_periodic_SIR['consumption in last epoch'].sum()

  # Puttin the analysis data into the data frame:
  #=========================================================================================================
  results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
  
  for i in range(len(results)):
    inventory_data_result_scenarios.iloc[[0],[(i*18)+7]]=results[i]['shortage'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+8]]=results[i]['current'].iloc[-1],
    inventory_data_result_scenarios.iloc[[0],[(i*18)+9]]=round((1-(results[i]['shortage'] != 0).sum()/len(results[i]['shortage']-1)),2)
    inventory_data_result_scenarios.iloc[[0],[(i*18)+10]]=(results[i]['shortage'] != 0).sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+11]]=results[i]['order'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+12]]=(results[i]['order'] != 0).sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+13]]=results[i]['real cost'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+14]]=results[i]['total cost'].sum()
    # the below command compute average effective percentage bias, by average we specify those epochs that orders can be made
    # if becuase of the lead time, we cannot place an order in a specific epoch (the end epochs) then the percentage bias
    # is not relevent to our study 
    inventory_data_result_scenarios.iloc[[0],[(i*18)+15]]=results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean()
    
    # Counting the effective number of epochs with under-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+16]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] < 0).sum()
    
    # Counting the effective number of epochs with over-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+17]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] > 0).sum()
    
    # Calulating the percentage of effective number of epochs with under-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+18]]=round((inventory_data_result_scenarios.iloc[0][(i*12)+17])*100/((inventory_data_result_scenarios.iloc[0][(i*12)+16])+(inventory_data_result_scenarios.iloc[0][(i*12)+17])),2)

    #Effective RMSE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+19]]=results[i]['std of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean() 
    #RMSE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+20]]=results[i]['std of previous epoch'][1:-1].mean()
    #Effective MAPE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+21]]=results[i]['MAPE'][1:math.floor((len(y)-L)/R)+1].mean() 
    #MAPE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+22]]=results[i]['MAPE'][1:-1].mean()
    #Effective MAE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+23]]=results[i]['MAE'][1:math.floor((len(y)-L)/R)+1].mean() 
    #MAE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+24]]=results[i]['MAE'][1:-1].mean()

  SIR_epochs_bias=((results[0].T.tail(1)).reset_index()).iloc[:,1:-1].add_prefix('SIR percentage bias of epoch: ')
  Holt_epochs_bias=((results[1].T.tail(1)).reset_index()).iloc[:,1:-1].add_prefix('Holt percentage bias of epoch: ')
  Naive_epochs_bias=((results[2].T.tail(1)).reset_index()).iloc[:,1:-1].add_prefix('Naive percentage bias of epoch: ')
  
  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_bias,
                                             Holt_epochs_bias,
                                             Naive_epochs_bias],axis=1)

  SIR_epochs_shortage=(results[0].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('SIR shortage of epoch: ')
  Holt_epochs_shortage=(results[1].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Holt shortage of epoch: ')
  Naive_epochs_shortage=(results[2].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Naive shortage of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_shortage,
                                             Holt_epochs_shortage,
                                             Naive_epochs_shortage],axis=1)

  SIR_epochs_leftover_in=(results[0].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('SIR left-over inventory of epoch: ')
  Holt_epochs_leftover_in=(results[1].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Holt left-over inventory of epoch: ')
  Naive_epochs_leftover_in=(results[2].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Naive left-over inventory of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_leftover_in,
                                             Holt_epochs_leftover_in,
                                             Naive_epochs_leftover_in],axis=1)

  #inventory_data_result_scenarios  
  complete_inventory_results_scenarios_lagged=complete_inventory_results_scenarios_lagged.append([inventory_data_result_scenarios],ignore_index=True)

#Exporting the data into the a csv file which will be saved into the drive
complete_inventory_results_scenarios_lagged.to_csv('Inventory management results with %s scenarios(lagged data) .csv'%(no_scenarios))

"""##### version II (day) fixed lag:"""

# Choose the type of data to be used in the SIR forecasting method:
# Normal
# Replenished
# ==============================================================================================
# This is the "NORMAL" data type


#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
SIR_forecast_df=pd.read_csv('forecast SIR without cumulative-complete 122 days.csv')
#setting the date as the index of the dataframe
SIR_forecast_df=SIR_forecast_df.set_index(SIR_forecast_df['Date']).drop(columns='Date')
#rounding up the value of forecast in the data frame 
#SIR_forecast_df=np.ceil(SIR_forecast_df)

#lag=14
no_scenarios=1

#this code will pass a clean table to the loop by removing all the rows of the dataframes

inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()
complete_inventory_results_scenarios_lagged_daily=inventory_data_result_scenarios.iloc[0:0].copy()
#inventory_data_result_scenarios = inventory_data_result_scenarios.iloc[0:0].copy()



for scenarios in range(no_scenarios):
  # Using uniform distribution to randomly choose a value for the following parameters
  inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()

  minimum_pallet = complete_sim_scenarios['Min pallet'][scenarios]
  maximum_pallet = complete_sim_scenarios['Max pallet'][scenarios]
  protocol_coef = complete_sim_scenarios['Protocol coefficient'][scenarios]
  L = complete_sim_scenarios['Lead time'][scenarios]
  inventory_capactiy_in_pallet = complete_sim_scenarios['Inventory capacity (in pallet)'][scenarios]
  service_level = complete_sim_scenarios['Specified service level'][scenarios]
  SNR = complete_sim_scenarios['SNR'][scenarios]
  lag = complete_sim_scenarios['Lag'][scenarios]

  # Calculating the following varibales
  item_in_pallet= item_in_box * box_in_pallet
  suppliers_capacity_min= item_in_pallet * minimum_pallet
  suppliers_capacity_max= item_in_pallet * maximum_pallet
  inventory_capactiy= inventory_capactiy_in_pallet * item_in_pallet


  #this code will pass a clean table to the loop by removing all the rows of the dataframes
  complete_inventory_results=inventory_data_result.iloc[0:0].copy()
  inventory_data_result = inventory_data_result.iloc[0:0].copy()



    #DATA SIMULATION
    #==============================================================================
    #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
    #noise recalculation should only be applied if we have have different coefficient of protocol, 
    #otherwise noise should stya the same, thats why it is outside of the loop
    
  provincial_data_CIHI['daily_consumption']=sim_demand.iloc[:,scenarios]
  
  provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()


      # >>>USER INPUT<<<
      # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
      # ======================================================================================================
  forecast_data_1 = provincial_data_CIHI.copy()

      # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
  consumption_date_type='daily_consumption'

  y = forecast_data_1[consumption_date_type]
  y_cumulative = forecast_data_1['cumulative_daily_consumption']

      # PERIODIC REVIEW SYSTEM with NAIVE forecasting
      #====================================================================================================================

  inventory_data_periodic_naive = inventory_data.copy()
  inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)


  for i in range(1,math.ceil(len(y)/R)+1):

    y_to_val = y[(i)*R:(i)*R+(R+L)]
        #y_set=y[0:(i+1)*R+(R+L)]
    epoch_no=i
    prediction = naive_forecast_method_lagged_daily(y,y_to_val,epoch_no,lag)
    y_previous_epoch=y[(i-1)*R:(i)*R]
    inventory_data_periodic_naive=peridoic_review(y_previous_epoch,inventory_data_periodic_naive,prediction,epoch_no)

      # PERIODIC REVIEW SYSTEM with HOLT forecasting
      #====================================================================================================================

  inventory_data_periodic_Holt = inventory_data
  inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

      #hospitalization_data_type='daily_consumption'

  try:
    hospitalization_data_type=hospitalization_data_type
  except NameError:
    hospitalization_data_type='daily_consumption'

  if hospitalization_data_type=='daily_consumption':
    y_holt=y
  else:
    y_holt=y_cumulative


  for i in range(1,math.ceil(len(y)/R)+1):
        
    y_to_val = y_holt[(i)*R:(i)*R+(R+L)]
    y_set=y_holt[0:(i)*R+(R+L)]
    y_previous_epoch_holt=y_holt[(i-1)*R:(i)*R]
    prediction = holt_lagged_daily(y_set,y_to_val,i,y_previous_epoch_holt,lag)
    y_previous_epoch=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_Holt=peridoic_review(y_previous_epoch,inventory_data_periodic_Holt,prediction,epoch_no)

    # PERIODIC REVIEW SYSTEM with SIR forecasting model
    #=========================================================================================================

  inventory_data_periodic_SIR = inventory_data.copy()
  inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)


  for i in range(1,math.ceil(len(y)/R)+1):
    y_to_train = y[0:(i)*R]
    y_to_val = y[(i)*R:(i)*R+(R+L)]
    y_set=y[0:(i)*R+(R+L)]
    prediction = SIR_forecast_model_lagged_daily(i,y_to_val,lag)
    y_previous_epoch=y[(i-1)*R:(i)*R]
    epoch_no=i
    inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction,epoch_no)

  # Putting everything into a dataframe
  #=========================================================================================================
  #putting in the main parameters of the scenarios in the data frames
  inventory_data_result_scenarios['Lead time']=L
  inventory_data_result_scenarios['SNR']=SNR
  inventory_data_result_scenarios['Protocol coefficient']=protocol_coef
  inventory_data_result_scenarios['Suppliers minimum capacity']=suppliers_capacity_min
  inventory_data_result_scenarios['Suppliers maximum capacity']=suppliers_capacity_max
  inventory_data_result_scenarios['Inventory capacity']=inventory_capactiy
  inventory_data_result_scenarios['Specified service level']=service_level
    # The total demand/consumption is the same for all methods, so we can pick any of them, in this case, SIR was picked
  inventory_data_result_scenarios['Total demand/consumption']=inventory_data_periodic_SIR['consumption in last epoch'].sum()

  # Puttin the analysis data into the data frame:
  #=========================================================================================================
  results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
  
  for i in range(len(results)):
    inventory_data_result_scenarios.iloc[[0],[(i*18)+7]]=results[i]['shortage'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+8]]=results[i]['current'].iloc[-1],
    inventory_data_result_scenarios.iloc[[0],[(i*18)+9]]=round((1-(results[i]['shortage'] != 0).sum()/len(results[i]['shortage']-1)),2)
    inventory_data_result_scenarios.iloc[[0],[(i*18)+10]]=(results[i]['shortage'] != 0).sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+11]]=results[i]['order'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+12]]=(results[i]['order'] != 0).sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+13]]=results[i]['real cost'].sum()
    inventory_data_result_scenarios.iloc[[0],[(i*18)+14]]=results[i]['total cost'].sum()
    # the below command compute average effective percentage bias, by average we specify those epochs that orders can be made
    # if becuase of the lead time, we cannot place an order in a specific epoch (the end epochs) then the percentage bias
    # is not relevent to our study 
    inventory_data_result_scenarios.iloc[[0],[(i*18)+15]]=results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean()
    
    # Counting the effective number of epochs with under-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+16]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] < 0).sum()
    
    # Counting the effective number of epochs with over-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+17]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] > 0).sum()
    
    # Calulating the percentage of effective number of epochs with under-forecast
    inventory_data_result_scenarios.iloc[[0],[(i*18)+18]]=round((inventory_data_result_scenarios.iloc[0][(i*12)+17])*100/((inventory_data_result_scenarios.iloc[0][(i*12)+16])+(inventory_data_result_scenarios.iloc[0][(i*12)+17])),2)

    #Effective RMSE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+19]]=results[i]['std of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean() 
    #RMSE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+20]]=results[i]['std of previous epoch'][1:-1].mean()
    #Effective MAPE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+21]]=results[i]['MAPE'][1:math.floor((len(y)-L)/R)+1].mean() 
    #MAPE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+22]]=results[i]['MAPE'][1:-1].mean()
    #Effective MAE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+23]]=results[i]['MAE'][1:math.floor((len(y)-L)/R)+1].mean() 
    #MAE
    inventory_data_result_scenarios.iloc[[0],[(i*18)+24]]=results[i]['MAE'][1:-1].mean()

  SIR_epochs_bias=((results[0].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('SIR percentage bias of epoch: ')
  Holt_epochs_bias=((results[1].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Holt percentage bias of epoch: ')
  Naive_epochs_bias=((results[2].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Naive percentage bias of epoch: ')
  
  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_bias,
                                             Holt_epochs_bias,
                                             Naive_epochs_bias],axis=1)

  SIR_epochs_shortage=(results[0].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('SIR shortage of epoch: ')
  Holt_epochs_shortage=(results[1].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Holt shortage of epoch: ')
  Naive_epochs_shortage=(results[2].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Naive shortage of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_shortage,
                                             Holt_epochs_shortage,
                                             Naive_epochs_shortage],axis=1)

  SIR_epochs_leftover_in=(results[0].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('SIR left-over inventory of epoch: ')
  Holt_epochs_leftover_in=(results[1].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Holt left-over inventory of epoch: ')
  Naive_epochs_leftover_in=(results[2].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Naive left-over inventory of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                             SIR_epochs_leftover_in,
                                             Holt_epochs_leftover_in,
                                             Naive_epochs_leftover_in],axis=1)
  
  SIR_order=(results[0].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('SIR order of epoch: ')
  Holt_order=(results[1].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Holt order of epoch: ')
  Naive_order=(results[2].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Naive order of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_order,
                                              Holt_order,
                                              Naive_order],axis=1)
  
  SIR_order_actual=(results[0].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('SIR order actual of epoch: ')
  Holt_order_actual=(results[1].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Holt order actual of epoch: ')
  Naive_order_actual=(results[2].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Naive order actual of epoch: ')

  inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_order_actual,
                                              Holt_order_actual,
                                              Naive_order_actual],axis=1)

  
  inventory_data_result_scenarios['lag']=lag
  #inventory_data_result_scenarios  
  complete_inventory_results_scenarios_lagged_daily=complete_inventory_results_scenarios_lagged_daily.append([inventory_data_result_scenarios],ignore_index=True)

#Exporting the data into the a csv file which will be saved into the drive
complete_inventory_results_scenarios_lagged_daily.to_csv('Inventory management results with %s scenarios(lagged data v2) lag %s.csv'%(no_scenarios,lag))

complete_inventory_results_scenarios_lagged_daily

complete_inventory_results_scenarios_lagged_daily

"""##### version II (day) variable lag:

###### lag creator
"""

# Creating 1000 iteration of random lag for the varibale scenario:
# the size is the number of epoch in this study
lag_df=pd.DataFrame()
for i in range(1000):
  temp_array=pd.Series(np.random.randint(lower_limit_lag, upper_limit_lag+1, size=math.ceil(len(y)/R)))
  lag_df=lag_df.append(temp_array,ignore_index=True)

#In this section, we analyze the lag to see if the lagged data at each epoch is shorter than the previous epoch:
lag_df_adj=lag_df.iloc[:0,:0]
for j in range(len(lag_df)):
  temp_len_lag_1=0
  temp_len_lag_2=0
  adj_var_lag=lag_df.iloc[j,:].copy()
  
  for i in range(1,math.ceil(len(y)/R)+1):
    no_day_in_epoch=i*R
    temp_len_lag_2=max(0,no_day_in_epoch-adj_var_lag[i-1])
    if temp_len_lag_1>=temp_len_lag_2:
      lag=no_day_in_epoch-temp_len_lag_1
    else:
      lag=adj_var_lag[i-1]
    
    temp_len_lag_1=no_day_in_epoch-lag
    adj_var_lag[i-1]=lag
  lag_df_adj=lag_df_adj.append(adj_var_lag,ignore_index=True,sort=None)

lag_df_adj

lag_df_adj.to_csv('adjusted variable lag df.csv')
lag_df.to_csv('variable lag df.csv')

"""###### importing the data

"""

lag_df_adj=pd.read_csv('adjusted variable lag df.csv')
lag_df_adj=lag_df_adj.drop(columns='Unnamed: 0')
lag_df_adj=lag_df_adj.astype(int)

# Choose the type of data to be used in the SIR forecasting method:
# Normal
# Replenished
# ==============================================================================================
# This is the "NORMAL" data type


#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
SIR_forecast_df=pd.read_csv('forecast SIR without cumulative-complete 122 days.csv')
#setting the date as the index of the dataframe
SIR_forecast_df=SIR_forecast_df.set_index(SIR_forecast_df['Date']).drop(columns='Date')
#rounding up the value of forecast in the data frame 
#SIR_forecast_df=np.ceil(SIR_forecast_df)

no_scenarios=1000

start_lag_array=960
end_lag_array=970
end_lag_array

"""###### Scenario Creator"""

#this code will pass a clean table to the loop by removing all the rows of the dataframes

inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()
complete_inventory_results_scenarios_lagged_daily_variable=inventory_data_result_scenarios.iloc[0:0].copy()
#inventory_data_result_scenarios = inventory_data_result_scenarios.iloc[0:0].copy()

for lag_no in range(start_lag_array,end_lag_array):
  var_lag=lag_df_adj.iloc[lag_no,:]


  inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()
  complete_inventory_results_scenarios_lagged_daily=inventory_data_result_scenarios.iloc[0:0].copy()
  #inventory_data_result_scenarios = inventory_data_result_scenarios.iloc[0:0].copy()

  for scenarios in range(no_scenarios):
    # Using uniform distribution to randomly choose a value for the following parameters
    inventory_data_result_scenarios=inventory_data_result_scenarios_empty.copy()

    minimum_pallet = complete_sim_scenarios['Min pallet'][scenarios]
    maximum_pallet = complete_sim_scenarios['Max pallet'][scenarios]
    protocol_coef = complete_sim_scenarios['Protocol coefficient'][scenarios]
    L = complete_sim_scenarios['Lead time'][scenarios]
    inventory_capactiy_in_pallet = complete_sim_scenarios['Inventory capacity (in pallet)'][scenarios]
    service_level = complete_sim_scenarios['Specified service level'][scenarios]
    SNR = complete_sim_scenarios['SNR'][scenarios]
    #lag = complete_sim_scenarios['Lag'][scenarios]

    # Calculating the following varibales
    item_in_pallet= item_in_box * box_in_pallet
    suppliers_capacity_min= item_in_pallet * minimum_pallet
    suppliers_capacity_max= item_in_pallet * maximum_pallet
    inventory_capactiy= inventory_capactiy_in_pallet * item_in_pallet


    #this code will pass a clean table to the loop by removing all the rows of the dataframes
    complete_inventory_results=inventory_data_result.iloc[0:0].copy()
    inventory_data_result = inventory_data_result.iloc[0:0].copy()



      #DATA SIMULATION
      #==============================================================================
      #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
      #noise recalculation should only be applied if we have have different coefficient of protocol, 
      #otherwise noise should stya the same, thats why it is outside of the loop
      
    provincial_data_CIHI['daily_consumption']=sim_demand.iloc[:,scenarios]
    
    provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()


        # >>>USER INPUT<<<
        # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
        # ======================================================================================================
    forecast_data_1 = provincial_data_CIHI.copy()

        # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
    consumption_date_type='daily_consumption'

    y = forecast_data_1[consumption_date_type]
    y_cumulative = forecast_data_1['cumulative_daily_consumption']

        # PERIODIC REVIEW SYSTEM with NAIVE forecasting
        #====================================================================================================================

    inventory_data_periodic_naive = inventory_data.copy()
    inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)


    for i in range(1,math.ceil(len(y)/R)+1):

      y_to_val = y[(i)*R:(i)*R+(R+L)]
          #y_set=y[0:(i+1)*R+(R+L)]
      epoch_no=i
      prediction = naive_forecast_method_lagged_daily(y,y_to_val,epoch_no,var_lag[i-1])
      y_previous_epoch=y[(i-1)*R:(i)*R]
      inventory_data_periodic_naive=peridoic_review(y_previous_epoch,inventory_data_periodic_naive,prediction,epoch_no)

        # PERIODIC REVIEW SYSTEM with HOLT forecasting
        #====================================================================================================================

    inventory_data_periodic_Holt = inventory_data
    inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

        #hospitalization_data_type='daily_consumption'

    try:
      hospitalization_data_type=hospitalization_data_type
    except NameError:
      hospitalization_data_type='daily_consumption'

    if hospitalization_data_type=='daily_consumption':
      y_holt=y
    else:
      y_holt=y_cumulative


    for i in range(1,math.ceil(len(y)/R)+1):
          
      y_to_val = y_holt[(i)*R:(i)*R+(R+L)]
      y_set=y_holt[0:(i)*R+(R+L)]
      y_previous_epoch_holt=y_holt[(i-1)*R:(i)*R]
      prediction = holt_lagged_daily(y_set,y_to_val,i,y_previous_epoch_holt,var_lag[i-1])
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_Holt=peridoic_review(y_previous_epoch,inventory_data_periodic_Holt,prediction,epoch_no)

      # PERIODIC REVIEW SYSTEM with SIR forecasting model
      #=========================================================================================================

    inventory_data_periodic_SIR = inventory_data.copy()
    inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)


    for i in range(1,math.ceil(len(y)/R)+1):
      y_to_train = y[0:(i)*R]
      y_to_val = y[(i)*R:(i)*R+(R+L)]
      y_set=y[0:(i)*R+(R+L)]
      prediction = SIR_forecast_model_lagged_daily(i,y_to_val,var_lag[i-1])
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction,epoch_no)

    # Putting everything into a dataframe
    #=========================================================================================================
    #putting in the main parameters of the scenarios in the data frames
    inventory_data_result_scenarios['Lead time']=L
    inventory_data_result_scenarios['SNR']=SNR
    inventory_data_result_scenarios['Protocol coefficient']=protocol_coef
    inventory_data_result_scenarios['Suppliers minimum capacity']=suppliers_capacity_min
    inventory_data_result_scenarios['Suppliers maximum capacity']=suppliers_capacity_max
    inventory_data_result_scenarios['Inventory capacity']=inventory_capactiy
    inventory_data_result_scenarios['Specified service level']=service_level
      # The total demand/consumption is the same for all methods, so we can pick any of them, in this case, SIR was picked
    inventory_data_result_scenarios['Total demand/consumption']=inventory_data_periodic_SIR['consumption in last epoch'].sum()

    # Puttin the analysis data into the data frame:
    #=========================================================================================================
    results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
    
    for i in range(len(results)):
      inventory_data_result_scenarios.iloc[[0],[(i*18)+7]]=results[i]['shortage'].sum()
      inventory_data_result_scenarios.iloc[[0],[(i*18)+8]]=results[i]['current'].iloc[-1],
      inventory_data_result_scenarios.iloc[[0],[(i*18)+9]]=round((1-(results[i]['shortage'] != 0).sum()/len(results[i]['shortage']-1)),2)
      inventory_data_result_scenarios.iloc[[0],[(i*18)+10]]=(results[i]['shortage'] != 0).sum()
      inventory_data_result_scenarios.iloc[[0],[(i*18)+11]]=results[i]['order'].sum()
      inventory_data_result_scenarios.iloc[[0],[(i*18)+12]]=(results[i]['order'] != 0).sum()
      inventory_data_result_scenarios.iloc[[0],[(i*18)+13]]=results[i]['real cost'].sum()
      inventory_data_result_scenarios.iloc[[0],[(i*18)+14]]=results[i]['total cost'].sum()
      # the below command compute average effective percentage bias, by average we specify those epochs that orders can be made
      # if becuase of the lead time, we cannot place an order in a specific epoch (the end epochs) then the percentage bias
      # is not relevent to our study 
      inventory_data_result_scenarios.iloc[[0],[(i*18)+15]]=results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean()
      
      # Counting the effective number of epochs with under-forecast
      inventory_data_result_scenarios.iloc[[0],[(i*18)+16]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] < 0).sum()
      
      # Counting the effective number of epochs with over-forecast
      inventory_data_result_scenarios.iloc[[0],[(i*18)+17]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] > 0).sum()
      
      # Calulating the percentage of effective number of epochs with under-forecast
      inventory_data_result_scenarios.iloc[[0],[(i*18)+18]]=round((inventory_data_result_scenarios.iloc[0][(i*12)+17])*100/((inventory_data_result_scenarios.iloc[0][(i*12)+16])+(inventory_data_result_scenarios.iloc[0][(i*12)+17])),2)

      #Effective RMSE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+19]]=results[i]['std of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean() 
      #RMSE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+20]]=results[i]['std of previous epoch'][1:-1].mean()
      #Effective MAPE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+21]]=results[i]['MAPE'][1:math.floor((len(y)-L)/R)+1].mean() 
      #MAPE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+22]]=results[i]['MAPE'][1:-1].mean()
      #Effective MAE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+23]]=results[i]['MAE'][1:math.floor((len(y)-L)/R)+1].mean() 
      #MAE
      inventory_data_result_scenarios.iloc[[0],[(i*18)+24]]=results[i]['MAE'][1:-1].mean()

    SIR_epochs_bias=((results[0].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('SIR percentage bias of epoch: ')
    Holt_epochs_bias=((results[1].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Holt percentage bias of epoch: ')
    Naive_epochs_bias=((results[2].iloc[:,[14]].T).reset_index()).iloc[:,1:-1].add_prefix('Naive percentage bias of epoch: ')
    
    inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_epochs_bias,
                                              Holt_epochs_bias,
                                              Naive_epochs_bias],axis=1)

    SIR_epochs_shortage=(results[0].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('SIR shortage of epoch: ')
    Holt_epochs_shortage=(results[1].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Holt shortage of epoch: ')
    Naive_epochs_shortage=(results[2].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Naive shortage of epoch: ')

    inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_epochs_shortage,
                                              Holt_epochs_shortage,
                                              Naive_epochs_shortage],axis=1)

    SIR_epochs_leftover_in=(results[0].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('SIR left-over inventory of epoch: ')
    Holt_epochs_leftover_in=(results[1].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Holt left-over inventory of epoch: ')
    Naive_epochs_leftover_in=(results[2].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Naive left-over inventory of epoch: ')

    inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                              SIR_epochs_leftover_in,
                                              Holt_epochs_leftover_in,
                                              Naive_epochs_leftover_in],axis=1)
    
    SIR_order=(results[0].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('SIR order of epoch: ')
    Holt_order=(results[1].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Holt order of epoch: ')
    Naive_order=(results[2].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Naive order of epoch: ')

    inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                                SIR_order,
                                                Holt_order,
                                                Naive_order],axis=1)
    
    SIR_order_actual=(results[0].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('SIR order actual of epoch: ')
    Holt_order_actual=(results[1].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Holt order actual of epoch: ')
    Naive_order_actual=(results[2].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Naive order actual of epoch: ')

    inventory_data_result_scenarios=pd.concat([inventory_data_result_scenarios,
                                                SIR_order_actual,
                                                Holt_order_actual,
                                                Naive_order_actual],axis=1)

    
    #inventory_data_result_scenarios['lag']=lag
    #inventory_data_result_scenarios  
    complete_inventory_results_scenarios_lagged_daily=complete_inventory_results_scenarios_lagged_daily.append([inventory_data_result_scenarios],ignore_index=True)
  complete_inventory_results_scenarios_lagged_daily_variable=complete_inventory_results_scenarios_lagged_daily_variable.append([complete_inventory_results_scenarios_lagged_daily],ignore_index=True)

complete_inventory_results_scenarios_lagged_daily_variable.to_csv('Results of var lag %dto%d.csv'%(start_lag_array,end_lag_array))

complete_inventory_results_scenarios_lagged_daily_variable.to_csv('Results of var lag %dto%d.csv'%(start_lag_array,end_lag_array))

complete_inventory_results_scenarios_lagged_daily_variable

"""#### Results

##### Importing data
"""

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
complete_inventory_results_scenarios_lagged_daily=pd.read_csv('Inventory management results with 10000 scenarios(lagged data v2) - 2.csv')
#setting the date as the index of the dataframe
complete_inventory_results_scenarios_lagged_daily=complete_inventory_results_scenarios_lagged_daily.drop(columns='Unnamed: 0')
complete_inventory_results_scenarios_lagged_daily=complete_inventory_results_scenarios_lagged_daily.iloc[:1000,:]

"""##### Presentation of results

###### Data filtering
"""

#creating a separet daraframe in order to not touch the original copy
df_temp_present_lagged_daily=complete_inventory_results_scenarios_lagged_daily.copy()

# SHORTAGE
#============================================
# filtering the data for sepcified category
shortage_epoch_df_SIR_lagged_daily=df_temp_present_lagged_daily[df_temp_present_lagged_daily.columns[pd.Series(df_temp_present_lagged_daily.columns).str.startswith('SIR shortage of epoch')]]
shortage_epoch_df_Holt_lagged_daily=df_temp_present_lagged_daily[df_temp_present_lagged_daily.columns[pd.Series(df_temp_present_lagged_daily.columns).str.startswith('Holt shortage of epoch')]]
shortage_epoch_df_Naive_lagged_daily=df_temp_present_lagged_daily[df_temp_present_lagged_daily.columns[pd.Series(df_temp_present_lagged_daily.columns).str.startswith('Naive shortage of epoch')]]


# Left-over inventory
#============================================
# filtering the data for sepcified category
leftover_epoch_df_SIR_lagged_daily=df_temp_present_lagged_daily[df_temp_present_lagged_daily.columns[pd.Series(df_temp_present_lagged_daily.columns).str.startswith('SIR left-over inventory of epoch')]]
leftover_epoch_df_Holt_lagged_daily=df_temp_present_lagged_daily[df_temp_present_lagged_daily.columns[pd.Series(df_temp_present_lagged_daily.columns).str.startswith('Holt left-over inventory of epoch')]]
leftover_epoch_df_Naive_lagged_daily=df_temp_present_lagged_daily[df_temp_present_lagged_daily.columns[pd.Series(df_temp_present_lagged_daily.columns).str.startswith('Naive left-over inventory of epoch')]]

# Bais-over inventory
#============================================
# filtering the data for sepcified category
bias_epoch_df_SIR_lagged_daily=df_temp_present_lagged_daily[df_temp_present_lagged_daily.columns[pd.Series(df_temp_present_lagged_daily.columns).str.startswith('SIR percentage bias of epoch')]]
bias_epoch_df_Holt_lagged_daily=df_temp_present_lagged_daily[df_temp_present_lagged_daily.columns[pd.Series(df_temp_present_lagged_daily.columns).str.startswith('Holt percentage bias of epoch')]]
bias_epoch_df_Naive_lagged_daily=df_temp_present_lagged_daily[df_temp_present_lagged_daily.columns[pd.Series(df_temp_present_lagged_daily.columns).str.startswith('Naive percentage bias of epoch')]]

#creating a separet daraframe in order to not touch the original copy
#df_temp_present_lagged=complete_inventory_results_scenarios_lagged.copy()

# SHORTAGE
#============================================
# filtering the data for sepcified category
#shortage_epoch_df_SIR_lagged=df_temp_present_lagged[df_temp_present_lagged.columns[pd.Series(df_temp_present_lagged.columns).str.startswith('SIR shortage of epoch')]]
#shortage_epoch_df_Holt_lagged=df_temp_present_lagged[df_temp_present_lagged.columns[pd.Series(df_temp_present_lagged.columns).str.startswith('Holt shortage of epoch')]]
#shortage_epoch_df_Naive_lagged=df_temp_present_lagged[df_temp_present_lagged.columns[pd.Series(df_temp_present_lagged.columns).str.startswith('Naive shortage of epoch')]]


# Left-over inventory
#============================================
# filtering the data for sepcified category
#leftover_epoch_df_SIR_lagged=df_temp_present_lagged[df_temp_present_lagged.columns[pd.Series(df_temp_present_lagged.columns).str.startswith('SIR left-over inventory of epoch')]]
#leftover_epoch_df_Holt_lagged=df_temp_present_lagged[df_temp_present_lagged.columns[pd.Series(df_temp_present_lagged.columns).str.startswith('Holt left-over inventory of epoch')]]
#leftover_epoch_df_Naive_lagged=df_temp_present_lagged[df_temp_present_lagged.columns[pd.Series(df_temp_present_lagged.columns).str.startswith('Naive left-over inventory of epoch')]]

# Bais-over inventory
#============================================
# filtering the data for sepcified category
#bias_epoch_df_SIR_lagged=df_temp_present_lagged[df_temp_present_lagged.columns[pd.Series(df_temp_present_lagged.columns).str.startswith('SIR percentage bias of epoch')]]
#bias_epoch_df_Holt_lagged=df_temp_present_lagged[df_temp_present_lagged.columns[pd.Series(df_temp_present_lagged.columns).str.startswith('Holt percentage bias of epoch')]]
#bias_epoch_df_Naive_lagged=df_temp_present_lagged[df_temp_present_lagged.columns[pd.Series(df_temp_present_lagged.columns).str.startswith('Naive percentage bias of epoch')]]

"""###### Shortage vs epoch"""

shortage_epoch_df_SIR_relative_to_demand_lagged_daily=100*shortage_epoch_df_SIR_lagged_daily.div(grouped_demand.values)
shortage_epoch_df_Holt_relative_to_demand_lagged_daily=100*shortage_epoch_df_Holt_lagged_daily.div(grouped_demand.values)
shortage_epoch_df_Naive_relative_to_demand_lagged_daily=100*shortage_epoch_df_Naive_lagged_daily.div(grouped_demand.values)


graph_generator_for_epochs(shortage_epoch_df_SIR_relative_to_demand_lagged_daily,
                           shortage_epoch_df_Holt_relative_to_demand_lagged_daily,
                           shortage_epoch_df_Naive_relative_to_demand_lagged_daily,'Shortage','Normal')

#10000
graph_generator_for_epochs(shortage_epoch_df_SIR_lagged_daily,
                           shortage_epoch_df_Holt_lagged_daily,
                           shortage_epoch_df_Naive_lagged_daily,'Shortage','delay')

#100
graph_generator_for_epochs(shortage_epoch_df_SIR_lagged_daily,
                           shortage_epoch_df_Holt_lagged_daily,
                           shortage_epoch_df_Naive_lagged_daily,'Shortage','delay')

#epoch
#graph_generator_for_epochs(shortage_epoch_df_SIR_lagged,
#                           shortage_epoch_df_Holt_lagged,
#                           shortage_epoch_df_Naive_lagged,'Shortage')

"""###### Left-over inventory vs epoch"""

leftover_epoch_df_SIR_relative_to_demand_lagged_daily=100*leftover_epoch_df_SIR_lagged_daily.div(grouped_demand.values)
leftover_epoch_df_Holt_relative_to_demand_lagged_daily=100*leftover_epoch_df_Holt_lagged_daily.div(grouped_demand.values)
leftover_epoch_df_Naive_relative_to_demand_lagged_daily=100*leftover_epoch_df_Naive_lagged_daily.div(grouped_demand.values)

leftover_epoch_df_SIR_relative_to_demand_lagged_daily.iloc[:,0]=0
leftover_epoch_df_Holt_relative_to_demand_lagged_daily.iloc[:,0]=0
leftover_epoch_df_Naive_relative_to_demand_lagged_daily.iloc[:,0]=0

# 10000
graph_generator_for_epochs(leftover_epoch_df_SIR_relative_to_demand_lagged_daily,
                           leftover_epoch_df_Holt_relative_to_demand_lagged_daily,
                           leftover_epoch_df_Naive_relative_to_demand_lagged_daily,'Inventory level','delay')

# 10,000
graph_generator_for_epochs(leftover_epoch_df_SIR_lagged_daily,
                           leftover_epoch_df_Holt_lagged_daily,
                           leftover_epoch_df_Naive_lagged_daily,'Inventory level','delay')

# epoch
#graph_generator_for_epochs(leftover_epoch_df_SIR_lagged,
#                           leftover_epoch_df_Holt_lagged,
#                           leftover_epoch_df_Naive_lagged,'Shortage','delay')

"""###### Percentage bias vs epoch"""

#10000
graph_generator_for_epochs(bias_epoch_df_SIR_lagged_daily,
                           bias_epoch_df_Holt_lagged_daily,
                           bias_epoch_df_Naive_lagged_daily,'Percentage bias')

#100
graph_generator_for_epochs(bias_epoch_df_SIR_lagged_daily,
                           bias_epoch_df_Holt_lagged_daily,
                           bias_epoch_df_Naive_lagged_daily,'Percentage bias','delay')

# epoch
graph_generator_for_epochs(bias_epoch_df_SIR_lagged,
                           bias_epoch_df_Holt_lagged,
                           bias_epoch_df_Naive_lagged,'Percentage bias')

"""###### Shortage vs. final inventory"""

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)

x_plot=df_temp_present_lagged['SIR forecast: Shortage']
y_plot=df_temp_present_lagged['SIR forecast: Inventory level at the end']
plt.scatter(x_plot, y_plot, color = 'red',label='SIR forecast',marker='o')

x_plot=df_temp_present_lagged['Holt forecast: Shortage']
y_plot=df_temp_present_lagged['Holt forecast: Inventory level at the end']
plt.scatter(x_plot, y_plot, color = 'blue',label='Holt forecast',marker='x')

x_plot=df_temp_present_lagged['Naive forecast: Shortage']
y_plot=df_temp_present_lagged['Naive forecast: Inventory level at the end']
plt.scatter(x_plot, y_plot, color = 'green',label='Naive forecast',marker='^')

plt.title('%s scnearios of inventory management in British Columbia'%(no_scenarios),fontsize=16)
plt.xlabel('Shortage',fontsize=16)
plt.ylabel('Final inventory',fontsize=16)
plt.legend(fontsize=13)

plt.show()

"""## Scenario IV. Scenario II: CC deviation; Noisy data

#### Parameters
"""

# In this section the parameters for ranges of scenarios are defined
#==============================================================================



# What is the upper limit of deviation?
CC_dev_upper_limit=300
# What is the lower limit of devation?
CC_dev_lower_limit=225
# what is the steps?
CC_dev_step=25

no_scenarios=1000

"""#### Scenario creation"""

List_CC_dev_step=np.arange(CC_dev_lower_limit,(CC_dev_upper_limit+CC_dev_step), CC_dev_step).tolist()
List_CC_dev_step

y

y_adjusted

i=6
y_to_train = y_adjusted[0:(i)*R]
y_to_val = y[(i)*R:(i)*R+(R+L)]
y_set=y_adjusted[0:(i)*R+(R+L)]
y_previous_epoch_holt=y_adjusted[(i-1)*R:(i)*R]
prediction_holt = holt(y_set, y_to_train,y_to_val,i,y_previous_epoch_holt)
prediction_holt

# Choose the type of data to be used in the SIR forecasting method:
# Normal
# Replenished
# ==============================================================================================
# This is the "NORMAL" data type


#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
SIR_forecast_df=pd.read_csv('forecast SIR without cumulative-complete 18 epochs.csv')
#setting the date as the index of the dataframe
SIR_forecast_df=SIR_forecast_df.set_index(SIR_forecast_df['Date']).drop(columns='Date')
#rounding up the value of forecast in the data frame 
#SIR_forecast_df=np.ceil(SIR_forecast_df)

#this code will pass a clean table to the loop by removing all the rows of the dataframes

inventory_data_result_scenarios_combine=inventory_data_result_scenarios_combine_empty.copy()
complete_inventory_results_scenarios_combine=inventory_data_result_scenarios_combine.iloc[0:0].copy()
#inventory_data_result_scenarios = inventory_data_result_scenarios.iloc[0:0].copy()

List_CC_dev_step=np.arange(CC_dev_lower_limit,(CC_dev_upper_limit+CC_dev_step), CC_dev_step).tolist()

for scenarios in range(no_scenarios):
  # Using uniform distribution to randomly choose a value for the following parameters

  minimum_pallet = complete_sim_scenarios['Min pallet'][scenarios]
  maximum_pallet = complete_sim_scenarios['Max pallet'][scenarios]
  protocol_coef = complete_sim_scenarios['Protocol coefficient'][scenarios]
  L = complete_sim_scenarios['Lead time'][scenarios]
  inventory_capactiy_in_pallet = complete_sim_scenarios['Inventory capacity (in pallet)'][scenarios]
  service_level = complete_sim_scenarios['Specified service level'][scenarios]
  SNR = complete_sim_scenarios['SNR'][scenarios]






  # Calculating the following varibales
  item_in_pallet= item_in_box * box_in_pallet
  suppliers_capacity_min= item_in_pallet * minimum_pallet
  suppliers_capacity_max= item_in_pallet * maximum_pallet
  inventory_capactiy= inventory_capactiy_in_pallet * item_in_pallet
  
  
  # This will create noise for CC based on the SNR, we need this noise to be the same, since in the next section
  # only CC is being changed based on our perception of consumption, but the actual noise for the actual consumption
  # remains the same

  noise_for_specified_CC=sim_noise.iloc[:,scenarios]


  #this code will pass a clean table to the loop by removing all the rows of the dataframes
  #complete_inventory_results=inventory_data_result.iloc[0:0].copy()
  #inventory_data_result = inventory_data_result.iloc[0:0].copy()

  for CC_dev in range(len(List_CC_dev_step)):
    inventory_data_result_scenarios_combine=inventory_data_result_scenarios_combine_empty.copy()

      #DATA SIMULATION
      #==============================================================================
      #For the context of this project we use "consumption" for the simulated real data and "demand" for the forecast of this consumption.
      #noise recalculation should only be applied if we have have different coefficient of protocol, 
      #otherwise noise should stya the same, thats why it is outside of the loop
      
    #creating 2 sets of simulated data with the same noise,one for actuall demand and one for deviated demand
    simulated_data_with_CCdev=noise_creator_with_SNR_CCdev_2(provincial_data_CIHI['daily_Hospitalizations'],
                                                             protocol_coef,
                                                             List_CC_dev_step[CC_dev],
                                                             noise_for_specified_CC)
    #provincial_data_CIHI['daily_consumption']=noise_creator_with_SNR(provincial_data_CIHI['daily_Hospitalizations'],
    #                                                                  protocol_coef)
    
    provincial_data_CIHI['daily_consumption']=simulated_data_with_CCdev[0]
    provincial_data_CIHI['daily_consumption_adjusted_with_CCdev']=simulated_data_with_CCdev[1]
    
    
    provincial_data_CIHI['cumulative_daily_consumption']=provincial_data_CIHI['daily_consumption'].cumsum()



        # >>>USER INPUT<<<
        # in this section we assign the desired data to a variable 'y' which will be used throughout this section 
        # ======================================================================================================
    forecast_data_1 = provincial_data_CIHI.copy()

        # Important Note: User needs to specify what type of data needs to be used for forecast: 1 - 'cumulative_daily_consumption', 2 - 'daily_consumption'
    consumption_date_type='daily_consumption'

    y = forecast_data_1[consumption_date_type]
    y_cumulative = forecast_data_1['cumulative_daily_consumption']

    y_adjusted=forecast_data_1['daily_consumption_adjusted_with_CCdev']

        # PERIODIC REVIEW SYSTEM with NAIVE forecasting
        #====================================================================================================================

    inventory_data_periodic_naive = inventory_data.copy()
    inventory_data_periodic_naive.drop(inventory_data_periodic_naive.index[1:],0,inplace=True)


    for i in range(1,math.ceil(len(y)/R)+1):
      y_to_train = y_adjusted[max(0,i-2)*R:(i)*R]

      y_to_val = y[(i)*R:(i)*R+(R+L)]
          #y_set=y[0:(i+1)*R+(R+L)]
      prediction_naive = naive_forecast_method(y_to_train,y_to_val)
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_naive=peridoic_review(y_previous_epoch,inventory_data_periodic_naive,prediction_naive,epoch_no)

        # PERIODIC REVIEW SYSTEM with HOLT forecasting
        #====================================================================================================================

    inventory_data_periodic_Holt = inventory_data
    inventory_data_periodic_Holt.drop(inventory_data_periodic_Holt.index[1:],0,inplace=True)

        #hospitalization_data_type='daily_consumption'

    try:
      hospitalization_data_type=hospitalization_data_type
    except NameError:
      hospitalization_data_type='daily_consumption'

    if hospitalization_data_type=='daily_consumption':
      y_holt=y
    else:
      y_holt=y_cumulative


    for i in range(1,math.ceil(len(y)/R)+1):
      y_to_train = y_adjusted[0:(i)*R]
      y_to_val = y[(i)*R:(i)*R+(R+L)]
      y_set=y_adjusted[0:(i)*R+(R+L)]
      y_previous_epoch_holt=y_adjusted[(i-1)*R:(i)*R]
      prediction_holt = holt(y_set, y_to_train,y_to_val,i,y_previous_epoch_holt)
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_Holt=peridoic_review(y_previous_epoch,inventory_data_periodic_Holt,prediction_holt,epoch_no)

      # PERIODIC REVIEW SYSTEM with SIR forecasting model
      #=========================================================================================================

    inventory_data_periodic_SIR = inventory_data.copy()
    inventory_data_periodic_SIR.drop(inventory_data_periodic_SIR.index[1:],0,inplace=True)


    for i in range(1,math.ceil(len(y)/R)+1):
      y_to_train = y[0:(i)*R]
      y_to_val = y[(i)*R:(i)*R+(R+L)]
      y_set=y[0:(i)*R+(R+L)]
      prediction_SIR = SIR_forecast_model_devited(i,y_to_val,List_CC_dev_step[CC_dev])
      y_previous_epoch=y[(i-1)*R:(i)*R]
      epoch_no=i
      inventory_data_periodic_SIR=peridoic_review(y_previous_epoch,inventory_data_periodic_SIR,prediction_SIR,epoch_no)

    # Putting everything into a dataframe
    #=========================================================================================================
    #putting in the main parameters of the scenarios in the data frames
    inventory_data_result_scenarios_combine['Lead time']=L
    inventory_data_result_scenarios_combine['SNR']=SNR
    inventory_data_result_scenarios_combine['Protocol coefficient']=protocol_coef
    inventory_data_result_scenarios_combine['Suppliers minimum capacity']=suppliers_capacity_min
    inventory_data_result_scenarios_combine['Suppliers maximum capacity']=suppliers_capacity_max
    inventory_data_result_scenarios_combine['Inventory capacity']=inventory_capactiy
    inventory_data_result_scenarios_combine['Specified service level']=service_level
      # The total demand/consumption is the same for all methods, so we can pick any of them, in this case, SIR was picked
    inventory_data_result_scenarios_combine['Total demand/consumption']=inventory_data_periodic_SIR['consumption in last epoch'].sum()
    inventory_data_result_scenarios_combine['CC deviation']=List_CC_dev_step[CC_dev]

    # Puttin the analysis data into the data frame:
    #=========================================================================================================
    results=(inventory_data_periodic_SIR,inventory_data_periodic_Holt,inventory_data_periodic_naive)
    
    for i in range(len(results)):
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+7]]=results[i]['shortage'].sum()
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+8]]=results[i]['current'].iloc[-1],
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+9]]=round((1-(results[i]['shortage'] != 0).sum()/len(results[i]['shortage']-1)),2)
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+10]]=(results[i]['shortage'] != 0).sum()
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+11]]=results[i]['order'].sum()
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+12]]=(results[i]['order'] != 0).sum()
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+13]]=results[i]['real cost'].sum()
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+14]]=results[i]['total cost'].sum()
      # the below command compute average effective percentage bias, by average we specify those epochs that orders can be made
      # if becuase of the lead time, we cannot place an order in a specific epoch (the end epochs) then the percentage bias
      # is not relevent to our study 
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+15]]=results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean()
      
      # Counting the effective number of epochs with under-forecast
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+16]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] < 0).sum()
      
      # Counting the effective number of epochs with over-forecast
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+17]]=(results[i]['percentage bias of previous epoch'][1:math.floor((len(y)-L)/R)+1] > 0).sum()
      
      # Calulating the percentage of effective number of epochs with under-forecast
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+18]]=round((inventory_data_result_scenarios_combine.iloc[0][(i*12)+17])*100/((inventory_data_result_scenarios_combine.iloc[0][(i*12)+16])+(inventory_data_result_scenarios_combine.iloc[0][(i*12)+17])),2)

      #Effective RMSE
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+19]]=results[i]['std of previous epoch'][1:math.floor((len(y)-L)/R)+1].mean() 
      #RMSE
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+20]]=results[i]['std of previous epoch'][1:-1].mean()
      #Effective MAPE
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+21]]=results[i]['MAPE'][1:math.floor((len(y)-L)/R)+1].mean() 
      #MAPE
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+22]]=results[i]['MAPE'][1:-1].mean()
      #Effective MAE
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+23]]=results[i]['MAE'][1:math.floor((len(y)-L)/R)+1].mean() 
      #MAE
      inventory_data_result_scenarios_combine.iloc[[0],[(i*18)+24]]=results[i]['MAE'][1:-1].mean()

    SIR_epochs_bias=((results[0].T.tail(1)).reset_index()).iloc[:,1:-1].add_prefix('SIR percentage bias of epoch: ')
    Holt_epochs_bias=((results[1].T.tail(1)).reset_index()).iloc[:,1:-1].add_prefix('Holt percentage bias of epoch: ')
    Naive_epochs_bias=((results[2].T.tail(1)).reset_index()).iloc[:,1:-1].add_prefix('Naive percentage bias of epoch: ')
    
    inventory_data_result_scenarios_combine=pd.concat([inventory_data_result_scenarios_combine,
                                              SIR_epochs_bias,
                                              Holt_epochs_bias,
                                              Naive_epochs_bias],axis=1)
    
    SIR_epochs_shortage=(results[0].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('SIR shortage of epoch: ')
    Holt_epochs_shortage=(results[1].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Holt shortage of epoch: ')
    Naive_epochs_shortage=(results[2].iloc[:,[3]].T.reset_index()).iloc[:,1:].add_prefix('Naive shortage of epoch: ')

    inventory_data_result_scenarios_combine=pd.concat([inventory_data_result_scenarios_combine,
                                              SIR_epochs_shortage,
                                              Holt_epochs_shortage,
                                              Naive_epochs_shortage],axis=1)

    SIR_epochs_leftover_in=(results[0].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('SIR left-over inventory of epoch: ')
    Holt_epochs_leftover_in=(results[1].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Holt left-over inventory of epoch: ')
    Naive_epochs_leftover_in=(results[2].iloc[:,[1]].T.reset_index()).iloc[:,1:].add_prefix('Naive left-over inventory of epoch: ')

    inventory_data_result_scenarios_combine=pd.concat([inventory_data_result_scenarios_combine,
                                              SIR_epochs_leftover_in,
                                              Holt_epochs_leftover_in,
                                              Naive_epochs_leftover_in],axis=1)
    
    SIR_order=(results[0].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('SIR order of epoch: ')
    Holt_order=(results[1].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Holt order of epoch: ')
    Naive_order=(results[2].iloc[:,[2]].T.reset_index()).iloc[:,1:].add_prefix('Naive order of epoch: ')

    inventory_data_result_scenarios_combine=pd.concat([inventory_data_result_scenarios_combine,
                                                SIR_order,
                                                Holt_order,
                                                Naive_order],axis=1)
    
    SIR_order_actual=(results[0].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('SIR order actual of epoch: ')
    Holt_order_actual=(results[1].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Holt order actual of epoch: ')
    Naive_order_actual=(results[2].iloc[:,[15]].T.reset_index()).iloc[:,1:].add_prefix('Naive order actual of epoch: ')

    inventory_data_result_scenarios_combine=pd.concat([inventory_data_result_scenarios_combine,
                                                SIR_order_actual,
                                                Holt_order_actual,
                                                Naive_order_actual],axis=1)


    #inventory_data_result_scenarios  
    complete_inventory_results_scenarios_combine=complete_inventory_results_scenarios_combine.append([inventory_data_result_scenarios_combine],ignore_index=True)

#Exporting the data into the a csv file which will be saved into the drive
complete_inventory_results_scenarios_combine.to_csv('Combine method Inventory management results with %s scenarios(un-toched data)225 - 300.csv'%(no_scenarios))

complete_inventory_results_scenarios_combine

"""#### Results

##### Importing data
"""

#the forecast has already been performed and the data has been stored in the csv file
#importing the csv file of forecast
complete_inventory_results_scenarios_combine=pd.read_csv('Combine method Inventory management results with 1000 scenarios(un-toched data)100 - 200.csv')
#setting the date as the index of the dataframe
complete_inventory_results_scenarios_combine=complete_inventory_results_scenarios_combine.drop(columns='Unnamed: 0')
#rounding up the value of forecast in the data frame 
#SIR_forecast_df=np.ceil(SIR_forecast_df)
df_data_analysis=(complete_inventory_results_scenarios_combine.copy().reset_index()).drop(columns='index')
#df_data_analysis = df_data_analysis.apply(pd.to_numeric)

List_CC_dev_step=np.arange(CC_dev_lower_limit,(CC_dev_upper_limit+CC_dev_step), CC_dev_step).tolist()

"""##### Shortage and left-over inventory vs deviation"""

# Filtering the data for graph:

# Normal data
df_sh_in_combine=complete_inventory_results_scenarios_combine[['CC deviation','SIR forecast: Shortage',
                  'Holt forecast: Shortage',
                  'Naive forecast: Shortage',
                  'SIR forecast: Inventory level at the end',
                  'Holt forecast: Inventory level at the end',
                  'Naive forecast: Inventory level at the end']]

# Adding the total consumption of each iteration to the dataframe
df_sh_in_combine['Total demand/consumption']=complete_inventory_results_scenarios_combine['Total demand/consumption']

# getting the mean of each step in CC devaition
df_presentation_combine=df_sh_in_combine.groupby('CC deviation').mean()

# Relative data
df_sh_in_combine_relative=df_sh_in_combine.copy()
df_sh_in_combine_relative.iloc[:,1:7]=100*df_sh_in_combine.iloc[:,1:7].div(df_sh_in_combine.iloc[:,7],axis=0)
df_presentation_combine_relative=df_sh_in_combine_relative.groupby('CC deviation').mean()

#protocol_coef_variation=[3,3.5,4,4.5,5]
#service_level_variation

df_data=df_presentation_combine_relative
#df_data=df_presentation_combine

plt.figure(figsize=(23,10))

plt.subplot(1, 2, 1)
#this part plots different Protocol coefficient and their related shortages
#plt.figure(figsize=(17,8.5))

x_combine=List_CC_dev_step
y_shortage_SIR_combine=df_data['SIR forecast: Shortage']
y_shortage_Holt_combine=df_data['Holt forecast: Shortage']
y_shortage_Naive_combine=df_data['Naive forecast: Shortage']

#x=range(len(y_bias_SIR))

plt.plot(x_combine,y_shortage_SIR_combine, color = 'blue',label='SEIRHD',marker='^')
plt.plot(x_combine,y_shortage_Holt_combine, color = 'green',label='Holt',marker='^')
plt.plot(x_combine,y_shortage_Naive_combine, color = 'red',label='Naïve',marker='^')



#chart.set_xticklabels(chart.get_xmajorticklabels(), fontsize = 14)
#chart.set_yticklabels(chart.get_ymajorticklabels(), fontsize = 12)
plt.ylabel("Shortage", size=16)
plt.xlabel("CC deviation", size=16)
#plt.title("Total number of shortage with different protocol coefficient", size=18)
plt.legend(fontsize=13)

plt.subplot(1, 2, 2)
#this part plots different Protocol coefficient and their related real costs
#plt.figure(figsize=(17,8.5))


x=List_CC_dev_step
y_inventory_SIR_combine=df_data['SIR forecast: Inventory level at the end']
y_inventory_Holt_combine=df_data['Holt forecast: Inventory level at the end']
y_inventory_Naive_combine=df_data['Naive forecast: Inventory level at the end']

#x=range(len(y_bias_SIR))

plt.plot(x_combine,y_inventory_SIR_combine, color = 'blue',label='SEIRHD',marker='^')
plt.plot(x_combine,y_inventory_Holt_combine, color = 'green',label='Holt',marker='^')
plt.plot(x_combine,y_inventory_Naive_combine, color = 'red',label='Naïve',marker='^')



#chart.set_xticklabels(chart.get_xmajorticklabels(), fontsize = 14)
#chart.set_yticklabels(chart.get_ymajorticklabels(), fontsize = 12)
plt.ylabel("Shortage", size=16)
plt.xlabel("CC deviation", size=16)


#chart.set_xticklabels(chart.get_xmajorticklabels(), fontsize = 14)
#chart.set_yticklabels(chart.get_ymajorticklabels(), fontsize = 12)
plt.ylabel("Left-over inventory", size=16)
plt.xlabel("CC deviation", size=16)
#plt.title("Real cost with different protocol coefficient", size=18)
plt.legend(fontsize=13)


#plt.savefig('Shortage and left-over inventory for 1000 scenarios over 10 steps for CC deviation',dpi=300)
