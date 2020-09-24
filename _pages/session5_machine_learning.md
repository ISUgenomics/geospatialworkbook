# Distributed Machine Learning Pipeline: NDVI ~ Soil + Weather Dynamics
By Kerrie Geil, USDA-ARS
August 2020
 ---

This tutorial is also provided as a python notebook, which can be fetched by right-clicking, and downloading the linked file.

* [session5\_machine\_learning.ipynb](https://raw.githubusercontent.com/kerriegeil/SCINET-GEOSPATIAL-RESEARCH-WG/master/tutorials/session5_machine_learning.ipynb)

Fetching the python notebook via `curl` or `wget` should also be possible.

```
curl https://raw.githubusercontent.com/kerriegeil/SCINET-GEOSPATIAL-RESEARCH-WG/master/tutorials/session5_machine_learning.ipynb
```
 
This tutorial walks thru a machine learning pipeline. This example excludes the *Extract* component in the often referenced *ETL* (Extract, Transform, Learn) machine learning nomenclature. The overall goal of this analysis is to predict NDVI dynamics from soil and lagged precipitation, temperature, and vapor pressure deficit observations. The brief outline of the tutorial is:

1. Read and transform the NDVI, Soil, and Weather data.
2. Merge the three datasets and add 26 weekly lags of precipitation, vpd, and temperature as features.
3. Shuffle and split data into three groups:
  * 3% for hyperparameter optimization (Group 1)
  * 97 % for final model
    * 77.6% (97% * 80%) for final model training (Group 2)
    * 19.4% (97% * 20%) for final model testing (validation) (Group 3)
3. Optimize the hyperparamters in an XGBoost model (Xtreme Gradient Boosting) using a small subset of the data.
4. Using the "best fit" hyperparameters, train the model 77.6% of the data (Group 2).
5. Validation with the test (hold-out) data (19.4% - Group 3)

## Table of Contents
1. [Build a Distributed Cluster](#build-a-distributed-cluster)
2. [Preprocess, Transform, and Merge the Data](#preprocess-transfor-and-merge-the-data)
3. [Machine Learning: XGBoost Model](#machine-learning-xgboost-model)
4. [Interpreting the Model](#interpreting-the-model)


```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import dask_jobqueue as jq
import dask
from dask import dataframe as ddf
from dask import array as da
import os
from dask.distributed import Client, wait
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.notebook import tqdm
```

## Build a Distributed Cluster<a class="anchor" id="build-a-distributed-cluster"></a>

We will use [dask-jobqueue](https://jobqueue.dask.org/en/latest/) to launch and scale a cluster. For a more detailed example of how this works, please see the other tutorials in the SCINet Geospatial 2020 Workshop. For a quick review, the workflow for defining a cluster and scaling is:<br>
  1. Dask-jobqueue submits jobs to Slurm with an sbatch script
  2. The sbatch scripts define the dask workers with the following info:
    * Partition to launch jobs/workers (```partition```)
    * X number of processes (i.e. dask workers) per sbatch script (```num_processes```).
    * Number of threads/cpus per dask worker (```num_threads_per_process```)
    * Memory allocated per sbatch scipt (```mem```), which is spread evenly between the dask workers.
  3. Scale the cluster to the total # of workers. Needs to be a multiple of num_processes.

In this example, we are defining one process (dask worker) per sbatch script. Each process will have 40 cpus (an entire node). We then scale the cluster to 9 workers, which total 360 threads at 1.15TB of memory.


```python
partition='short,brief-low'
num_processes = 1
num_threads_per_process = 40
mem = 3.2*num_processes*num_threads_per_process
n_cores_per_job = num_processes*num_threads_per_process
container = '/lustre/project/geospatial_tutorials/wg_2020_ws/data_science_im_rs_vSCINetGeoWS_2020.sif'
env = 'py_geo'
clust = jq.SLURMCluster(queue=partition,
                        processes=num_processes,
                        memory=str(mem)+'GB',
                        cores=n_cores_per_job,
                        interface='ib0',
                        local_directory='$TMPDIR',
                        death_timeout=30,
                        python="singularity exec {} /opt/conda/envs/{}/bin/python".format(container,env),
                        walltime='01:30:00',
                        job_extra=["--output=/dev/null","--error=/dev/null"])
cl=Client(clust)
cl
print('The Dask dashboard address is: /user/'+os.environ['USER']+'/proxy/'+cl.dashboard_link.split(':')[-1].split('/')[0]+'/status')
```

**View Cluster Dashboard**
To view the cluster with the dask dashboard interaface click the dask icon on the left menu pane. Copy and paste the above dashboard address (in the form of /user/{User.Name}/proxy/{port#}/status) into the address bar. Then click on the "Workers", "Progress", "Task Stream", and "CPU" to open those tabs. Drag and arrange in convineint layout on right-hand side of the screen. Note these panes should be mostly blank as we have yet to scale the cluster, which is the next step below.

Dask Icon:
<img src="https://avatars3.githubusercontent.com/u/17131925?s=400&v=4" width=20/>

**Scale the Cluster** to 9 workers (40 cpus per worker). This may take 5-20 seconds to complete.



```python
#scale the cluster
n_workers=9
clust.scale(n=n_workers*num_processes)
#Wait for the cluster to load, show progress bar.
with tqdm(total=n_workers*num_processes) as pbar:
    while (((cl.status == "running") and (len(cl.scheduler_info()["workers"]) < n_workers*num_processes))):
        pbar.update(len(cl.scheduler_info()["workers"])-pbar.n)
    pbar.update(len(cl.scheduler_info()["workers"])-pbar.n)
cl
```


```python
#Lets see the workers are running in SLURM
me = os.environ['USER']
!squeue -u $me
```

## Preprocess, Transform, and Merge the Data<a class="anchor" id="preprocess-transfor-and-merge-the-data"></a>

#### Harmonized Landsat Sentinel Data

Link to data repository: https://hls.gsfc.nasa.gov/

**Workflow:**
  1. Data is stored in the Zarr format with three dimensions (x,y,time).
  2. Read with xarray.
  3. Divide the data into chunks. Here we have chunked the data by: x=20 pixels, y=20 pixels, date=Entire Dataset
  4. Subset the data to only included "growing season" months.
  5. Convert the xarray object to a 2-Dimensional dataframe.
  
Notice that the data is not read to memory. The only information stored is the "task graph" and metadata about the final results.


```python
#Read the data with Xarray and rechunk
ndvi = xr.open_zarr('/lustre/project/geospatial_tutorials/wg_2020_ws/data/cper_hls_ndvi.zarr/').chunk({'x':20,'y':20,'date':-1})
ndvi
```


```python
#Select relevant months and then convert to a dataframe
ndvi_df = ndvi.sel(date=ndvi['date.month'].isin([5,6,7,8,9])).to_dask_dataframe()
#Only include reasonable values (.1 < NDVI < 1.0) in the analysis
ndvi_df = ndvi_df[(ndvi_df.ndvi>.1)&(ndvi_df.ndvi<1.)]
print('There are '+f'{len(ndvi_df):,}'+' NDVI observations.')
ndvi_df
```

#### Polaris Soil Hydraulic Data

Paper Describing the Data: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018WR022797<br>
Data Repository Source:  http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/

**Workflow:**
  1. Data is stored in the Zarr format with two dimensions (x,y) and includes 13 variables at 6 depths (78 total). Read with xarray.
  2. Interpolate the data to the same grid as the HLS NDVI data.
  3. Convert the xarray object to a 2-Dimensional Pandas dataframe.


```python
soil = xr.open_zarr('/lustre/project/geospatial_tutorials/wg_2020_ws/data/polaris_soils.zarr/')
#Interpolate to the HLS NDVI grid
soil_df = soil.interp(x=ndvi.x,y=ndvi.y,method='linear').squeeze().to_dataframe().reset_index()
soil_df
```

#### PRISM Precipitation, Tempature, and Vapor Pressure Deficit Data

PRISM Data in a CSV file. Note this data was queried at a single point at the center of CPER.

**Workflow:**
  1. Data is stored in the csv format and includes 7 variables. Read with Pandas using:
    * Skip the 1st 10 rows (PRISM metadata)
    * Convert the time column from a generic object to a date-time object.
  2.  Rename the "Date" to "date" to match HLS NDVI data.
  3. Set the "date" column as the index.
  4. Sort the data into descending.


```python
df_env = pd.read_csv('/lustre/project/geospatial_tutorials/wg_2020_ws/data/PRISM_ppt_tmin_tmean_tmax_tdmean_vpdmin_vpdmax_provisional_4km_20120101_20200101_40.8269_-104.7154.csv',
                      skiprows=10,
                      infer_datetime_format=True,
                      parse_dates = ['Date']).rename(columns={'Date':'date'}).set_index('date').sort_index(ascending=False)
df_env
```

#### Transform Function to Merge NDVI, Soil, and PRISM data.

Here we develop a class to merge the three dataset. Note the most import code is in the ```def transform``` function.


```python
#Costum transformer in the scikit-learn API syntax
class merge_dsets(BaseEstimator,TransformerMixin):
    def __init__(self, df_soil, df_env,lag):
        self.soil = df_soil
        self.env = df_env
        self.lag = lag
        #self.lag_interv = lag_interval
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        df = X.copy()
        df = df.merge(self.soil, on =['x','y'])
        df_env_m = pd.DataFrame()
        for i,d in enumerate(df.date.unique()):
            df_env_temp = df_env[df_env.index<d+pd.Timedelta('1days')].resample('1W-'+d.day_name()[0:3].upper(),
                                                                                label='right').agg({'ppt (mm)':'sum',
                                                                                                    'tmean (degrees C)':'mean',
                                                                                                    'vpdmin (hPa)':'mean',
                                                                                                    'vpdmax (hPa)':'mean'}).sort_index(ascending=False).iloc[0:self.lag].reset_index().reset_index().rename(columns={'index':'week'})
            df_env_temp = df_env_temp.drop(columns='date').melt(id_vars='week')
            df_env_temp['col']='week'+df_env_temp.week.astype(str)+'_'+df_env_temp.variable.str.split(' ',expand=True).values[:,0]
            df_env_temp = df_env_temp.set_index('col').drop(columns=['week','variable']).T
            df_env_temp['date']=d
            df_env_temp = df_env_temp.set_index('date',drop=True)
            df_env_m = df_env_m.append(df_env_temp)
        df = df.merge(df_env_m,left_on='date',right_index=True)
        df['DOY'] = df.date.dt.dayofyear
        return(df.drop(columns=['date','x','y','ndvi']),df[['ndvi']])#.to_dask_array(lengths=True))
```

## Machine Learning: XGBoost Model<a class="anchor" id="machine-learning-xgboost-model"></a>
The "*learn*" portion in the ETL pipeline.


```python
from sklearn.pipeline import Pipeline
import xgboost as xgb
#from dask_ml.xgboost import XGBRegressor as dask_XGBRegressor
from dask_ml.model_selection import train_test_split
from sklearn.metrics import r2_score
from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV as sk_GridSearchCV
import joblib
```

### Hyperparameter Optimization

Shuffle and subset data to a *managable size* (e.g. will fit in memory when running 360 simaltaneous models). We will use a grid-search, combined with 3-fold cross validation, approach to optimize the relevant hyperparameters (see table below).

| Hyperparameter  |        Grid       | n |
|------------|-------------------|---|
|n_estimators | [150, 250, 300, 350] | 4 |
|learning_rate| [0.05, 0.1, 0.2, 0.3]|4|
|max_depth|[5, 7, 9, 11]|4|
|colsample_bytree|[.1, .2, .3]|3|
|gamma|[.05, .1, .2]|3|

A total of 1728 models (4 * 4 * 4 * 3 * 3 * 3) will be fit. The hyperparameters assocated with the best scoring model (highest R2) will be used to train the remianing data.

This search can take ~1-2 hour using 360 cores. To run the hyperparameter gridsearch cross validation, set the ```optimize_hyperparameter``` variable to ```True``` (see two cells below). If you leave as ```False```, we will skip the hyperparameter calculatoins, and just use the hyperparameter values previously calculated.


```python
X_train_hyp, X = train_test_split(ndvi_df,
                                  test_size=0.97,
                                  shuffle=True,
                                  random_state=34)
X_train_hyp,Y_train_hyp = dask.compute(*merge_dsets(df_soil=soil_df,
                                      df_env=df_env,
                                      lag=26).transform(X_train_hyp))
X_train_hyp
```


```python
# Set to True if you want to run the Gridsearch. This can take >1.5 hrs. Therefore, 
# if set to false, the results (best hyperparameters) hardcoded from a previous run 
# of the model
optimize_hyperparameters = False
```


```python
if optimize_hyperparameters:
    #Define the grid - space
    param_dist = {'n_estimators': [150,250,300,350],
        'learning_rate': [0.05, 0.1, 0.2, 0.3],
        'max_depth': [5, 7, 9, 11],
        'colsample_bytree': [.1, .2, .3],
        'gamma': [.05, .1, .2]}
    #Define the XGBoost model
    reg = xgb.XGBRegressor(n_jobs=1,verbosity=3)
    #Setup the GridsearchCV function
    gs = GridSearchCV(reg,param_dist,cv=3,scheduler=cl,refit=False,cache_cv=False)
    #Fit all the models
    gs.fit(X_train_hyp.values,Y_train_hyp.values)
    #Get the best fitting parameters
    df_params = pd.DataFrame(gs.cv_results_)
    best_params = df_params[df_params.mean_test_score==df_params.mean_test_score.max()]
    best_params = best_params.params.values[0]
    print(best_params)
else:
    #Best fit parameters from previous run
    best_params = {'colsample_bytree': 0.2,
                   'gamma': 0.1,
                   'learning_rate': 0.05,
                   'max_depth': 7,
                   'n_estimators': 350}
    print('Using the previously calculated parameters, which are:')
    print(best_params)
```

### Distributed XGBoost Model

 *  Shuffle and split data into "training" (80%) and "testing" (20%). Leave as dask dataframes (data needs to be distributed across all workers), so we will call ```dask.persist``` to trigger the calculation (rather than dask.compute).
 *  Train XGBoost model using the training data.
 *  Model Validation / Accuracy (r2) with "testing" data


```python
# Split the data
X_train, X_test = train_test_split(X,
                                   test_size=0.2,
                                   shuffle=True)
#Merge the weather/soil data and persist the data across the cluster
[X_train,Y_train],[X_test,Y_test] = dask.persist(*[merge_dsets(df_soil=soil_df,df_env=df_env,lag=26).transform(X_train),
                                               merge_dsets(df_soil=soil_df,df_env=df_env,lag=26).transform(X_test)])
wait([X_train,X_test,Y_train,Y_test])
X_train
```


```python
#Setup the Distributed XGBoost model and train it on the "training" data
dtrain = xgb.dask.DaskDMatrix(cl, X_train, Y_train)
reg_b = xgb.dask.train(cl,
                       best_params,
                       dtrain,
                       num_boost_round=125,
                       evals=[(dtrain, 'train')])
print(reg_b)
```


```python
#Get the R2 results for the testing data
dtest = xgb.dask.DaskDMatrix(cl, X_test)
pred = xgb.dask.predict(cl, reg_b['booster'], dtest)
reg_r2 = r2_score(Y_test.ndvi.compute().values,pred)
print("The overall R2 is: "+str(reg_r2))
```


```python
#Big Data Plotting Libraries
import datashader as ds
import holoviews as hv
from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
hv.extension('bokeh')
```


```python
#Plot the results
Y_plotting = Y_test.compute()
Y_plotting['pred']=pred.compute()
Y_plotting
```


```python
#To plot all the points, we need to rasterize the data (aka a 2d histogram)
pts_res = hv.Points(Y_plotting.values,label="")
rasterize(pts_res).redim.range(Count=(10, 2000)).opts(cmap='viridis',
                                                      tools=['hover'],
                                                      xlim=(0.15,.6),
                                                      ylim=(0.15,.6),
                                                      clipping_colors={'min': 'transparent'},
                                                      xlabel='HLS NDVI',
                                                      ylabel='Predicted NDVI',
                                                      logz=True)
```

## Interpreting the Model<a class="anchor" id="interpreting-the-model"></a>

**Use the [SHAP (SHapley Additive exPlanations) package](https://github.com/slundberg/shap)** to interpret the model results and better understand the features "driving" ndvi dynamics.

SHAP Papers: https://www.nature.com/articles/s42256-019-0138-9 and http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

SHAP Blog: https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27


```python
#Standard approaches can render different results
#Show the top 20 most import features as defined by the XGBoost model
xgb.plot_importance(reg_b['booster'],max_num_features=20,importance_type='weight')
xgb.plot_importance(reg_b['booster'],max_num_features=20,importance_type='gain')
xgb.plot_importance(reg_b['booster'],max_num_features=20,importance_type='cover')
```


```python
#Import the SHAP libraries
import shap
import matplotlib.pyplot as plt
shap.initjs()
```


```python
#Split data into better manageable slices
X_shap, _= train_test_split(X_test,test_size=0.95,shuffle=True)
```

**Apply SHAP Model:** Below we split the data by month, and examine the effect of the features on the model (by month).


```python
#Day of Year for each month
months = {'May':[121,152],
          'June':[153,182],
          'July':[183,2013],
          'August':[214,244],
          'September':[245,274]}

#Function for calculating SHAP values. We will map this function across the data on the cluster
def calc_shap_vals(block,explainer):
    if len(block)>0:
        block_vals = explainer.shap_values(block)
        return(block_vals)
    else:
        return(np.empty((0,184)))

#Loop over each month and create plot
explainer = shap.TreeExplainer(reg_b['booster'])
for k in months.keys():
    print(k)
    start = months[k][0]
    end = months[k][1]
    #Select only the data in the month
    X_shap1 = X_shap[(X_shap.DOY>=start)&(X_shap.DOY<=end)].repartition(npartitions=9).persist()
    wait(X_shap1)
    #Compute the SHAP values
    shap_vals = X_shap1.to_dask_array(lengths=True).map_blocks(calc_shap_vals,explainer=explainer,dtype='float32').compute()
    #Show the SHAP summary plots for each month
    print('Using an array of size:' +str(shap_vals.shape))
    plt.title(k)
    shap.summary_plot(shap_vals, X_shap1.compute(),max_display=20,title=k)
```


```python
shap_vals = X_shap.to_dask_array(lengths=True).map_blocks(calc_shap_vals,explainer=explainer,dtype='float32').compute()
shap_vals = shap_vals[~np.isnan(shap_vals).any(axis=1)]
shap.dependence_plot("week0_tmean", shap_vals, X_shap.compute(),interaction_index='DOY')
shap.dependence_plot("week0_ppt", shap_vals, X_shap.compute(),interaction_index='DOY')
shap.dependence_plot("week4_vpdmax", shap_vals, X_shap.compute(),interaction_index='DOY')
```
