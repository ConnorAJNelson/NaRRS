"""
Author: Connor Nelson
Email: connor.nelson.21@ucl.ac.uk
"""

import sys
sys.path.append('/home/cn/NaRRS')
sys.path.append('/home/cn/pysiral')
import os
import time
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from pysiral.waveform import TFMRALeadingEdgeWidth
from Utility.utility_functions import get_sentinel3_product_dataframe_in_interval, get_sentinel3_sral_filename_components, get_S3_data_ds, get_tb_corrected_fpath, get_waveform_features_ds, interpolate_auxiliary_data_to_track, filter_sentinel3_dataset
from Utility.utility_functions  import correct_tbs_for_ow, calculate_gradient_ratio
from Utility.utility_functions  import get_new_mallett_sden, calc_corr_fb, get_alexandrov_cice_density, calc_sit
from Utility.directory_paths import get_local_directory_path
from joblib import Parallel, delayed
from pathlib import Path
from scipy import sparse

########################################################################################################################################################
###################################################### Functions #######################################################################################
########################################################################################################################################################

def make_ridge_pipeline(alpha=1.0, poly_degree=1, solver='auto'):
    """
    Make a sklearn pipeline for ridge regression with polynomial features and standard scaling.
    
    Parameters:
    alpha (float): Regularization strength for ridge regression. Default is 1.0.
    poly_degree (int): Degree of the polynomial features. Default is 1 (no polynomial features).
    solver (str): Solver to use for ridge regression. Default is 'auto'. If the input data is sparse, 'sparse_cg' will be used. If the input data is dense, 'cholesky' will be used.

    Returns:
    sklearn.pipeline.Pipeline: A pipeline object for ridge regression.
    """

    steps = []
    steps.append(('scaler', StandardScaler()))
    steps.append(('poly', PolynomialFeatures(degree=poly_degree)))
    steps.append(('ridge', Ridge(alpha=alpha, solver=solver)))
    return Pipeline(steps)

#=====================================================================================================================================================#

def calculate_leading_edge_rising_width(s3_ds):
    """
    Calculate the leading edge rising width of the Ku-band waveforms. That is, the width between the bins containing 30% and 70% of the first peak power above 15% of the noise floor.
    The method is adapted from taken from pysiral (https://github.com/pysiral/pysiral/blob/main/pysiral/sentinel3/l1_adapter.py).
    
    Parameters:
    s3_ds (xarray.Dataset): Dataset containing the Sentinel-3 data, including the Ku-band waveforms and the operation mode.
    
    Returns:
    leading_edge_width (array): Array of leading edge rising widths (in bins) for each waveform.
    """

    #method 
    wfm_counts = s3_ds.waveform_20_ku.values.astype(np.float64)
    n_records, n_range_bins = wfm_counts.shape
    range_bin_index = np.arange(n_range_bins)

    #set the operation mode
    op_mode =  s3_ds.instr_op_mode_20_ku.values
    op_mode_translator = [0, 1, 1]
    radar_mode = np.array([op_mode_translator[int(val)] for val in op_mode]).astype("int8")
    range_bins = (np.ones((n_records, n_range_bins)) * range_bin_index).astype(np.float32)
    is_valid =  np.ones((n_records), dtype=np.int32)
    
    tfmra_options = {
        "first_maximum_normalized_threshold":[np.nan, 0.15, np.nan],
        "wfm_oversampling_factor": 1,
        "noise_level_range_bin_idx": [0, 5],
        "wfm_smoothing_window_size": [np.nan, 3, np.nan],
        "first_maximum_ignore_leading_bins": 0,
    }

    width = TFMRALeadingEdgeWidth(range_bins, wfm_counts, radar_mode, is_valid, tfmra_options=tfmra_options)
    leading_edge_start_threshold = 0.30
    leading_edge_end_threshold = 0.70
    leading_edge_width, leading_edge_start_bin, leading_edge_start_power, leading_edge_end_bin, leading_edge_end_power = width.get_width_from_thresholds(leading_edge_start_threshold, leading_edge_end_threshold, return_all_values=True)
    return leading_edge_width, leading_edge_start_bin, leading_edge_end_bin

#=====================================================================================================================================================#

def get_footprint_statistic(mwr_xcs, mwr_ycs, var_xcs, var_ycs, var_ds, var_names, min_samples, max_dist, plot_hists=False, statistic='mean', bin_filter=False):
    
    """
    Calculate a specified statistic (e.g., mean, median, standard deviation, count) of variable values within a specified distance of each MWR data point. 
    
    Parameters:
    mwr_xcs (list or array): x coordinates of the MWR data points
    mwr_ycs (list or array): y coordinates of the MWR data points
    var_xcs (list or array): x coordinates of the variable data points
    var_ycs (list or array): y coordinates of the variable data points
    var_ds (xarray.Dataset): Dataset containing the variable data points
    var_names (list): List of variable names to calculate the statistic for
    min_samples (int): Minimum number of samples required to calculate the statistic
    max_dist (float): Maximum distance (in meters) to consider for the variable data points
    plot_hists (bool): Whether to plot histograms of the variable values for each MWR data point
    statistic (str): Statistic to calculate ('mean', 'median', 'std', 'count')
    bin_filter (bool): Whether to apply a bin filter to the variable values. If True, values that are outside of 3 IQRs outside the 25th and 75th percentiles will be set to NaN before calculating the statistic.
    """
    
    tree = KDTree(np.column_stack((var_xcs, var_ycs)))
    values_stats = {var_name: np.full(len(mwr_xcs), np.nan) for var_name in var_names}

    stat_functions = {
        'mean': np.mean,
        'median': np.median,
        'std': np.std,
        'count': lambda x: np.sum(~np.isnan(x))
    }
    
    def apply_statistic(values, stat_func, min_samples):
        return stat_func(values) if values.size >= min_samples else np.nan

    for i, (xc, yc) in enumerate(zip(mwr_xcs, mwr_ycs)):
        idx = tree.query_ball_point([xc, yc], max_dist)
        valid_indices = np.ones(len(idx), dtype=bool)
        
        #get all the nan indices for each variable, including the use of an IQR filter if chosen
        for var_name in var_names:
            values = var_ds[var_name].values[idx]
            if bin_filter:
                values[~iqr_filter(values, 3)] = np.nan
            valid_indices &= ~np.isnan(values)

        for var_name in var_names:
            values = var_ds[var_name].values[idx][valid_indices]
            stat_func = stat_functions.get(statistic)
            values_stats[var_name][i] = apply_statistic(values, stat_func, min_samples)
            if plot_hists and statistic != 'count':
                plot_variable_histogram(values, var_ds[var_name].values)

    return values_stats

#=====================================================================================================================================================#

def plot_variable_histogram(variable_values, variable_name):
    """
    Plot a histogram of the variable values.
    
    Parameters:
    variable_values (array-like): Values of the variable to plot the histogram for.
    variable_name (str): Name of the variable to use in the axis label and text box.
    
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(variable_values, bins=40, color='blue', alpha=0.5)
    # ax.set_xlabel(f'{variable_name} (units)')
    ax.set_ylabel('Frequency')
    ax.axvline(np.nanmean(variable_values), color='red', linestyle='dashed', linewidth=1)
    ax.axvline(np.nanmedian(variable_values), color='orange', linestyle='dashed', linewidth=1)
    ax.text(0.5, 0.9, f'Mean: {np.nanmean(variable_values):.2f}\nStd: {np.nanstd(variable_values):.2f}\nMedian: {np.nanmedian(variable_values):.2f}\nCount:{sum(~np.isnan(variable_values))}', transform=ax.transAxes, verticalalignment='top')
    plt.show()

#=====================================================================================================================================================#

def iqr_filter(data, factor=3):
    """
    Get an IQR filter mask for the data. Values that are outside of factor*IQR outside the 25th and 75th percentiles will be considered outliers and will be marked as False in the mask.
    
    Parameters:
    data (array-like): Input data to filter.
    factor (float): The factor to multiply the IQR by to determine the outlier thresholds.
    """
    
    lower_bound, upper_bound = get_iqr_filter_bounds(data, factor=factor)
    return (data >= lower_bound) & (data <= upper_bound)

#=====================================================================================================================================================#

def get_iqr_filter_bounds(data, factor=3):
    """
    Get the lower and upper bounds for an IQR filter. 
    The lower bound is calculated as Q1 - factor*IQR and the upper bound is calculated as Q3 + factor*IQR.
    
    Parameters:
    data (array-like): Input data to calculate the IQR bounds for.
    factor (float): The factor to multiply the IQR by to determine the outlier thresholds.
    """
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor*iqr
    upper_bound = q3 + factor*iqr
    return lower_bound, upper_bound


##############################################################################################################################################################
###################################################################  Main  ###################################################################################
##############################################################################################################################################################

#Set working directory
os.chdir('/home/cn/NaRRS')

#Set hemisphere to process
hemisphere = 'nh'

#Set the min/max latitude for the hemisphere
min_lat = 50 if hemisphere == 'nh' else -40

season_start_years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
season_files = []
for year in season_start_years:
    season_files_df = get_sentinel3_product_dataframe_in_interval(f'{year}-10-01', f'{year+1}-05-01', hemisphere, lat_limit=min_lat, enhanced_measurements=True)
    season_files.extend(season_files_df['product_path'].tolist())

all_files = sorted(list(set(season_files)))

#Load the databse of producted that do not have atmospherically corrected brightness temperatures
#very rarely (< 0.001% of files) produce no RTTOV corrections. 
#inspection of these files shows they tend not to be a fault of RTTOV but rather bad quality (flagged)/missing input data or no valid ocean data (all Tbs obvs in the file corresponds to land)
#using the saved list of failed RTTOV runs, we'll filter these files out of the to-load list
failed_rttov = pd.read_csv(f'Data/rttov_{hemisphere}_failed_products.txt')
all_files = [fpath for fpath in all_files if fpath.split('/')[-1] not in failed_rttov['Product'].tolist()]

print(f"Total number of files to process: {len(all_files)}")

#open the overlap dataset
bin_diameter = 23.5e3
overlap_ds = xr.open_dataset(f'Data/sentinel3_oib_xo_unified.nc')
#Filter out samples with ice concentration below 75% 
min_sic = 75
overlap_ds = overlap_ds.where(overlap_ds['ice_conc_01'] >= min_sic, drop=True)

#Get our fitted model
coord_subset = [f'time_01', f'xc_01', f'yc_01', f'lon_01', f'lat_01']

FEATURE_SUBSET = ['gradient_ratio_corrected_01',
                    'leading_edge_width_mean_01',
                    'trailing_edge_decay_mean_01',
                    ]

waveform_feature_counts = [feature.replace('_mean_01', '_count_01') for feature in FEATURE_SUBSET if 'mean' in feature]

df = overlap_ds.to_dataframe().reset_index()[coord_subset+FEATURE_SUBSET+waveform_feature_counts+['mean_snow_depth_01', 'sitype_osisaf_01']]
df = df.dropna().reset_index(drop=True)

#Set a minimum number of samples for our waveform features
min_wf_count = 15 if hemisphere == 'nh' else 10

for feature in waveform_feature_counts:
    df = df.where(df[feature] >= min_wf_count).dropna().reset_index(drop=True)

X_oib = df[FEATURE_SUBSET]

y_oib = df['mean_snow_depth_01']

alpha = 1
poly_degree = 1

if not sparse.issparse(X_oib):
    print('Solver = Cholesky')
else:
    print('Solver = Sparse CG')


model = make_ridge_pipeline(alpha=alpha, poly_degree=poly_degree, solver='auto')
model.fit(X_oib, y_oib)

#print the mean and std used to scale each feature
print('Feature scaling parameters:')
scaler_object = model.named_steps['scaler']
for i, f in enumerate(X_oib.columns.tolist()):
    print(f'{f}:\nMean: {scaler_object.mean_[i]:.3f}\n STD: {scaler_object.scale_[i]:.3f}')

#print the model coefficients
print('Model coefficients:')
for f, w in zip(X_oib.columns.tolist(), model.named_steps['ridge'].coef_[1:]):
    print(f'{f}: {w.round(3)}')

#print the model intercept
print(f'Intercept: {model.named_steps["ridge"].intercept_.round(3)}')

#print the equation
equation = 'y = '
for f, w in zip(X_oib.columns.tolist(), model.named_steps['ridge'].coef_[1:]):
    equation += f'{w.round(3)} * {f} + '
equation += f'{model.named_steps["ridge"].intercept_.round(3)}'
print(equation)

feature_bounds = {}
for feature in FEATURE_SUBSET:
        lower_bound, upper_bound = get_iqr_filter_bounds(X_oib[feature].values, factor=3)
        feature_bounds[feature] = (lower_bound, upper_bound)
    
#use the every combination of the bounds to predict a value of snow depth
feature_combinations = []
for feature in FEATURE_SUBSET:
    lower_bound, upper_bound = feature_bounds[feature]
    feature_combinations.append(np.linspace(lower_bound, upper_bound, 2))

feature_combinations = np.meshgrid(*feature_combinations)
feature_combinations = np.stack([feature_combinations[i].flatten() for i in range(len(feature_combinations))]).T

predicted_snow_depth = model.predict(feature_combinations)

#get the max possible snow depth
MAX_SNOW_DEPTH = np.max(predicted_snow_depth).round(3)
print(f'Max possible predicted snow depth: {MAX_SNOW_DEPTH}')

def rollout_main(pipeline, s3_dirpath, hemisphere, bin_diameter, min_sic, min_wf_count, overwrite=False):
    print(f"Processing {s3_dirpath}")
    product_metadata = get_sentinel3_sral_filename_components(s3_dirpath.split('/')[-1])
    output_path = Path(get_local_directory_path('NaRRS_data')) / product_metadata['mission_id'] / product_metadata['cycle_number'] / s3_dirpath.split('/')[-1]
    output_path = output_path / ('NaRRS_Sentinel3_snow_depth_and_sea_ice_thickness.nc')
    
    #wait a random amount of time (up to 2 seconds) to avoid multiple processes trying to create the same directory
    time.sleep(np.random.uniform(0, 2))
    os.makedirs(output_path.parent, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"File already exists: {output_path}")
        return
    
    hz_1_names = ['time_01', 'lat_01', 'lon_01', 'tb_238_01', 'tb_238_quality_flag_01', 'tb_238_std_01', 'tb_365_01', 'tb_365_quality_flag_01', 'tb_365_std_01']
    hz_20_names = ['time_20_ku', 'lat_20_ku', 'lon_20_ku', 'surf_type_class_20_ku', 'freeboard_20_ku', 'waveform_20_ku','instr_op_mode_20_ku']

    if hemisphere == 'nh':
        ease2_epsg = 6931
    elif hemisphere == 'sh':
        ease2_epsg = 6932

    s3_ds = get_S3_data_ds(s3_dirpath, vars_01=hz_1_names, vars_20_ku=hz_20_names, decode_times=True, coord_transform_epsg=ease2_epsg)
    s3_ds_raw_times = get_S3_data_ds(s3_dirpath, vars_01=['time_01'], vars_20_ku=['time_20_ku'], decode_times=False)
    s3_ds['time_01_raw'] = (('time_01'), s3_ds_raw_times['time_01'].values)
    s3_ds['time_01_raw'].attrs = s3_ds_raw_times['time_01'].attrs
    s3_ds['time_20_ku_raw'] = (('time_20_ku'), s3_ds_raw_times['time_20_ku'].values)
    s3_ds['time_20_ku_raw'].attrs = s3_ds_raw_times['time_20_ku'].attrs

    #Interpolate auxiliary data to the Sentinel-3 track
    handle_missing_sic_ds = 'temporal_interp'
    handle_missing_edge_ds = 'use_sic'
    s3_ds = interpolate_auxiliary_data_to_track(s3_ds, hemisphere, distance_to_ice_edge=False, handle_missing_sic_ds=handle_missing_sic_ds, handle_missing_edge_ds=handle_missing_edge_ds)

    #Get the sentinel-3 corrected brightness temperatures
    era5_source = 'gcp'
    corrected_tb_fpath = get_tb_corrected_fpath(s3_dirpath.split('/')[-1], hemisphere, era5_source)
    corrected_tb_ds = xr.open_dataset(corrected_tb_fpath)

    #Merge the corrected brightness temperatures with the Sentinel-3 data
    #conflicts have been noted to arise because of extra decimal places in the original brightness temperature vars
    #therefore, we will override conflicts
    s3_ds = xr.merge([s3_ds, corrected_tb_ds], join='inner', compat='override')
    
    #Define some filters for the Sentinel-3 data
    lat_limit = 40 if hemisphere == 'nh' else -40
    lat_range = [90, lat_limit] if hemisphere == 'nh' else [lat_limit, -90]
    min_dist_coast = 25e3
    nsidc_region = np.arange(1,30,1)
    min_sic = min_sic
    max_t2m = 273.15 + 2

    #Filter the Sentinel-3 data
    #let's do the filtering separately for the 1hz and 20hz data as we can go closer the coast with the 20hz data
    s3_ds_1hz = s3_ds.drop_dims('time_20_ku')
    s3_ds_20_ku = s3_ds.drop_dims('time_01')

    lat_limit = 40 if hemisphere == 'nh' else -40
    lat_range = [90, lat_limit] if hemisphere == 'nh' else [lat_limit, -90]

    s3_ds_1hz = filter_sentinel3_dataset(s3_ds_1hz, measurement_frequency = ['01'], lat_range=lat_range, min_dist_coast=min_dist_coast, nsidc_region=nsidc_region, min_sic=min_sic, max_t2m=max_t2m, drop=True)
    s3_ds_20_ku = filter_sentinel3_dataset(s3_ds_20_ku, measurement_frequency = ['20_ku'], lat_range=lat_range, nsidc_region=nsidc_region, min_sic=min_sic, max_t2m=max_t2m, drop=False)
    s3_ds_20_ku = s3_ds_20_ku.dropna(dim='time_20_ku', how='all', subset=['waveform_20_ku'])
    s3_ds = xr.merge([s3_ds_1hz, s3_ds_20_ku])

    #We also want to set to nan any freeeboard values with a leading edge rise width of > 3 bins (Tilling et al. 2018)
    #This width is calcuated between 30% and 70% of the first peak power above 15% of the noise floor (slight adaption as Tilling uses 20% of the max peak)
    leading_edge_width, leading_edge_start_bin, leading_edge_end_bin = calculate_leading_edge_rising_width(s3_ds)
    # As the above results are in fractions of a bin, we will round down to get the whole bin number and recalculate the width
    leading_edge_start_bin = np.floor(leading_edge_start_bin)
    leading_edge_end_bin = np.floor(leading_edge_end_bin)
    leading_edge_width = leading_edge_end_bin - leading_edge_start_bin

    #add the leading edge rising width to the dataset
    s3_ds['leading_edge_rising_width_20_ku'] = (('time_20_ku'), leading_edge_width)
    s3_ds['leading_edge_rising_width_20_ku'].attrs = {'units':'bins', 'long_name':'Leading Edge Rising Width', 'description':'Width of the leading edge between the bins containing 30% and 70% of the first peak power above 15% of the noise floor'}
    
    #set the freeboard to nan where the leading edge width is > 3 bins
    max_bins = 3
    s3_ds['freeboard_20_ku'] = s3_ds['freeboard_20_ku'].where(s3_ds['leading_edge_rising_width_20_ku'] <= max_bins)

    #we can drop the waveforms from the dataset now
    s3_ds = s3_ds.drop_vars('waveform_20_ku').drop_dims('wf_sample_ind')

    if s3_ds.time_01.size == 0:
        print(f"No valid data points for {s3_dirpath}")
        #remove the created dir 
        os.rmdir(output_path.parent)
        return
    
    #calulate the concentration-corrrected brightness temperatures
    s3_ds = correct_tbs_for_ow(s3_ds, hemisphere)

    #calculate the gradient ratio
    s3_ds['gradient_ratio_corrected_01'] = calculate_gradient_ratio(s3_ds['tb_238_corrected_01'], s3_ds['tb_365_corrected_01'])
    s3_ds['gradient_ratio_corrected_01'].attrs = {'units':'K', 'long_name':'37/24 Gradient Ratio corrected for ow influence'}

    max_dist = bin_diameter/2
    bin_filter = True
    
    #load the pysiral-derived waveform features
    try:
        waveform_df = get_waveform_features_ds(s3_dirpath, hemisphere, si_only=True).to_dataframe().reset_index()
    except Exception as e:
        print(f"An error occurred: {e}")
        #check if the file was known to have failed the pysiral conversion
        failed_products = pd.read_csv(f'Data/no_waveform_ncs_{hemisphere}.csv')
        if s3_dirpath in failed_products['product_path'].tolist():
            print(f"Waveform file not found for {s3_dirpath}. It is known to have failed the pysiral conversion")
            os.rmdir(output_path.parent)
            return
        else:
            print(f"Error retrieving waveform features for {s3_dirpath} and product not known to have failed the pysiral conversion")
            raise e
        
    wf_vars_of_interest = ['leading_edge_width',  'trailing_edge_decay']
    wf_filter_vars = [ 'trailing_edge_width', 'trailing_edge_decay_fit_quality', 'first_maximum_power']
    wf_coords = ['time_20_ku', 'xc_20_ku', 'yc_20_ku', 'lon_20_ku', 'lat_20_ku']
    waveform_df = waveform_df[wf_coords+wf_vars_of_interest+wf_filter_vars].dropna().reset_index(drop=True)

    #rename the waveform var to match the S3 naming convention
    for k, wf_var in enumerate(wf_vars_of_interest):
        waveform_df = waveform_df.rename(columns={wf_var: wf_var+'_20_ku'})
        wf_vars_of_interest[k] = wf_var+'_20_ku'

    #Sometimes we get trailing edge decays that are extremly large (1e6), leading to a trailing edge width of 1 bin.
    #These appear to be misclassified waveforms (ie not sea ice) or contaminated waveforms. 
    #we willtherefore filter out values where the trailing edge width == 1 as quality control.
    #We will also ensure the trailing edge decay fit is good quality (fit quality > 0.8).
    #and, ensure the first maximum power is the overall maximum power.

    waveform_df = waveform_df[(waveform_df['trailing_edge_width'] > 1) & 
                              (waveform_df['trailing_edge_decay_fit_quality'] > 0.8) & 
                              (waveform_df['first_maximum_power'] == 1)]

    #ok, we don't need the filter columns anymore
    waveform_df = waveform_df.drop(columns=wf_filter_vars).reset_index(drop=True)

    # Calculate the mean value within a r km radius of each S3 data point
    s3_wf_vars_df = s3_ds.drop_dims('time_01').to_dataframe().reset_index()[['lon_20_ku', 'lat_20_ku', 'freeboard_20_ku']]

    #merge the two waveform dfs
    #first let's ensure that the coordinates in both dataframes are rounded to 6 dp so they can be accurately matched.
    waveform_df['lon_20_ku'] = waveform_df['lon_20_ku'].round(6)
    waveform_df['lat_20_ku'] = waveform_df['lat_20_ku'].round(6)
    s3_wf_vars_df['lon_20_ku'] = s3_wf_vars_df['lon_20_ku'].round(6)
    s3_wf_vars_df['lat_20_ku'] = s3_wf_vars_df['lat_20_ku'].round(6)
    
    #merge the two dataframes on lon and lat
    waveform_df = waveform_df.merge(s3_wf_vars_df, on=['lon_20_ku', 'lat_20_ku'], how='inner')
    waveform_df = waveform_df.dropna(subset=wf_vars_of_interest).reset_index(drop=True)

    min_ku_samples = min_wf_count

    for stat in ['mean', 'std', 'count']:
        footprint_statistic = get_footprint_statistic(s3_ds['xc_01'].values, s3_ds['yc_01'].values, 
                                                            waveform_df['xc_20_ku'].values, waveform_df['yc_20_ku'].values,
                                                            waveform_df, wf_vars_of_interest, min_ku_samples, 
                                                            max_dist,  statistic=stat, 
                                                            bin_filter=bin_filter, plot_hists=False,)
        for var_name in wf_vars_of_interest:
            s3_ds[f"{var_name.split('_20_ku')[0]}_{stat}_01"] = (('time_01'), footprint_statistic[var_name])
            s3_ds[f"{var_name.split('_20_ku')[0]}_{stat}_01"].attrs = {'long_name':var_name.replace('_20_ku', ''), 'description': f"{stat} Ku-band {var_name.split('_20_ku')[0].replace('_', ' ')} within {max_dist/1e3} km of S3 1Hz data point"}

        #do freeboard separately as it is not a snow depth predictor
        # we want the footprint mean rfb for later use but don't want it to cause unnecessary filtering of good predictors
        footprint_statistic = get_footprint_statistic(s3_ds['xc_01'].values, s3_ds['yc_01'].values,
                                                            waveform_df['xc_20_ku'].values, waveform_df['yc_20_ku'].values,
                                                            waveform_df, ['freeboard_20_ku'], min_ku_samples,
                                                            max_dist, statistic=stat,
                                                            bin_filter=False, plot_hists=False, )
        s3_ds[f"freeboard_{stat}_01"] = (('time_01'), footprint_statistic['freeboard_20_ku'])
        s3_ds[f"freeboard_{stat}_01"].attrs = {'long_name':'Radar Freeboard', 'description': f"{stat} Ku-band Freeboard within {max_dist/1e3} km of S3 1Hz data point"}

    #let's get the predicted snow depth for each data point using the fitted model
    X_s3 = (s3_ds[FEATURE_SUBSET]
            .to_dataframe()[FEATURE_SUBSET]
            .reset_index(drop=True))
    
    #but first, apply an IQR filter to the data relative to the features the model was fitted on to remove extreme extrapolations.
    for feature in FEATURE_SUBSET:
        lower_bound, upper_bound = get_iqr_filter_bounds(X_oib[feature].values, factor=3)
        # print(f'{feature} IQR bounds: {lower_bound} -> {upper_bound}')
        X_s3.loc[~((X_s3[feature] >= lower_bound) & (X_s3[feature] <= upper_bound))] = np.nan
        
    #drop nans
    filtered_X_s3 = X_s3.dropna()

    if len(filtered_X_s3) == 0:
        print(f"No valid data points for rollout on {s3_dirpath}")
        os.rmdir(output_path.parent)
        return

    predicted_snow_depth = pipeline.predict(filtered_X_s3).round(3)
    filtered_X_s3['snow_depth'] = predicted_snow_depth
    X_s3 = X_s3.merge(filtered_X_s3[['snow_depth']], left_index=True, right_index=True, how='left')
    
    #make a negative snow depth flag
    X_s3['negative_snow_depth_flag'] = (X_s3['snow_depth'] < 0).astype(int)
    #set negative snow depth flag to nan where snow depth is nan
    X_s3.loc[np.isnan(X_s3['snow_depth']), 'negative_snow_depth_flag'] = np.nan
    #set negative snow depths to nan
    X_s3.loc[X_s3['snow_depth'] < 0, 'snow_depth'] = np.nan

    # check again if we have valid data points
    if np.sum(~np.isnan(X_s3['snow_depth'])) == 0:
        print(f"No valid snow depth data points for {s3_dirpath}")
        os.rmdir(output_path.parent)
        return

    #Add the predicted snow depth to the dataset
    s3_ds['snow_depth_01'] = (('time_01'), X_s3['snow_depth'].values)
    s3_ds['snow_depth_01'].attrs = {'units':'m', 'long_name':'snow depth', 'description':f'NaRRS predicted snow depth using a ridge regression model trained on OIB data. Features used in the model are the MWR gradient ratio GR(36.5/23.8), and leading edge width (LEW) and trailing edge decay (TED) of the Ku-band waveform'}
    s3_ds['negative_snow_depth_flag_01'] = (('time_01'), X_s3['negative_snow_depth_flag'].values)
    s3_ds['negative_snow_depth_flag_01'].attrs = {'long_name':'Negative Snow Depth Flag', 'description':'Flag indicating whether the predicted snow depth was negative (1) or not (0). Negative snow depths were subsequently set to NaN'}
    
    #create the final dataset with the desired variables and attributes
    output_vars = ['time_01', 'lon_01', 'lat_01', 'xc_01', 'yc_01', 'ice_conc_01', 'sitype_osisaf_01', 
                   'tb_238_corrected_01', 'tb_365_corrected_01','gradient_ratio_corrected_01', 
                   'leading_edge_width_mean_01', 'leading_edge_width_std_01', 'leading_edge_width_count_01',
                   'trailing_edge_decay_mean_01', 'trailing_edge_decay_std_01', 'trailing_edge_decay_count_01',
                   'freeboard_mean_01', 'freeboard_std_01', 'freeboard_count_01',
                    'snow_depth_01', 'negative_snow_depth_flag_01' ]
    
    #coordinates
    s3_ds['xc_01'].attrs = {'units':'m', 'long_name':'EASE-2 x Coordinate', 'description':f'EASE-2 x coordinate in meters (EPSG:{ease2_epsg})'}
    s3_ds['yc_01'].attrs = {'units':'m', 'long_name':'EASE-2 y Coordinate', 'description':f'EASE-2 y coordinate in meters (EPSG:{ease2_epsg})'}
    
    #auxiliary variables
    s3_ds['ice_conc_01'].attrs = {'units':'%', 'long_name':'Fully filtered concentration of sea ice using atmospheric correction of brightness temperatures and open water filters', \
        'standard_name':'sea_ice_area_fraction', 'comment':'Sea ice concentration from the OSI SAF ICDR/CDR (OSI-430-a/OSI-450-a) dataset interpolated to the Sentinel-3 track using bilinear interpolation.'}
    s3_ds['sitype_osisaf_01'].attrs['comment'] = 'Sea ice type flag from the OSI SAF OSI-403-d product interpolated to the Sentinel-3 track using nearest neighbour interpolation.'
    
    #MWR variables
    s3_ds['tb_238_corrected_01'].attrs = {'units':'K', 'long_name':'Corrected Brightness Temperature at 23.8 GHz', 'description':'Brightness temperature at 23.8 GHz corrected for the atmospheric and surface water radiative contributions, thereby approximating the Tb at the top of the sea ice/snow surface'}
    s3_ds['tb_365_corrected_01'].attrs = {'units':'K', 'long_name':'Corrected Brightness Temperature at 36.5 GHz', 'description':'Brightness temperature at 36.5 GHz corrected for the atmospheric and surface water radiative contributions, thereby approximating the Tb at the top of the sea ice/snow surface'}
    s3_ds['gradient_ratio_corrected_01'].attrs = {'units':'', 'long_name':'36.5/23.8 GHz Gradient Ratio', 'description':'Gradient ratio calculated as (Tb_365 - Tb_238) / (Tb_365 + Tb_238), where Tb are the brightness temperatures corrected for atmospheric and surface water influences (tb_238_corrected_01 and tb_365_corrected_01)'}
    
    #SRAL variables
    s3_ds['trailing_edge_decay_mean_01'].attrs = {'units':'', 'long_name':'Trailing Edge Decay', 'description':f'Mean trailing edge decay (TED) of the Ku-band waveforms within {max_dist/1e3} km of S3 (lon_01, lat_01) data point. TED is defined as The decay exponent from a inverse powerlaw function fit to the peak power and the lower envelope of the trailing edge (Müller et al., 2023; https://doi.org/10.5194/tc-17-809-2023)'}
    s3_ds['trailing_edge_decay_std_01'].attrs = {'units':'', 'long_name':'Trailing Edge Decay Standard Deviation', 'description':f'Standard deviation of the trailing edge decay (TED) of the Ku-band waveforms within {max_dist/1e3} km of S3 (lon_01, lat_01) data point.'}
    s3_ds['trailing_edge_decay_count_01'].attrs = {'long_name':'Trailing Edge Decay Count', 'description':f'Number of valid trailing edge decay (TED) data points within {max_dist/1e3} km of S3 (lon_01, lat_01) data used to compute the mean/std statistics.'}
    
    s3_ds['leading_edge_width_mean_01'].attrs = {'units':'bins', 'long_name':'Leading Edge Width', 'description':f"Mean leading edge width (LEW) of the Ku-band waveforms within {max_dist/1e3} km of S3 (lon_01, lat_01) data point. LEW is defined as the width between the 5 % and 95 % power levels of the waveform's rising edge (Hendricks & Paul, 2023; https://doi.org/10.5281/zenodo.10044554)"}
    s3_ds['leading_edge_width_std_01'].attrs = {'units':'bins', 'long_name':'Leading Edge Width Standard Deviation', 'description':f'Standard deviation of the leading edge width (LEW) of the Ku-band waveforms within {max_dist/1e3} km of S3 (lon_01, lat_01) data point.'}
    s3_ds['leading_edge_width_count_01'].attrs = {'long_name':'Leading Edge Width Count', 'description':f'Number of valid leading edge width (LEW) data points within {max_dist/1e3} km of S3 (lon_01, lat_01) data used to compute the mean/std statistics.'}
    
    s3_ds['freeboard_mean_01'].attrs = {'units':'m', 'long_name':'Radar Freeboard', 'description':f'Mean radar freeboard within {max_dist/1e3} km of S3 (lon_01, lat_01) data point. The radar freeboard were obtained from the Sentinel-3 Sea Ice Thematic BC005 freeboard_20_ku variable.'}
    s3_ds['freeboard_std_01'].attrs = {'units':'m', 'long_name':'Radar Freeboard Standard Deviation', 'description':f'Standard deviation of the radar freeboard within {max_dist/1e3} km of S3 (lon_01, lat_01) data point.'}
    s3_ds['freeboard_count_01'].attrs = {'long_name':'Radar Freeboard Count', 'description':f'Number of valid radar freeboard data points within {max_dist/1e3} km of S3 (lon_01, lat_01) data used to compute the mean/std statistics.'}   
    
    #calculate SIT using the average radar freeboard in each footprint and our predicted snow depths
    if hemisphere == 'nh':
        #to do this, we require the snow density and cice density
        s3_ds['snow_density_01'] = (('time_01'), [get_new_mallett_sden(datetime) for datetime in pd.to_datetime(s3_ds['time_01'].values)])
        s3_ds['snow_density_01'].attrs = {'units':'kg/m^3', 'long_name':'Snow Density', 'description':'Density of snow from the expression in Mallett (2025; https://doi.org/10.1017/jog.2025.5)'}

        s3_ds['cice_density_01'] = (('time_01'), [get_alexandrov_cice_density(ice_type) for ice_type in s3_ds['sitype_osisaf_01'].values])
        s3_ds['cice_density_01'].attrs = {'units':'kg/m^3', 'long_name':'Sea Ice Density', 'description':'Density of sea ice based on the ice type flag from the OSI SAF OSI-403-d product (10.15770/EUM_SAF_OSI_NRT_2006) and the density classification in Alexandrov et al. 2010 (https://doi.org/10.5194/tc-4-373-2010)'}
        
        #calculate the corrected freeboard
        s3_ds['corrected_freeboard_01'] = (('time_01'), calc_corr_fb(s3_ds['freeboard_mean_01'].values, s3_ds['snow_depth_01'].values, s3_ds['snow_density_01'].values))
        #ensure that corrected freeboards are between -0.3 and 3 m (tilling et al. 2018)
        s3_ds['corrected_freeboard_01'] = s3_ds['corrected_freeboard_01'].where((s3_ds['corrected_freeboard_01'] >= -0.3) & (s3_ds['corrected_freeboard_01'] <= 3))
        s3_ds['corrected_freeboard_01'].attrs = {'units':'m', 'long_name':'Corrected Freeboard', 'description':'Freeboard (freeboard_mean_01) corrected for reduction of radar propagation speed through the snowpack, giving the approximate sea ice freeboard. Values are filtered to be in the range -0.3 and 3m as per Tilling et al. 2018 (https://doi.org/10.1016/j.asr.2017.10.051)'} 
        
        #calculate the sea ice thickness
        s3_ds['sea_ice_thickness_01'] = (('time_01'), calc_sit(s3_ds['corrected_freeboard_01'].values, s3_ds['snow_depth_01'].values, s3_ds['snow_density_01'].values, s3_ds['cice_density_01'].values))
        s3_ds['sea_ice_thickness_01'].attrs = {'units':'m', 'long_name':'Sea Ice Thickness', 'description':'Sea ice thickness calculated assuming hydrostatic equilibrium, using the corrected freeboard (corrected_freeboard_01), NaRRS snow depth (snow_depth_01), snow density (snow_density_01) and sea ice density (cice_density_01), and a sea water density of 1023.9 kg/m^3.'}

        #make a negative SIT flag
        s3_ds['negative_sit_flag_01'] = (('time_01'), xr.where(s3_ds['sea_ice_thickness_01'].values < 0, 1, 0))
        s3_ds['negative_sit_flag_01'].attrs = {'long_name':'Negative Sea Ice Thickness Flag', 'description':'Flag indicating if the sea ice thickness is negative: 0 = not negative, 1 = negative'}

        #make an extreme SIT flag (0 = not extreme, 1 = extreme, where extreme is defined as > 10.5m) (AWI)
        s3_ds['extreme_sit_flag_01'] = (('time_01'), xr.where(s3_ds['sea_ice_thickness_01'].values > 10.5, 1, 0))
        s3_ds['extreme_sit_flag_01'].attrs = {'long_name':'Extreme Sea Ice Thickness Flag', 'description':'Flag indicating if the sea ice thickness is extreme: 0 = in normal range, 1 = extreme. Extreme is defined as > 10.5 m according to AWI SIT convention (Hendricks & Paul, 2023; https://doi.org/10.5281/zenodo.10044554)'}  

        output_vars = output_vars + ['snow_density_01', 'cice_density_01',
                                     'corrected_freeboard_01', 'sea_ice_thickness_01', 
                                     'negative_sit_flag_01', 'extreme_sit_flag_01',]


    #revert the time variables back to the original format
    s3_ds['time_01'] = s3_ds['time_01_raw']
    s3_ds['time_20_ku'] = s3_ds['time_20_ku_raw']

    output_s3_ds = s3_ds[output_vars]
    output_s3_ds.attrs = {'author': 'Connor Nelson',
                          'institution': 'UCL',
                          'contact': 'connor.nelson.21@ucl.ac.uk',
                          'description': 'Snow depth and sea ice thickness retrieved from Sentinel-3 using the NaRRS method, along with the derivation variables.',
                          'version': '1.0',
                          'processed_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          'note': 'This dataset contains variables (time_01, lon_01, lat_01, xc_01, yc_01) and modified variables (tb_238_01 --> tb_238_corrected_01, tb_365_01 --> tb_365_corrected_01, radar_freeboard_01 --> freeboard_mean_01/corrected_freeboard_01) from the Copernicus Sentinel-3 L2 Sea Ice Thematic BC005 product (Aublanc et al. 2025; https://doi.org/10.1038/s41597-025-04956-3), see https://doi.org/10.57780/s3d-6c5ea4 for access information. The gradient ratio (gradient_ratio_corrected_01) was computed using the modified brightness temperatures, and the leading edge width (leading_edge_width_mean_01) and trailing edge decay (trailing_edge_decay_mean_01) were computed using Ku-band waveforms (waveform_20_ku in the Sentinel-3 L2 Sea Ice Thematic BC005 product) processed with the pysiral toolbox (https://doi.org/10.5281/zenodo.10727996).'
    }
    
    #we were getting a serialization error when trying to save the dataset with the lat and lon variables, so we will encode they are floats
    for coord in ['lat_01', 'lon_01', 'xc_01', 'yc_01']:
        output_s3_ds[coord] = output_s3_ds[coord].astype(float)
        
    #save the dataset to netcdf with compression
    output_encoding = {var: {'zlib': True, 'complevel': 4} for var in output_s3_ds.data_vars}
    output_s3_ds.to_netcdf(output_path, encoding=output_encoding)
    print(f"Processed {s3_dirpath}")
    return

overwrite = True
njobs=24
Parallel(n_jobs=njobs)(delayed(rollout_main)(model, s3_dirpath, hemisphere, bin_diameter, min_sic, min_wf_count, overwrite) for s3_dirpath in all_files)
print('Finished :)')
