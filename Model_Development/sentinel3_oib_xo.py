#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: connornelson
"""

import os
import sys
sys.path.append('/home/cn/NaRRS')
import pyproj
import time
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
import verde as vd
from glob import glob
from cftime import num2pydate
from Utility.directory_paths import get_local_directory_path
from Utility.utility_functions import get_sentinel3_sral_filename_components
from Utility.utility_functions import get_oib_flight_df, get_S3_data_ds, get_tb_corrected_fpath, get_waveform_features_ds
from Utility.utility_functions import interpolate_auxiliary_data_to_track, filter_sentinel3_dataset
from Utility.utility_functions import correct_tbs_for_ow, calculate_gradient_ratio
from Utility.SeaIceAdvection import SeaIceDrift

########################################################################################################################################################
###################################################### Functions #######################################################################################
########################################################################################################################################################

#======================================================================================================================================================#

def drift_correct_oib_data(oib_df, s3_ds, hemisphere='nh'):
    """
    Drift correct the OIB data to the Sentinel-3 overpass time using the SeaIceDrift class and OSI-455 motion vectors.
    
    Parameters:
    oib_df (pandas.DataFrame): OIB dataframe containing the snow depth data and coordinates.
    s3_ds (xarray.Dataset): Sentinel-3 dataset containing.
    hemisphere (str): Hemisphere of the data ('nh' or 'sh').
    
    Returns:
    xarray.Dataset: OIB dataset with additional variables for the drift-corrected coordinates and the drift distance. 
    The dataset will have the following additional variables:
    - xc_corrected: x coordinate of the drift-corrected OIB data points
    - yc_corrected: y coordinate of the drift-corrected OIB data points
    - lon_corrected: longitude of the drift-corrected OIB data points
    - lat_corrected: latitude of the drift-corrected OIB data points
    - dX: drift distance in the x direction (m)
    - dY: drift distance in the y direction (m)
    """
    
    rounded_mean_s3_dt = pd.to_datetime(s3_ds['time_01'].mean().values).round('h') #we'll be performing hourly drift correction

    drift = SeaIceDrift()
    directory_path = get_local_directory_path('Drift_OSI455')
    drift.set_product_type('455')
    drift.load_motion_vectors(directory_path, 
                              rounded_mean_s3_dt.date()-pd.Timedelta(days=1),
                              rounded_mean_s3_dt.date()+pd.Timedelta(days=1),
                              hemisphere=hemisphere)

    #resample the motion vectors to hourly
    resample_time = True
    time_resolution = '1h'
    drift.resample_motion_vectors(resample_time=resample_time, time_resolution=time_resolution)

    #drift correct the OIB data
    spatial_interpolation = 'linear'
    results_time_resolution = '1h'

    datetimes = oib_df['datetime'].values
    xcs = oib_df['xc'].values
    ycs = oib_df['yc'].values

    results = drift.simulate_advection(datetimes, xcs, ycs, rounded_mean_s3_dt, 
                                        spatial_interpolation=spatial_interpolation,
                                        results_time_resolution=results_time_resolution)
    
    corrected_coords = drift.get_coords_at_datetime(results, rounded_mean_s3_dt).reset_index()
    corrected_coords[['start_x', 'start_y', 'start_dt']] = corrected_coords['start_coords'].str.extract(r'\((.*),(.*),(.*)\)')
    corrected_coords[['start_x', 'start_y']] = corrected_coords[['start_x', 'start_y']].astype(float)
    corrected_coords.set_index(['start_x', 'start_y'], inplace=True)

    oib_df.set_index(['xc', 'yc'], inplace=True)
    
    oib_df['xc_corrected'] = corrected_coords['x']
    oib_df['yc_corrected'] = corrected_coords['y']
    oib_df['lon_corrected'] = corrected_coords['lon']
    oib_df['lat_corrected'] = corrected_coords['lat']

    oib_df.reset_index(inplace=True)

    oib_df['dX'] = oib_df['xc_corrected'] - oib_df['xc']
    oib_df['dY'] = oib_df['yc_corrected'] - oib_df['yc']

    return oib_df.reset_index()

#======================================================================================================================================================#

def plot_track_overlap(s3_ds, oib_df, drift_corrected = False, s3_ts='01', save_fig = False):
    """
    Plot the OIB flight track (drift-corrected or uncorrected) and the Sentinel-3 data points to visualise the overlap between the two datasets. 
    
    Parameters:
    s3_ds (xarray.Dataset): Sentinel-3 dataset.
    oib_df (pandas.DataFrame): OIB dataframe.
    drift_corrected (bool): Whether to plot the drift-corrected OIB coordinates or the original coordinates.
    s3_ts (str): Timestamp identifier for the Sentinel-3 data to plot (e.g '01' or '20_ku').
    save_fig (bool): Whether to save the figure to the Media directory. If True, the figure will be saved to 'NaRRS/MEDIA/drift_correction' if drift_corrected is True, and 'NaRRS/MEDIA/OIB_S3_overlap' if drift_corrected is False.
    """

    if drift_corrected:
        x_coord, y_coord = 'xc_corrected', 'yc_corrected'
        save_dir = f'{get_local_directory_path("NaRRS_workspace")}/Media/drift_correction'
        fig_name = f'OIB_{oib_df["datetime"].min().strftime("%Y%m%d")}_S3_{pd.to_datetime(s3_ds[f"time_{s3_ts}"].mean().values).strftime("%Y%m%d")}_drift_corrected_track_overlap_oib_snod.png'
    else:
        x_coord, y_coord = 'xc', 'yc'
        save_dir = f'{get_local_directory_path("NaRRS_workspace")}/Media/OIB_S3_overlap'
        fig_name = f'OIB_{oib_df["datetime"].min().strftime("%Y%m%d")}_S3_{pd.to_datetime(s3_ds[f"time_{s3_ts}"].mean().values).strftime("%Y%m%d")}_track_overlap_oib_snod.png'
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    axs[0].set_extent([-180, 180, 65, 90], ccrs.PlateCarree())
    axs[0].add_feature(cfeature.COASTLINE)
    axs[0].add_feature(cfeature.LAND, color='silver')
    axs[0].scatter(s3_ds[f'lon_{s3_ts}'], s3_ds[f'lat_{s3_ts}'], color='orange', alpha=0.5, s=10, transform=ccrs.PlateCarree(), label='S3')
    oib_scatter = axs[0].scatter(oib_df[x_coord], oib_df[y_coord], c=oib_df['snow_depth'], s=10, transform=ccrs.epsg(6931), vmax=0.5)
    axs[0].gridlines(draw_labels=True, dms=False, x_inline=True, y_inline=True, xlocs=[-135, -90, -45, 0 , 45, 90, 135, 180], ylocs=[55, 65, 75, 85],alpha=0.5)
    plt.colorbar(oib_scatter, ax=axs[0], label='Snow Depth (m)', pad=0.01, extend='max' if oib_df['snow_depth'].max() > 0.5 else 'neither')
    axs[0].set_aspect('auto')

    axs[1].set_extent([oib_df['xc'].min()-100e3, oib_df['xc'].max()+100e3, oib_df['yc'].min()-100e3, oib_df['yc'].max()+100e3], ccrs.epsg(6931))
    axs[1].add_feature(cfeature.COASTLINE)
    axs[1].add_feature(cfeature.LAND, color='silver')
    axs[1].scatter(s3_ds[f'lon_{s3_ts}'], s3_ds[f'lat_{s3_ts}'], color='orange', alpha=0.5, s=20, transform=ccrs.PlateCarree())
    oib_scatter = axs[1].scatter(oib_df[x_coord], oib_df[y_coord], c=oib_df['snow_depth'], s=20, transform=ccrs.epsg(6931), vmax=0.5)
    axs[1].gridlines(draw_labels=False, dms=False, x_inline=True, y_inline=True, xlocs=[-135, -90, -45, 0 , 45, 90, 135, 180], ylocs=[55, 65, 75, 85],alpha=0.5)
    plt.colorbar(oib_scatter, ax=axs[1], label='Snow Depth (m)', pad=0.01, extend='max' if oib_df['snow_depth'].max() > 0.5 else 'neither')
    axs[1].set_aspect('auto')
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(save_dir, fig_name))
        print(f'Figure saved to {os.path.join(save_dir, fig_name)}')
    plt.show()

#======================================================================================================================================================#

def plot_oib_drift_correction(s3_ds, oib_df, save_fig = False):
    """
    Plot the original and drift-corrected OIB flight paths to visualise the drift correction. The left panel shows the original and drift-corrected flight paths, with the drift vectors shown as arrows. The right panel shows a histogram of the drift distances.
   
    Parameters:
    s3_ds (xarray.Dataset): Sentinel-3 dataset.
    oib_df (pandas.DataFrame): OIB dataframe containing the original and drift-corrected coordinates and the drift distances.
    save_fig (bool): Whether to save the figure to the Media directory.
    """
    regions_ds = xr.open_dataset(f'{get_local_directory_path("Surface_Masks")}/NSIDC-0780_SeaIceRegions_EASE2-N12.5km_v1.0.nc')
    land = xr.where(regions_ds['sea_ice_region_surface_mask'] >= 30, 1, np.nan)
    del regions_ds

    figure, axs = plt.subplots(1, 2, figsize=(12, 5))
    land.plot(ax=axs[0], add_colorbar=False, add_labels=False, cmap='Greys', alpha=0.5)
    axs[0].plot(oib_df['xc'], oib_df['yc'], color='blue', label='OIB')
    axs[0].quiver(oib_df['xc'][::500], oib_df['yc'][::500], oib_df['dX'][::500], oib_df['dY'][::500], scale=1e6, width=5e-3, headlength=1.5, headaxislength=1.25, headwidth=3, color='black')
    axs[0].plot(oib_df['xc_corrected'], oib_df['yc_corrected'], color='red', alpha=0.5, label='Drift-corrected OIB')
    axs[0].set_xlim([oib_df['xc'].min()-75e3, oib_df['xc'].max()+75e3])
    axs[0].set_ylim([oib_df['yc'].min()-75e3, oib_df['yc'].max()+75e3])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].legend(fontsize='x-small')
    axs[1].grid(alpha=0.5, zorder=0)
    lower_bound = np.round(oib_df['drift'].min() - 500, -3) - 1e3
    upper_bound = np.round(oib_df['drift'].max() + 500, -3) + 1e3
    bins = np.arange(lower_bound/1e3, upper_bound/1e3, 0.25)
    axs[1].hist(oib_df['drift']/1000, bins=bins, zorder=10)
    axs[1].set_xlabel('Drift (km)')
    axs[1].set_ylabel('Frequency')

    if save_fig:
        save_dir = f'{get_local_directory_path("NaRRS")}/Media/drift_correction'
        save_name = f'OIB_{oib_df["datetime"].min().strftime("%Y%m%d")}_S3_{pd.to_datetime(s3_ds["time_01"].mean().values).strftime("%Y%m%d")}_drift_correction.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_name))
        print(f'Figure saved to {os.path.join(save_dir, save_name)}')
    plt.show()

#======================================================================================================================================================#

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

#======================================================================================================================================================#

def iqr_filter(data, factor=1.5):
    """
    Get an IQR filter mask for the data. Values that are outside of factor*IQR outside the 25th and 75th percentiles will be considered outliers and will be marked as False in the mask.
    
    Parameters:
    data (array-like): Input data to filter
    factor (float): The factor to multiply the IQR by to determine the outlier thresholds.
    """
    
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor*iqr
    upper_bound = q3 + factor*iqr
    return (data >= lower_bound) & (data <= upper_bound)

#======================================================================================================================================================#

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

########################################################################################################################################################
###################################################### Main ############################################################################################
########################################################################################################################################################

if __name__ == '__main__':
    
    #set working directory
    os.chdir(get_local_directory_path('NaRRS_workspace'))

    #set hemisphere to process
    hemisphere = 'nh'

    #set OIB data directory
    oib_dir = get_local_directory_path('OIB')

    #load Sentinel-3 – OIB crossover file
    #this file was generated by analysig each OIB flight and locating the Sentinel-3 products within 24 hours containing any MWR data within 100 km of the flight
    crossover_df = pd.read_csv('Data/S3_OIB_overlap_counts.csv')
    crossover_df = crossover_df.rename(columns={'name':'S3', 'oib_file':'OIB'})
    crossover_df['s3_start_date'] = crossover_df['S3'].apply(lambda x: pd.to_datetime(x.split('_')[-15]).date())
    crossover_df['s3_end_date'] = crossover_df['S3'].apply(lambda x: pd.to_datetime(x.split('_')[-14]).date())
    crossover_df['oib_start_date'] = crossover_df['OIB'].apply(lambda x: pd.to_datetime(x.split('_')[1].removesuffix('.txt')).date())
    
    #iterate through the crossover file and process each overlap
    overlap_ds_ = []

    for i, row in crossover_df.iterrows():
        print(f"Processing {row['S3']} and {row['OIB']}")
        print(f'Combination {i+1} of {crossover_df.shape[0]}')
    
        oib_fpath = os.path.join(oib_dir, row['oib_start_date'].strftime('%Y/%m/') + row['OIB'])
        min_snod = 0.00
        oib_df = get_oib_flight_df(oib_fpath, min_snod)
        oib_df = oib_df[oib_df['snow_depth'] < oib_df['mean_fb']] #quality control (e.g., Kurtz et al. (2013))
        oib_df = oib_df[oib_df['surface_roughness'] >= 0] #this ensures the snow depths used are contrained by valid ATM data
        
        s3_metadata = get_sentinel3_sral_filename_components(row['S3'])
        s3_dirpath = get_local_directory_path(f'{s3_metadata["mission_id"]}_L2_SI')
        s3_dirpath = os.path.join(s3_dirpath, s3_metadata["cycle_number"],row['S3'])
        hz_1_names = ['time_01', 'lat_01', 'lon_01', 'tb_238_01', 'tb_238_quality_flag_01', 'tb_238_std_01', 'tb_365_01', 'tb_365_quality_flag_01', 'tb_365_std_01', 'sig0_water_01_c', 'sig0_water_01_plrm_ku']
        hz_20_names = ['time_20_ku', 'lat_20_ku', 'lon_20_ku','sig0_sea_ice_sheet_20_ku', 'surf_type_class_20_ku', 'freeboard_20_ku', 'sig0_ocog_20_ku']

        ease2_epsg = 6931
        print(f'Loading {row["S3"]}')
        s3_ds = get_S3_data_ds(s3_dirpath, vars_01=hz_1_names, vars_20_ku=hz_20_names, coord_transform_epsg=ease2_epsg)
        
        #plot the OIB flight path and the Sentinel-3 data
        # plot_track_overlap(s3_ds, oib_df, save_fig=False)

        #interpolate auxiliary data to the Sentinel-3 track
        s3_ds = interpolate_auxiliary_data_to_track(s3_ds, hemisphere, distance_to_ice_edge=False)

        #get the sentinel-3 corrected brightness temperatures
        print(f'Loading {row["S3"]} corrected brightness temperatures')
        corrected_tb_fpath = get_tb_corrected_fpath(row['S3'], hemisphere, 'gcp')
        corrected_tb_ds = xr.open_dataset(corrected_tb_fpath)

        #merge the corrected brightness temperatures with the Sentinel-3 data
        s3_ds = xr.merge([s3_ds, corrected_tb_ds], join='inner', compat='override')
        
        #filters the Sentinel-3 data
        lat_range = [90, 60]
        min_dist_coast = 25e3
        nsidc_region = np.arange(1,30,1)
        min_sic = 30
        max_t2m = 273.15 + 2 # +2 degrees to account for ERA5 warm bias

        # s3_ds = filter_sentinel3_dataset(s3_ds, hemisphere=hemisphere, min_lat=min_lat, min_dist_coast=min_dist_coast, nsidc_region=nsidc_region, min_sic=min_sic, max_t2m=max_t2m)
        s3_ds_1hz = s3_ds.drop_dims('time_20_ku')
        s3_ds_20_ku = s3_ds.drop_dims('time_01')
        s3_ds_1hz = filter_sentinel3_dataset(s3_ds_1hz, measurement_frequency = ['01'], lat_range=lat_range, min_dist_coast=min_dist_coast, nsidc_region=nsidc_region, min_sic=min_sic, max_t2m=max_t2m)
        s3_ds_20_ku = filter_sentinel3_dataset(s3_ds_20_ku, measurement_frequency = ['20_ku'], lat_range=lat_range, nsidc_region=nsidc_region, min_sic=min_sic, max_t2m=max_t2m)
        s3_ds = xr.merge([s3_ds_1hz, s3_ds_20_ku])
        
        #drift correct the OIB data
        oib_df = drift_correct_oib_data(oib_df, s3_ds)
        oib_df = oib_df.dropna(subset=['xc_corrected', 'yc_corrected']).reset_index(drop=True)
        oib_df['drift'] = np.sqrt(oib_df['dX']**2 + oib_df['dY']**2)

        #plot the original and corrected OIB flight paths
        plot_oib_drift_correction(s3_ds, oib_df, save_fig=False)
        
        #plot the dirft-corrected OIB flight path and the Sentinel-3 data
        plot_track_overlap(s3_ds, oib_df, drift_corrected=True, save_fig=False)

        #calulate the concentration-corrrected brightness temperatures
        s3_ds = correct_tbs_for_ow(s3_ds, hemisphere)

        #Calculate the gradient ratio
        s3_ds['gradient_ratio_corrected_01'] = calculate_gradient_ratio(s3_ds['tb_238_corrected_01'], s3_ds['tb_365_corrected_01'])
        s3_ds['gradient_ratio_corrected_01'].attrs = {'units':'K', 'long_name':'37/24 Gradient Ratio corrected for ow influence'}

        #for each brightness temperature datpoint, get various OIB snow depth statistics within a r km radius
        max_dist = 23.5e3/2
        min_oib_samples = 250
        bin_filter = True

        icebridge_vars = ['snow_depth', 'surface_roughness', 'mean_fb', 'thickness']
        for stat in ['mean', 'std', 'count',]:
            footprint_statistic = get_footprint_statistic(s3_ds['xc_01'].values, s3_ds['yc_01'].values, 
                                                                oib_df['xc_corrected'].values, oib_df['yc_corrected'].values,
                                                                oib_df, icebridge_vars, min_samples=min_oib_samples, 
                                                                max_dist=max_dist,  statistic=stat, 
                                                                bin_filter=bin_filter, plot_hists=False,)
            for var_name in icebridge_vars:
                s3_ds[f'{stat}_{var_name}_01'] = (('time_01'), footprint_statistic[var_name])
                s3_ds[f'{stat}_{var_name}_01'].attrs = {'long_name':var_name.replace('_', ' '), 'description': f"{stat} OIB {var_name} within {max_dist/1e3} km of S3 1Hz data point"}
        
        if s3_ds['mean_snow_depth_01'].count().values == 0:
            continue
        
        #get the pysiral-derived waveform features
        print(f'Loading {row["S3"]} waveform features')
        waveform_df = get_waveform_features_ds(s3_dirpath, hemisphere, si_only=True).to_dataframe().reset_index()
        #round the longitudes and latitudes to 6 decimal places to match the original S3 data format
        waveform_df['lon_20_ku'] = waveform_df['lon_20_ku'].round(6)
        waveform_df['lat_20_ku'] = waveform_df['lat_20_ku'].round(6)

        wf_vars_of_interest = ['peakiness', 'leading_edge_width', 'leading_edge_quality', 'leading_edge_peakiness', 'late_tail_to_peak_power', 'trailing_edge_decay','trailing_edge_quality',
                                    'trailing_edge_mean_absolute_deviation', 'trailing_edge_width', 'trailing_edge_decay_fit_quality', 'sigma0_ocean','first_maximum_power']
        wf_coords = ['time_20_ku', 'xc_20_ku', 'yc_20_ku', 'lon_20_ku', 'lat_20_ku']
        waveform_df = waveform_df[wf_coords+wf_vars_of_interest].dropna().reset_index(drop=True)

        #rename in the waveform var to match the S3 naming convention
        for k, wf_var in enumerate(wf_vars_of_interest):
            waveform_df = waveform_df.rename(columns={wf_var: wf_var+'_20_ku'})
            wf_vars_of_interest[k] = wf_var+'_20_ku'

        #sometimes we get trailing edge decays that are extremly large (1e6). 
        # these seem to be associated with waveforms that are contaminated.
        # the increadibly steep trailing edge lead to a trailing edge width of 1 bin.
        #therefore, we will filter out values where the trailing edge width ==1

        waveform_df = waveform_df[waveform_df['trailing_edge_width_20_ku'] > 1]

        #let's also ensure our trailing edge fit quality is reasonable
        waveform_df = waveform_df[waveform_df['trailing_edge_decay_fit_quality_20_ku'] > 0.8]

        #let's also fitler out waveforms in which the first maximum peak != the waveform maximum peak
        waveform_df = waveform_df[waveform_df['first_maximum_power_20_ku'] == 1]
        
        #calculate the mean value within a r km radius of each S3 data point
        min_ku_samples = 0
        extra_wf_vars = ['sig0_sea_ice_sheet_20_ku']
        s3_wf_vars_df = s3_ds.drop_dims('time_01').to_dataframe().reset_index()[['lon_20_ku', 'lat_20_ku'] + extra_wf_vars].dropna().reset_index(drop=True)
        
        #ensure the lon and lat are rounded to 6 decimal places for proper merging
        s3_wf_vars_df['lon_20_ku'] = s3_wf_vars_df['lon_20_ku'].round(6)
        s3_wf_vars_df['lat_20_ku'] = s3_wf_vars_df['lat_20_ku'].round(6)
        
        #merge the two waveform dfs on lon and lat
        waveform_df = waveform_df.merge(s3_wf_vars_df, on=['lon_20_ku', 'lat_20_ku'], how='inner')

        for stat in ['mean', 'std', 'count']:
            footprint_statistic = get_footprint_statistic(s3_ds['xc_01'].values, s3_ds['yc_01'].values, 
                                                                waveform_df['xc_20_ku'].values, waveform_df['yc_20_ku'].values,
                                                                waveform_df, wf_vars_of_interest+extra_wf_vars, min_ku_samples, 
                                                                max_dist,  statistic=stat, 
                                                                bin_filter=bin_filter, plot_hists=False,)
            for var_name in wf_vars_of_interest+extra_wf_vars:
                s3_ds[f"{var_name.split('_20_ku')[0]}_{stat}_01"] = (('time_01'), footprint_statistic[var_name])
                s3_ds[f"{var_name.split('_20_ku')[0]}_{stat}_01"].attrs = {'long_name':var_name.replace('_20_ku', ''), 'description': f"{stat} Ku-band {var_name.split('_20_ku')[0].replace('_', ' ')} within {max_dist/1e3} km of S3 1Hz data point"}     

        print(f'Appending {row["S3"]}-OIB overlap to list')
        overlap_ds_.append(s3_ds.drop_dims('time_20_ku'))
        time.sleep(1)
    
    print('Merge all the overlap datasets')
    overlap_ds = xr.concat(overlap_ds_, dim='time_01')
    overlap_ds = overlap_ds.where(overlap_ds['mean_snow_depth_01'] >= 0, drop=True) #keep only data with valid snow depth values
    bin_diameter_str = str(max_dist*2/1e3).replace('.','_') + 'km'
    overlap_ds.to_netcdf(f'Data/senitnel3_oib_xo_unified.nc')
    print(f'Overlap ds saved to Data/senitnel3_oib_xo_unified.nc')

