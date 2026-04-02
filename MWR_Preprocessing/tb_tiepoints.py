#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: connornelson
"""
import sys
sys.path.append('/home/cn/NaRRS')
import os
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from Utility.utility_functions  import get_sentinel3_product_dataframe_in_interval, get_tb_corrected_fpath, interpolate_auxiliary_data_to_track, calculate_gradient_ratio, get_unique_dates_within_interval
from Utility.directory_paths import get_local_directory_path

########################################################################################################################
################################################### Functions ##########################################################
########################################################################################################################

def get_sentinel3_tb_ds(sentinel3_product_dir, hemisphere='nh', era5_source='gcp', interp_auxiliary_data=True, 
                        handle_missing_corrected_tb_file = 'raise', handle_missing_sic_ds='temporal_interp',
                        handle_missing_edge_ds='use_sic'):
    
    """Reads the RTTOV-corrected brightness temperatures for a given Sentinel-3 product, and optionally interpolates auxiliary data to the track.  
    
    Parameters:
    sentinel3_product_dir (str): The name of the directory of the Sentinel-3 product to read (ends in .SEN3).
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere. Default is 'nh'.
    era5_source (str): The source of the ERA5 data interpolated to the Sentinel-3 track for RTTOV processing. Options are 'gcp' or 'cds' (Copernicus Climate Data Store). Default is 'gcp' (Google Cloud Public dataset).
    interp_auxiliary_data (bool): Whether to interpolate auxiliary data (sea ice concentration, sea ice type, NSIDC region, distance to coast, distance to ice edge) to the Sentinel-3 track. Default is True.
    handle_missing_corrected_tb_file (str): How to handle missing RTTOV-corrected brightness temperature files. Options are 'raise' to raise a FileNotFoundError, or 'ignore' to return None. Default is 'raise'.
    handle_missing_sic_ds (str): How to handle missing sea ice concentration datasets when interpolating auxiliary data. Options are 'raise' to raise a FileNotFoundError, 'temporal_interp' to use temporal interpolation between adjacent dates. Defaults to 'raise'.
    handle_missing_edge_ds (str): How to handle missing sea ice edge datasets when calculating distance to ice edge. Options are 'raise' to raise a FileNotFoundError, 'use_sic' to use the CDR sea ice concentration dataset to define the edge instead. Defaults to 'raise'.

    Returns:
    xarray.Dataset: A dataset containing the RTTOV-corrected brightness temperatures and optionally the interpolated auxiliary data for the given Sentinel-3 product.
    """
    
    try:
        ds = xr.open_dataset(get_tb_corrected_fpath(sentinel3_product_dir, hemisphere, era5_source))
    except FileNotFoundError:
        if handle_missing_corrected_tb_file == 'raise':
            raise FileNotFoundError(f'RTTOV corrected file not found for {sentinel3_product_dir}')
        elif handle_missing_corrected_tb_file == 'ignore':
            return
    
    if 'xc_01' not in ds:
        raise ValueError(f'No EASE2 grid coordinates found in {sentinel3_product_dir}') 

    if interp_auxiliary_data:
        ds = interpolate_auxiliary_data_to_track(ds, hemisphere, ['01'], sea_ice_concentration=True, sea_ice_type=True, nsidc_region=True, 
                                                    distance_to_coast=True, distance_to_ice_edge=True, handle_missing_sic_ds=handle_missing_sic_ds, handle_missing_edge_ds=handle_missing_edge_ds)
    return ds


########################################################################################################################
###################################################### Main ############################################################
########################################################################################################################

if __name__ == "__main__":

    hemisphere = 'nh'
    lat_limit = 40

    #dates of interest
    season_start_years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
    dates = []
    for year in season_start_years:
        dates.append(pd.date_range(f'{year}-10-01', f'{year+1}-05-01', freq='D'))
    dates = np.array([d for sublist in dates for d in sublist])

    overwrite = False
    reset_enumerator = False

    #for each date, we'll find the sentinel-3 products within a two week interval (± 1 week from the date of interest) 
    #then, we'll read the RTTOV-corrected brightness temperatures for all the tracks in these products, and filter to only retain relevant good-quality samples for our tie points.
    #these samples will then be saved for tie point computation and analysis in the subsequent scripts. 
    #to improve efficiency, if we have a continuous interval of dates, we can itertively add the next day to the previous interval and drop the last day, rather than re-reading all the data for each new interval.
    for i, date in enumerate(dates):
        print(f'Processing date: {date.strftime("%Y-%m-%d")}')
        
        #check if the tie points already exist
        save_dir = f'{get_local_directory_path("Tie_Points")}/{date.strftime("%Y/%m")}'
        save_name = f's3_mwr_tiepoints_{hemisphere}_{date.strftime("%Y%m%d")}.nc'
        
        if os.path.exists(f'{save_dir}/{save_name}') and not overwrite:
            print(f'Tie points already exist for {date.strftime("%Y-%m-%d")}')
            reset_enumerator = True
            continue
        
        if reset_enumerator:
            i = 0
            reset_enumerator = False
            
        os.makedirs(save_dir, exist_ok=True)
        
        #check if dates are continuous
        if i > 0:
            if (date - dates[i-1]).days != 1:
                i = 0
                
        #get all the dates within two weeks of the overlap
        days = 7
        all_dates = get_unique_dates_within_interval([date], days)

        if i == 0:
            
            #find all the polar sentinel3 products on these dates
            products_df = get_sentinel3_product_dataframe_in_interval(all_dates[0], all_dates[-1]+pd.Timedelta(days=1), hemisphere, lat_limit, enhanced_measurements=True)

            #read the tb-corrected sentinel3 data for each day in the interval in parallel
            era5_source = 'gcp'
            interp_auxiliary_data = True
            handle_missing_corrected_tb_file = 'ignore'
            handle_missing_sic_ds = 'temporal_interp'
            handle_missing_edge_ds = 'use_sic'
            s3_ds = Parallel(n_jobs=4)(delayed(get_sentinel3_tb_ds)(s3_dir, hemisphere, era5_source, interp_auxiliary_data, handle_missing_corrected_tb_file, handle_missing_sic_ds, handle_missing_edge_ds) for s3_dir in products_df['name'])

            #remove None values
            s3_ds = [ds for ds in s3_ds if ds is not None]
            
            #concatenate all the data and add a date coordinate
            s3_ds = xr.concat(s3_ds, dim='time_01')
            s3_ds['date'] = xr.DataArray(pd.to_datetime(s3_ds['time_01'].values).date, dims='time_01')

        else:
            #add the next day from the new interval
            new_products_df = get_sentinel3_product_dataframe_in_interval(all_dates[-1], all_dates[-1]+pd.Timedelta(days=1), hemisphere, lat_limit, enhanced_measurements=True)
            new_day_ds = Parallel(n_jobs=4)(delayed(get_sentinel3_tb_ds)(s3_dir, hemisphere, 'gcp', True, 'ignore') for s3_dir in new_products_df['name'])
            new_day_ds = [ds for ds in new_day_ds if ds is not None]
            new_day_ds = xr.concat(new_day_ds, dim='time_01')
            new_day_ds['date'] = xr.DataArray(pd.to_datetime(new_day_ds['time_01'].values).date, dims='time_01')
            s3_ds = xr.concat([s3_ds,new_day_ds], dim='time_01')
        
        #crop the data to only the dates of interest (some tracks may overlap two dates,
        #or may have an old date if iterating over a constant date interval)
        s3_ds = s3_ds.where(s3_ds.date.isin([d.date() for d in all_dates]), drop=True)

        #drop any points where we do not have the corrected brightness temperatures
        s3_ds = s3_ds.where((~s3_ds['corr_tb_238_01'].isnull()) & (~s3_ds['corr_tb_365_01'].isnull()), drop=True)

        #compute the gradient ratio
        s3_ds['corr_gradient_ratio_01'] = calculate_gradient_ratio(s3_ds['corr_tb_238_01'], s3_ds['corr_tb_365_01'])
        s3_ds['corr_gradient_ratio_01'].attrs = {'units':'K', 'long_name':'37/24 Gradient Ratio'}

        #do some inital quality-control filtering
        good_quality_flag = 0
        s3_ds = s3_ds.where((s3_ds['tb_238_quality_flag_01'] == good_quality_flag) & (s3_ds['tb_365_quality_flag_01'] == good_quality_flag), drop=True) 

        #some filtering common to both tie point sets (min distance to land and ice edge)
        min_dist_to_coast = 100e3
        min_dist_to_ice_edge = 100e3
        s3_ds = s3_ds.where((s3_ds['dist_to_coast_01'] > min_dist_to_coast) & (s3_ds['dist_to_ice_edge_01'] > min_dist_to_ice_edge), drop=True)
        
        #set SIC var name from the data source (obs = OSISAF SIC, era5 = ERA5 SIC)
        sic_source = 'obs' #era5
        sic_var = 'ice_conc_01' if sic_source == 'obs' else 'siconc_01'
        
        #get sea ice tie point samples
        s3_ic_ds = s3_ds.copy()
        min_sic = 95 if sic_source == 'obs' else 0.95
        s3_ic_ds = s3_ic_ds.where((s3_ic_ds[sic_var] > min_sic), drop=True)

        #get open water tie point samples
        s3_ow_ds = s3_ds.copy()
        max_sic = 0
        max_dist_to_ice_edge = 300e3
        s3_ow_ds = s3_ow_ds.where((s3_ow_ds[sic_var] <= max_sic) & (s3_ow_ds['dist_to_ice_edge_01'] <= max_dist_to_ice_edge), drop=True)    

        #merge the two tie point sample sets
        tp_ds = xr.concat([s3_ow_ds, s3_ic_ds], dim='time_01')
        tp_ds = tp_ds.sortby('time_01')
        vars_to_save = ['tb_238_01', 'tb_365_01', 'corr_tb_238_01', 'corr_tb_365_01', 'corr_gradient_ratio_01',
                            sic_var, 'region_01', 'dist_to_coast_01', 'dist_to_ice_edge_01', 't2m_01']
        tp_ds = tp_ds[vars_to_save]
        tp_ds['tp_type'] = (('time_01'), np.where(tp_ds[sic_var] > 0, 1, 0))
        tp_ds['tp_type'].attrs = {'units':'1 = Sea Ice, 0 = Open Water', 'long_name':'Tie Point Type'}
        
        #save the tie point samples
        tp_ds.to_netcdf(f'{save_dir}/{save_name}')
        print(f'Tie points saved for {date.strftime("%Y-%m-%d")}')