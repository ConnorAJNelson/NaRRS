"""
Author: Connor Nelson
Email: connor.nelson.21@ucl.ac.uk
"""

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from glob import glob
import pyproj
from scipy.spatial import KDTree
from cftime import num2pydate
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from joblib import Parallel, delayed
from Utility.directory_paths import get_local_directory_path

#=====================================================================================================================================================#
#=====================================================================================================================================================#
#=====================================================================================================================================================#

def get_sentinel3_sral_filename_components(filename):
    """
    Get product file information from the components of a Sentinel-3 SRAL filename

    Parameters:
    filename (str): the filename to extract the components from

    Returns:
    dict: dictionary containing the components of the filename
    """

    pattern = (r"(?P<mission_id>S3[AB_]?)_"
               r"(?P<instrument>SR)_"
               r"(?P<processing_level>[012]?)_"
               r"(?P<data_type>LAN_[A-Z]{2})_"
               r"(?P<sensing_start>\d{8}T\d{6})_"
               r"(?P<sensing_stop>\d{8}T\d{6})_"
               r"(?P<creation_date>\d{8}T\d{6})_"
               r"(?P<sensing_duration>\d{4})_"
               r"(?P<cycle_number>\d{3})_"
               r"(?P<relative_orbit>\d{3})_"
               r"_{3,}_"
               r"(?P<production_centre>[A-Z0-9]{3,})_"
               r"(?P<platform>[POFRD])_"
               r"(?P<timeliness>NR|ST|NT)_"
               r"(?P<baseline_collection>\d{3})\.SEN3")
    
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Filename format does not match Sentinel-3 naming convention: {filename}")
    
    return match.groupdict()

#=====================================================================================================================================================#

def transform_coords(proj1, proj2, x, y):
    """
    Transform coordinates from proj1 to proj2
    
    Parameters:
    proj1 (int): EPSG code of the current projection
    proj2 (int): EPSG code of the target projection
    x (float or array-like): x-coordinate(s) in the current projection
    y (float or array-like): y-coordinate(s) in the current projection
    
    Returns:
    tuple: Transformed x and y coordinates in the target projection
    
    Common projections: 
    WGS 84 (4326)
    EASE2 (NH: 6931, SH:6932)
    Polar Stereographic (NH: 3413, SH:3976)
    EASE (NH:3408, SH:3409)
    """

    proj = pyproj.Transformer.from_crs(proj1, proj2, always_xy=True)

    return proj.transform(x, y)

#=====================================================================================================================================================#

def convert_lon_easting_format(longitude):
    if longitude >180 and longitude <= 360:
        longitude = longitude - 360
    return longitude

#=====================================================================================================================================================#

def get_ease2_epsg(hemisphere):
    if hemisphere == 'nh':
        return 6931
    if hemisphere == 'sh':
        return 6932

#=====================================================================================================================================================#

def get_sentinel3_product_dataframe_in_interval(start_datetime, end_datetime, hemisphere, lat_limit=None, enhanced_measurements=True):
    
    """
    Get a dataframe of Sentinel-3 L2 Sea Ice Thematic products within a specified time interval, hemisphere, and optional latitude limit. 
    This function assumes the 'sentinel3_sral_products_20160921_20230430.parquet' is available in the '../Misc/' directory.
    This parquet file should contain metadata about the Sentinel-3 L2 Sea Ice Thematic products, including, for each product:
    - 'product_path': Path to the product file.
    - 'satellite': Satellite identifier (e.g., 'S3A' or 'S3B').
    - 'name': Name of the product.
    - 'sensing_start': Start datetime of sensing.
    - 'sensing_end': End datetime of sensing.
    - 'cycle': Orbit cycle number.
    - 'enhanced_exists': Boolean indicating if enhanced measurements are available. 
    - 'min_lat': Minimum latitude covered by the product.
    - 'max_lat': Maximum latitude covered by the product.

    Parameters:
    start_datetime (Timestamp): Start of the time interval.
    end_datetime (Timestamp): End of the time interval.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    lat_limit (float, optional): Latitude limit for filtering products. Defaults to None.
    enhanced_measurements (bool, optional): If True, only include products with enhanced measurements available (rarely, there is no enhanced_measurement.nc processed by ESA). Defaults to True.
    
    Returns:
    DataFrame: Filtered dataframe of Sentinel-3 products.
    
    """
    
    all_files_df = pd.read_parquet(f'{get_local_directory_path("NaRRS_workspace")}/Data/sentinel3_sral_products_20160921_20230430.parquet')
    if enhanced_measurements:
        all_files_df = all_files_df[all_files_df['enhanced_exists'] == True]
    
    if lat_limit is not None:
        if hemisphere == 'nh':
            all_files_df = all_files_df[all_files_df['max_lat'] >= lat_limit]
        elif hemisphere == 'sh':
            all_files_df = all_files_df[all_files_df['min_lat'] <= lat_limit]
        else:
            raise ValueError('Hemisphere must be either "nh" or "sh"')

    all_files_df = all_files_df[(all_files_df['sensing_end'] >= start_datetime) & (all_files_df['sensing_start'] < end_datetime)]
    return all_files_df

#=====================================================================================================================================================#


def get_S3_data_ds(dirpath, data_level='enhanced', vars_01=None, vars_20_ku=None, vars_20_c=None, decode_times=True, coord_transform_epsg=None):
    """
    Load Sentinel-3 L2 Sea Ice Thematic data as an xarray Dataset.
    
    Parameters:
    dirpath (str): The directory path where the Sentinel-3 L2 Sea Ice Thematic product data is stored.
    data_level (str, optional): The data level to load, either 'enhanced' or 'standard'. Defaults to 'enhanced'. Note MWR data is only in the enhanced products.
    vars_01 (list of str, optional): List of variable names to load from the 01 Hz data. If None, all variables will be loaded. Defaults to None.
    vars_20_ku (list of str, optional): List of variable names to load from the 20 Hz Ku-band data. If None, all variables will be loaded. Defaults to None.
    vars_20_c (list of str, optional): List of variable names to load from the 20 Hz C-band data. If None, all variables will be loaded. Defaults to None.
    decode_times (bool, optional): Whether to decode time variables into datetime objects. Defaults to True.
    coord_transform_epsg (int, optional): If provided, transform the latitude and longitude coordinates to the specified EPSG projection. Defaults to None.

    Returns:
    xarray.Dataset: The loaded Sentinel-3 L2 Sea Ice Thematic dataset.
    """
    
    data_fpath = os.path.join(dirpath, f'{data_level}_measurement.nc')
    if not os.path.exists(data_fpath):
        raise FileNotFoundError(f'{data_fpath} not found')
    s3_ds = []
    for ts, ts_vars in zip(['01', '20_ku', '20_c'], [vars_01, vars_20_ku, vars_20_c]):
        if ts_vars is not None:
            ds = xr.open_dataset(data_fpath, decode_times=decode_times)[ts_vars]
            lon_attrs = ds[f'lon_{ts}'].attrs
            lon_encoding = ds[f'lon_{ts}'].encoding
            ds[f'lon_{ts}'] = (ds[f'lon_{ts}'] + 180) % 360 - 180
            ds[f'lon_{ts}'].attrs = lon_attrs
            ds[f'lon_{ts}'].encoding = lon_encoding
            if coord_transform_epsg is not None:
                xc, yc = transform_coords(4326, coord_transform_epsg, ds[f'lon_{ts}'].values, ds[f'lat_{ts}'].values)
                ds = ds.assign_coords({f'xc_{ts}':((f'time_{ts}'),xc), f'yc_{ts}':((f'time_{ts}'),yc)})
                ds[f'xc_{ts}'].attrs = {'units':'m', 'long_name':f'{ts.split("_")[0]} Hz x-coordinate in EPSG:{coord_transform_epsg})'}
                ds[f'yc_{ts}'].attrs = {'units':'m', 'long_name':f'{ts.split("_")[0]} Hz y-coordinate in EPSG:{coord_transform_epsg})'}
            s3_ds.append(ds) 
    return xr.merge(s3_ds)

#=====================================================================================================================================================#

def get_tb_corrected_fpath(s3_product_name, hemisphere, era5_source, iteration=1):
    """
    Get the file path for the RTTOV-corrected brightness temperatures corresponding to a given Sentinel-3 product, hemisphere, ERA5 source, and iteration.
    
    Parameters:
    s3_product_name (str): The name of the Sentinel-3 product.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    era5_source (str): The source of the ERA5 data used for the RTTOV correction, either 'gcp' (Google Cloud Public Data) or 'cds' (Climate Data Store).
    iteration (int, optional): The iteration of the RTTOV correction, either 1 for standard RTTOV-corrected brightness temperatures or 2 for double-difference RTTOV-corrected brightness temperatures. Defaults to 1. 
    """
    rttov_dir = get_local_directory_path('rttov_tbs')
    if iteration not in [1, 2]:
        raise ValueError(f"Invalid iteration: {iteration}. Iteration must be either 1 or 2")
    
    file_identifiers = get_sentinel3_sral_filename_components(s3_product_name)
    satellite = file_identifiers['mission_id']
    cycle = file_identifiers['cycle_number']

    if era5_source not in ['gcp', 'cds']:
        raise ValueError(f"Invalid source: {era5_source}. Source must be one of 'gcp' or 'cds'")
    
    file_dir = os.path.join(rttov_dir, satellite, cycle, s3_product_name.split('/')[-1])

    if iteration == 1:  # standard RTTOV corrected Tbs
        file_paths = glob(f'{file_dir}/*rttov_corrected_tbs_{hemisphere}_*v{era5_source}.nc')
    elif iteration == 2:  # double difference RTTOV corrected Tbs
        file_paths = glob(f'{file_dir}/*rttov_dd_corrected_tbs_{hemisphere}_*v{era5_source}.nc')
    if len(file_paths) > 1:
        raise ValueError(f"Multiple RTTOV corrected files found for {s3_product_name}. Please check the directory: {file_dir}")
    elif len(file_paths) == 1:
        return file_paths[0]
    raise FileNotFoundError(f'RTTOV corrected file not found for {s3_product_name}.\nCheck the directory: {file_dir}\n')

#=====================================================================================================================================================#


def find_waveform_features_nc_paths(s3_product_name):
    l2_l1p_matched_df = pd.read_csv(f'{get_local_directory_path("NaRRS_workspace")}/Data/l2_l1p_matched.csv')
    l1p_paths = l2_l1p_matched_df[l2_l1p_matched_df['L2'] == s3_product_name]['L1P_path'].values.tolist()
    return l1p_paths
    
#========================================================================================================================#

def get_waveform_features_ds(s3_dirpath, hemisphere, si_only=True):
    """
    Load the pysrial-processed Sentinel-3 L2 Sea Ice Thematic Dataset""

    Parameters:
    s3_dirpath (str): The directory path of the Sentinel-3 L2 Sea Ice Thematic product.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    si_only (bool, optional): If True, filter the dataset to include only sea ice surface types as defined by the surface_type_classification product variable. Default is True.
    
    Returns:
    xr.Dataset: Dataset containing the waveform features and coordinates.

    Note, pysiral may split the original along-track dataset across multiple files and hence there may be multiple files to load and concatenate together. 
    The final dataset is sorted by time and duplicate time entries are dropped.

    """

    waveform_fpaths = find_waveform_features_nc_paths(s3_dirpath.split('/')[-1])
    waveforms = []
    for fpath in waveform_fpaths:
        classifer_ds = xr.open_dataset(fpath, group ='classifier', drop_variables=['brightness_temperature_238', 'brightness_temperature_365'])
        orbit_ds = xr.open_dataset(fpath, group ='time_orbit', drop_variables=['altitude', 'altitude_rate', 'antenna_pitch', 'antenna_roll', 'antenna_yaw', 'antenna_mispointing', 'orbit_flag'])
        ds = xr.merge([classifer_ds, orbit_ds], join='inner')
        ds =  ds.rename_dims({'n_records': 'time_20_ku'})
        ds = ds.assign_coords({'time_20_ku':num2pydate(ds['timestamp'].values, units="seconds since 1970-01-01 00:00:00.0")})
        ds = ds.rename_vars({'latitude': 'lat_20_ku', 'longitude': 'lon_20_ku'})
        ds['lon_20_ku'] = (('time_20_ku'),Parallel(n_jobs=20, prefer="threads")(delayed(convert_lon_easting_format)(longitude) for longitude in ds['lon_20_ku'].values))
        xc, yc = transform_coords(4326, get_ease2_epsg(hemisphere), ds['lon_20_ku'].values, ds['lat_20_ku'].values)
        ds = ds.assign_coords({'xc_20_ku':(('time_20_ku'),xc), 'yc_20_ku':(('time_20_ku'),yc)})
        if si_only:
            ds = ds.where(ds['surface_type_classification'] == 1, drop=True)
        waveforms.append(ds)
    waveform_ds = xr.concat(waveforms, dim='time_20_ku')
    waveform_ds = waveform_ds.drop_duplicates('time_20_ku')
    waveform_ds = waveform_ds.sortby('time_20_ku').drop_vars(['timestamp'])
    return  waveform_ds

#=====================================================================================================================================================#

def nearest_neighbour_interpolation(grid_xcs, grid_ycs, values, track_xcs, track_ycs, max_dist = None, return_dist = False):
    """
    Perform nearest neighbour interpolation

    Parameters:
    grid_xcs (array-like): x-coordinates of the grid points.
    grid_ycs (array-like): y-coordinates of the grid points.
    values (array-like): values at the grid points.
    track_xcs (array-like): x-coordinates of the track points to interpolate to.
    track_ycs (array-like): y-coordinates of the track points to interpolate to.
    max_dist (float, optional): Maximum distance for interpolation. Values beyond this distance will be set to NaN. Defaults to None.
    return_dist (bool, optional): If True, also return the distances from the track points to their nearest grid points. Defaults to False.

    Returns:
    np.ndarray or tuple: Array of interpolated values at the track points. If return_dist is True, also returns an array of distances to the nearest grid points.


    """
    if np.shape(grid_xcs) != np.shape(values):
        grid_xcs, grid_ycs = np.meshgrid(grid_xcs, grid_ycs)
    
    coords = list(zip(track_xcs, track_ycs))    
    tree = KDTree(np.c_[grid_xcs.ravel(),grid_ycs.ravel()])
    
    dist, idx = tree.query(coords, k=1)
    nearest_values = values.ravel()[idx]
    nearest_values = nearest_values.astype(float)

    if max_dist:
        nearest_values[dist >  max_dist] = np.nan
    if return_dist:
        return nearest_values, dist
    return nearest_values

#========================================================================================================================#

def linear_interpolation(grid_xcs, grid_ycs, values, track_xcs, track_ycs):
    """
    Perform linear interpolation using Verde's Linear interpolation class.
    
    Parameters:
    grid_xcs (array-like): x-coordinates of the grid points.
    grid_ycs (array-like): y-coordinates of the grid points.
    values (array-like): values at the grid points.
    track_xcs (array-like): x-coordinates of the track points to interpolate to.
    track_ycs (array-like): y-coordinates of the track points to interpolate to.
    
    Returns:
    np.ndarray: Array of interpolated values at the track points.
    """

    if np.shape(grid_xcs) != np.shape(values):
        grid_xcs, grid_ycs = np.meshgrid(grid_xcs, grid_ycs)

    import verde as vd

    interpolator = vd.Linear().fit((grid_xcs, grid_ycs), values)
    interped_points = interpolator.predict((track_xcs, track_ycs))

    return interped_points

#========================================================================================================================#


def get_cdr_sea_ice_concentation_ds(date, hemisphere):

    """
    Get the OSI SAF CDR/ICDR sea ice concentration dataset for a given date and hemisphere.
    
    Parameters:
    date (pd.Timestamp): The date for which to load the sea ice concentration dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    
    Returns:
    xr.Dataset: The sea ice concentration dataset.

    Note, this accounts for the temporal availiablilty of the CDR dataset and changes to the ICDR after 2021-01-01.
    """

    sic_dir = get_local_directory_path("OSISAF_SIC")
    if date < pd.to_datetime('2021-01-01'):
        target_dir = os.path.join(sic_dir, f"CDR/{date.strftime('%Y/%m')}")
        product_name = f"ice_conc_{hemisphere}_ease2-250_cdr-v3p0_{date.strftime('%Y%m%d1200')}.nc"
    else:
        target_dir = os.path.join(sic_dir, f"ICDR/{date.strftime('%Y/%m')}")
        product_name = f"ice_conc_{hemisphere}_ease2-250_icdr-v3p0_{date.strftime('%Y%m%d1200')}.nc"
    return xr.open_dataset(os.path.join(target_dir, product_name))

#========================================================================================================================#

def get_cdr_sea_ice_edge_ds(date, hemisphere):
    """
    Get the C3S CDR/ICDR sea ice edge dataset for a given date and hemisphere.
    
    Parameters:
    date (pd.Timestamp): The date for which to load the sea ice edge dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    
    Returns:
    xr.Dataset: The sea ice edge dataset.
    """

    edge_dir = get_local_directory_path("C3S_Ice_Edge")

    if date < pd.to_datetime('2021-01-01'):
        target_dir = os.path.join(edge_dir, f"C3S/CDR/{date.strftime('%Y/%m')}")
        product_name = f"ice_edge_{hemisphere}_ease2-125_cdr-v3p0_{date.strftime('%Y%m%d1200')}.nc"
    else:
        target_dir = os.path.join(edge_dir, f"C3S/ICDR/{date.strftime('%Y/%m')}")
        product_name = f"ice_edge_{hemisphere}_ease2-125_icdr-v3p0_{date.strftime('%Y%m%d1200')}.nc"
    return xr.open_dataset(os.path.join(target_dir, product_name))

#========================================================================================================================#

def get_polarstereographic_epsg(hemisphere, ellipsoid='WGS84'):
    if hemisphere == 'nh':
        if ellipsoid == 'WGS84':
            return 3413
        if ellipsoid == 'Hughes1980':
            return 3411
    if hemisphere == 'sh':
        if ellipsoid == 'WGS84':
            return 3976
        if ellipsoid == 'Hughes1980':
            return 3412
        
#========================================================================================================================#

def get_sea_ice_type_ds(date, hemisphere, osisaf_or_c3s = 'osisaf'):
    """
    Get the sea ice type dataset for a given date and hemisphere from either OSISAF or C3S.

    date (pd.Timestamp): The date for which to load the sea ice type dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    osisaf_or_c3s (str, optional): Whether to load the OSISAF ('osisaf') or C3S ('c3s') product. Defaults to 'osisaf'.
    
    Returns:
    xr.Dataset: The sea ice type dataset.

    Note, the OSISAF product is only available until 2020-12-31, while the C3S product is available until 2022-12-31.
    Also, there is no OSISAF product for 2016-10-17, so a dummy dataset with the same coordinates but NaN values is returned for this date.
    Also, the OSISAF product coordinates are transformed from polar stereographic grid to EASE2.
    """ 

    type_dir = get_local_directory_path("Ice_Type")
    
    if osisaf_or_c3s == 'osisaf':
        #there is no product for 2016-10-17, so we return a dummy dataset
        if date.strftime('%Y-%m-%d') == '2016-10-17':
            return get_missing_osisaf_ice_type_dummy_ds(hemisphere, type_dir)
        
        target_dir = os.path.join(type_dir, f"OSI403d/{date.strftime('%Y/%m')}")
        product_name = f"ice_type_{hemisphere}_polstere-100_multi_{date.strftime('%Y%m%d1200')}.nc"
        ds = xr.open_dataset(os.path.join(target_dir, product_name))
        xcs, ycs = np.meshgrid(ds['xc'].values, ds['yc'].values)
        xcs, ycs = transform_coords(get_polarstereographic_epsg(hemisphere, ellipsoid='Hughes1980'), get_ease2_epsg(hemisphere), xcs, ycs)
        ds = ds.assign_coords({'xc_ease2':(('yc','xc'),xcs), 'yc_ease2':(('yc','xc'),ycs)})
        return ds

    if date < pd.to_datetime('2021-01-01'):
        target_dir = os.path.join(type_dir, f"C3S/CDR/{date.strftime('%Y/%m')}")
        product_name = f"ice_type_{hemisphere}_ease2-250_cdr-v3p0_{date.strftime('%Y%m%d1200')}.nc"
    else:
        target_dir = os.path.join(type_dir, f"C3S/ICDR/{date.strftime('%Y/%m')}")
        product_name = f"ice_type_{hemisphere}_ease2-250_icdr-v3p0_{date.strftime('%Y%m%d1200')}.nc"
    return xr.open_dataset(os.path.join(target_dir, product_name))

#========================================================================================================================#

def get_missing_osisaf_ice_type_dummy_ds(hemisphere, type_dir):
    """
    Get a dummy dataset for the missing OSISAF ice type product on 2016-10-17.
    
    Parameters:
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    type_dir (str): The directory where the OSISAF ice type products are stored, used to load the coordinates and variable names from the original dataset.
     
    Returns:
    xr.Dataset: A dummy dataset with the same coordinates as the original dataset but with NaN values for the ice type, confidence level, and status flag variables.
    """

    #load the dataset for 2016-10-16 (the day before the missing date)
    date = pd.to_datetime('2016-10-16')
    target_dir = os.path.join(type_dir, f"OSI403d/{date.strftime('%Y/%m')}")
    product_name = f"ice_type_{hemisphere}_polstere-100_multi_{date.strftime('%Y%m%d1200')}.nc"
    ds = xr.open_dataset(os.path.join(target_dir, product_name))
    xcs, ycs = np.meshgrid(ds['xc'].values, ds['yc'].values)
    xcs, ycs = transform_coords(get_polarstereographic_epsg(hemisphere, ellipsoid='Hughes1980'), get_ease2_epsg(hemisphere), xcs, ycs)
    ds = ds.assign_coords({'xc_ease2':(('yc','xc'),xcs), 'yc_ease2':(('yc','xc'),ycs)})

    dummy_ds = ds.copy()
    dummy_ds['ice_type'] = xr.DataArray(np.full(ds['ice_type'].shape, np.nan), dims=ds['ice_type'].dims)
    dummy_ds['confidence_level'] = xr.DataArray(np.full(ds['confidence_level'].shape, np.nan), dims=ds['confidence_level'].dims)
    dummy_ds['status_flag'] = xr.DataArray(np.full(ds['status_flag'].shape, np.nan), dims=ds['status_flag'].dims)
    dummy_ds['time'] = [np.nan]
    dummy_ds = dummy_ds.drop_dims(['nv'])
    return dummy_ds
    
#========================================================================================================================#

def sic_to_edge_ds(sic_ds):
    """
    Convert a sea ice concentration dataset to a sea ice edge dataset by classifying the pixels as open water, open ice, and closed ice.
    We use logic from the C3S CDR flags.
    1: No ice or very open ice (less than 30% ice concentration)
    2: Open ice (30% to 70% ice concentration)
    3: Closed ice (more than 70% ice concentration)

    Parameters:
    sic_ds (xr.Dataset): The sea ice concentration dataset to convert. Must contain the variable 'ice_conc' with values between 0 and 1.
    
    Returns:
    xr.Dataset: A new ice edge dataset with an 'ice_edge' variable classifying the sea ice edge as 1 for no ice or very open ice, 2 for open ice, and 3 for closed ice.
    """

    if 'ice_conc' not in sic_ds:
        raise ValueError("Input dataset must contain the variable 'ice_conc'")
    
    if sic_ds['ice_conc'].min() < 0 or sic_ds['ice_conc'].max() > 1:
        raise ValueError("The 'ice_conc' variable must be between 0 and 1")

    edge_ds = sic_ds.copy()
    ice_edge = np.full(sic_ds['ice_conc'].shape, np.nan)
    ice_edge[sic_ds['ice_conc'] < 0.3] = 1
    ice_edge[(sic_ds['ice_conc'] >= 0.3) & (sic_ds['ice_conc'] <= 0.7)] = 2
    ice_edge[sic_ds['ice_conc'] > 0.7] = 3
    edge_ds['ice_edge'] = xr.DataArray(ice_edge, dims=sic_ds['ice_conc'].dims)
    edge_ds['ice_edge'].attrs = {'long_name':'Sea ice edge classification', 'flag_descriptions':'1=No ice or very open ice, 2=Open ice, 3=Closed ice'}
    edge_ds = edge_ds.drop_vars(['ice_conc', 'raw_ice_conc_values', 'total_standard_uncertainty', 'smearing_standard_uncertainty', 'algorithm_standard_uncertainty'])
    return edge_ds

#========================================================================================================================#

def add_distance_to_ice_edge(sentinel3_ds, hemisphere, measurement_frequency = ['01', '20_ku'], handle_missing_edge_ds='raise'):
    """
    Add the distance to the sea ice edge as a new variable in the Sentinel-3 dataset for the specified measurement frequencies.     
    The distance to the ice edge is calculated as the distance to the nearest pixel where the sea ice concentration is less than 30%, using the C3S CDR/ICDR sea ice edge dataset or the OSISAF CDR SIC dataset using the same logic.
    
    Parameters:
    sentinel3_ds (xr.Dataset): The Sentinel-3 dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    measurement_frequency (list of str, optional): List of measurement frequencies to calculate the distance for. Defaults to ['01', '20_ku'].
    handle_missing_edge_ds (str, optional): How to handle missing sea ice edge datasets. Options are 'raise' to raise a FileNotFoundError, 'use_sic' to use the CDR sea ice concentration dataset to define the edge instead. Defaults to 'raise'.  

    Returns:
    xr.Dataset: The Sentinel-3 dataset with the distance to the ice edge added, named 'dist_to_ice_edge_{measurement_frequency}'.
    """
    
    
    
    mean_ds_date = pd.to_datetime(sentinel3_ds['time_01'].mean().values)
    try:
        edge_ds = get_cdr_sea_ice_edge_ds(mean_ds_date, hemisphere)
    except FileNotFoundError as e:
        if handle_missing_edge_ds == 'raise':
            raise e
        elif handle_missing_edge_ds == 'use_sic':
            import warnings
            warnings.warn(f"Sea ice edge dataset not found for {mean_ds_date}. Using CDR sea ice concentration dataset to define the edge instead.")
            #load the sea ice concentration dataset instead
            try:
                sic_ds = get_cdr_sea_ice_concentation_ds(mean_ds_date, hemisphere)
            except FileNotFoundError as e:
                warnings.warn(f"Sea ice concentration dataset not found for {mean_ds_date}. Using temporal interpolation between adjacent dates instead.")
                sic_ds = interpolate_sic_temporally(mean_ds_date, hemisphere)
                warnings.warn(f"Sea ice edge for {mean_ds_date} created using temporally interpolated sea ice concentration dataset. Please review whether this is expected for this date.")
                                     
            #convert the sic ds to a ice edge ds by classifying the pixels
            edge_ds = sic_to_edge_ds(sic_ds)
    
    for ts in measurement_frequency:

        #for each sentinel-3 data point, the the distance to where the sic < edge_sic_threshold
        nearest_class = nearest_neighbour_interpolation(edge_ds['xc'].values*1e3, edge_ds['yc'].values*1e3, edge_ds['ice_edge'].values, 
                                                         sentinel3_ds[f'xc_{ts}'].values, sentinel3_ds[f'yc_{ts}'].values)
        dists = np.full(nearest_class.shape, np.nan)

        #Where the nearest class  == 1 (we're in a region containing open water), we find the distance to the open water-ice boundary
        ow_s3 = sentinel3_ds.where(xr.DataArray(nearest_class, dims=f'time_{ts}') == 1 , drop=True)
        if ow_s3[f'time_{ts}'].size > 0:
            ice_edge = edge_ds.where(edge_ds.ice_edge > 1)
            edge_df = ice_edge.to_dataframe().dropna().reset_index()
            classification, dist_to_ice = nearest_neighbour_interpolation(edge_df['xc'].values*1e3, edge_df['yc'].values*1e3, 
                                                                        edge_df['ice_edge'].values, ow_s3[f'xc_{ts}'].values, 
                                                                        ow_s3[f'yc_{ts}'].values, return_dist=True)
            dists[nearest_class == 1] = dist_to_ice    
        
        #Where the nearest class > 1 (we're in a region containing ice), we find the distance to the ice-open water boundary
        ci_s3 = sentinel3_ds.where(xr.DataArray(nearest_class, dims=f'time_{ts}') > 1 , drop=True)
        if ci_s3[f'time_{ts}'].size > 0:
            ice_edge = edge_ds.where(edge_ds.ice_edge == 1)
            edge_df = ice_edge.to_dataframe().dropna().reset_index()
            classification, dists_to_water = nearest_neighbour_interpolation(edge_df['xc'].values*1e3, edge_df['yc'].values*1e3, 
                                                                            edge_df['ice_edge'].values, ci_s3[f'xc_{ts}'].values, 
                                                                            ci_s3[f'yc_{ts}'].values, return_dist=True)
            
            dists[nearest_class > 1] = dists_to_water

        sentinel3_ds[f'dist_to_ice_edge_{ts}'] = ((f'time_{ts}'), dists)
        sentinel3_ds[f'dist_to_ice_edge_{ts}'].attrs = {'units':'m', 'long_name':'Distance to ice edge (m)'}
        
    return sentinel3_ds

#========================================================================================================================#

def add_distance_to_coast(sentinel3_ds , hemisphere, measurement_frequency = ['01', '20_ku']):
    """
    Add the distance to the coast as a new variable in the Sentinel-3 dataset for the specified measurement frequencies.
    The distance to the coast is calculated as the distance to the nearest pixel classified as coast, land ice, iceshelves or disconnected ocean in the NSIDC sea ice regions dataset.
    
    Parameters:
    sentinel3_ds (xr.Dataset): The Sentinel-3 dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    measurement_frequency (list of str, optional): List of measurement frequencies to calculate the distance for. Defaults to ['01', '20_ku'].
        
    Returns:
    xr.Dataset: The Sentinel-3 dataset with the distance to the coast added, named 'dist_to_coast_{measurement_frequency}'.
    """
    
    if hemisphere == 'nh':
        regions_ds = xr.open_dataset(f'{get_local_directory_path("Surface_Masks")}/NSIDC-0780_SeaIceRegions_EASE2-N6.25km_v1.0.nc')
        mask_name = 'sea_ice_region_surface_mask'
    elif hemisphere == 'sh':
        regions_ds = xr.open_dataset(f'{get_local_directory_path("Surface_Masks")}/NSIDC-0780_SeaIceRegions_EASE2-S6.25km_v1.0.nc')
        mask_name = 'sea_ice_region_RH_surface_mask'
    
    coastal_codes = [30, 33, 34, 35] #codes for the coast, land ice, iceshelves and disconnected ocean
    coast = regions_ds[mask_name].where(regions_ds[mask_name].isin(coastal_codes))
    coast = coast.to_dataframe().dropna().reset_index()
    
    for ts in measurement_frequency:
        code, dist_coast = nearest_neighbour_interpolation(coast['x'].values, coast['y'].values, coast[mask_name].values, 
                                                           sentinel3_ds[f'xc_{ts}'].values, sentinel3_ds[f'yc_{ts}'].values, return_dist=True)
        sentinel3_ds[f'dist_to_coast_{ts}'] = ((f'time_{ts}'), dist_coast)
        sentinel3_ds[f'dist_to_coast_{ts}'].attrs = {'units':'m', 'long_name':'Distance to coast (m)'}
    return sentinel3_ds

#========================================================================================================================#

def interpolate_sic_temporally(target_date, hemisphere):
    """
    Linearly interpolates sea ice concentration dataset for a target date using datasets from the previous and next days.

    Parameters:
    target_date (pd.Timestamp): The date for which to interpolate the sea ice concentration dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    
    Returns:
    xr.Dataset: The interpolated sea ice concentration dataset for the target date.
    """
    prev_date = target_date - pd.Timedelta(days=1)
    next_date = target_date + pd.Timedelta(days=1)

    sic_ds_prev =  get_cdr_sea_ice_concentation_ds(prev_date, hemisphere)
    sic_ds_next =  get_cdr_sea_ice_concentation_ds(next_date, hemisphere)

    #combine along time dimension and interpolate
    combined_ds = xr.concat([sic_ds_prev, sic_ds_next], dim='time')
    combined_ds['time'] = [prev_date, next_date]

    return combined_ds.interp(time=target_date)

#========================================================================================================================#

def add_track_cdr_sea_ice_concentration(sentinel3_ds, hemisphere, measurement_frequency = ['01', '20_ku'], handle_missing_sic_ds='raise'):
    """
    Add the CDR/ICDR sea ice concentration as a new variable in the Sentinel-3 dataset for the specified measurement frequencies.
    The sea ice concentration is interpolated from the CDR/ICDR sea ice concentration dataset for the date corresponding to the mean date of the Sentinel-3 dataset, using linear interpolation.

    Parameters:
    sentinel3_ds (xr.Dataset): The Sentinel-3 dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    measurement_frequency (list of str, optional): List of measurement frequencies to calculate the distance for
    handle_missing_sic_ds (str, optional): How to handle missing sea ice concentration datasets. Options are 'raise' to raise a FileNotFoundError, 'temporal_interp' to use temporal interpolation between adjacent dates to estimate the sea ice concentration dataset for the missing date. Defaults to 'raise'.

    Returns:
    xr.Dataset: The Sentinel-3 dataset with the sea ice concentration added, named 'ice_conc_{measurement_frequency}'.
    """
    
    mean_ds_date = pd.to_datetime(sentinel3_ds['time_01'].mean().values)
    for ts in measurement_frequency:
        try:
            sic_ds = get_cdr_sea_ice_concentation_ds(mean_ds_date, hemisphere)
        except FileNotFoundError as e:
            if handle_missing_sic_ds == 'raise':
                raise e
            elif handle_missing_sic_ds == 'temporal_interp':
                import warnings
                warnings.warn(f"Sea ice concentration dataset not found for {mean_ds_date}. Using temporal interpolation between adjacent dates instead.")
                sic_ds = interpolate_sic_temporally(mean_ds_date, hemisphere)
                   
        sic = linear_interpolation(sic_ds['xc'].values*1e3, sic_ds['yc'].values*1e3, sic_ds['ice_conc'].values.squeeze(), 
                                    sentinel3_ds[f'xc_{ts}'].values, sentinel3_ds[f'yc_{ts}'].values)
        
        sentinel3_ds[f'ice_conc_{ts}'] = ((f'time_{ts}'), np.round(sic, 2))
        sentinel3_ds[f'ice_conc_{ts}'].attrs = sic_ds['ice_conc'].attrs

    return sentinel3_ds

#========================================================================================================================#

def add_track_sea_ice_type(sentinel3_ds, hemisphere, measurement_frequency = ['01', '20_ku']):
    """
    Add the sea ice type as a new variable in the Sentinel-3 dataset for the specified measurement frequencies.
    The sea ice type is interpolated from the OSISAF sea ice type dataset for the date corresponding to the mean date of the Sentinel-3 dataset, using nearest neighbour interpolation. 
    Both the ice type and confidence level are added to the Sentinel-3 dataset.
    
    Parameters:
    sentinel3_ds (xr.Dataset): The Sentinel-3 dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    measurement_frequency (list of str, optional): List of measurement frequencies to calculate the distance for. Defaults to ['01', '20_ku'].

    Returns:
    xr.Dataset: The Sentinel-3 dataset with the sea ice type and associated uncertainty added.

    """

    for ts in measurement_frequency:
        mean_ds_date = pd.to_datetime(sentinel3_ds[f'time_{ts}'].mean().values)
        type_source = 'osisaf'
        sitype_ds = get_sea_ice_type_ds(mean_ds_date, hemisphere, type_source)
        sitype = nearest_neighbour_interpolation(sitype_ds['xc_ease2'].values*1e3, sitype_ds['yc_ease2'].values*1e3, 
                                                    sitype_ds['ice_type'].values.squeeze(), sentinel3_ds[f'xc_{ts}'].values, 
                                                    sentinel3_ds[f'yc_{ts}'].values, max_dist=15e3)
        if 'confidence_level' in sitype_ds:
            sitype_confidence = nearest_neighbour_interpolation(sitype_ds['xc_ease2'].values*1e3, sitype_ds['yc_ease2'].values*1e3, 
                                                    sitype_ds['confidence_level'].values.squeeze(), sentinel3_ds[f'xc_{ts}'].values, 
                                                    sentinel3_ds[f'yc_{ts}'].values, max_dist=15e3)
            sentinel3_ds[f'sitype_confidence_{type_source}_{ts}'] = ((f'time_{ts}'), sitype_confidence)
            sentinel3_ds[f'sitype_confidence_{type_source}_{ts}'].attrs = {key: value for key, value in sitype_ds['confidence_level'].attrs.items() if key != 'grid_mapping'}
        elif 'uncertainty' in sitype_ds:
            sitype_uncertainty = nearest_neighbour_interpolation(sitype_ds['xc_ease2'].values*1e3, sitype_ds['yc_ease2'].values*1e3, 
                                                    sitype_ds['uncertainty'].values.squeeze(), sentinel3_ds[f'xc_{ts}'].values, 
                                                    sentinel3_ds[f'yc_{ts}'].values, max_dist=15e3)
            sentinel3_ds[f'sitype_uncertainty_{type_source}_{ts}'] = ((f'time_{ts}'), sitype_uncertainty)
            sentinel3_ds[f'sitype_uncertainty_{type_source}_{ts}'].attrs = {key: value for key, value in sitype_ds['uncertainty'].attrs.items() if key != 'grid_mapping'}
    
        sentinel3_ds[f'sitype_{type_source}_{ts}'] = ((f'time_{ts}'), sitype)
        sentinel3_ds[f'sitype_{type_source}_{ts}'].attrs = {key: value for key, value in sitype_ds['ice_type'].attrs.items() if key != 'grid_mapping'}
    return sentinel3_ds

#========================================================================================================================#

def add_track_nsidc_regions(sentinel3_ds, hemisphere, measurement_frequency = ['01', '20_ku']):
    """
    Adds the NSIDC sea ice regions as a new variable in the Sentinel-3 dataset for the specified measurement frequencies using nearest neighbour interpolation.

    Parameters:
    sentinel3_ds (xr.Dataset): The Sentinel-3 dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    measurement_frequency (list of str, optional): List of measurement frequencies to calculate the distance for. Defaults to ['01', '20_ku'].
    
    Returns:
    xr.Dataset: The Sentinel-3 dataset with the NSIDC sea ice regions added.
    
    """
    if hemisphere == 'nh':
        regions_ds = xr.open_dataset(f'{get_local_directory_path("Surface_Masks")}/NSIDC-0780_SeaIceRegions_EASE2-N6.25km_v1.0.nc')
        mask_name = 'sea_ice_region_surface_mask'
    elif hemisphere == 'sh':
        regions_ds = xr.open_dataset(f'{get_local_directory_path("Surface_Masks")}/NSIDC-0780_SeaIceRegions_EASE2-S6.25km_v1.0.nc')
        mask_name = 'sea_ice_region_RH_surface_mask'
    
    for ts in measurement_frequency:
        regions = regions_ds[mask_name].interp(x=sentinel3_ds[f'xc_{ts}'], y=sentinel3_ds[f'yc_{ts}'], method='nearest')
        sentinel3_ds[f'region_{ts}'] = ((f'time_{ts}'), regions.data)
        sentinel3_ds[f'region_{ts}'].attrs = regions_ds[mask_name].attrs
    del regions_ds
    return sentinel3_ds

#========================================================================================================================#

def interpolate_auxiliary_data_to_track(sentinel3_ds, hemisphere, measurement_frequency = ['01', '20_ku'], sea_ice_concentration=True,
                                         sea_ice_type=True, nsidc_region=True, distance_to_coast=True, distance_to_ice_edge=True, 
                                         handle_missing_sic_ds='raise', handle_missing_edge_ds='raise'):

    """
    Interpolates auxiliary data to the Sentinel-3 track.
    
    Parameters:
    sentinel3_ds (xr.Dataset): The Sentinel-3 dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    measurement_frequency (list of str, optional): List of measurement frequencies to calculate the distance for. Defaults to ['01', '20_ku'].
    sea_ice_concentration (bool, optional): Whether to interpolate sea ice concentration to the track. Defaults to True.
    sea_ice_type (bool, optional): Whether to interpolate sea ice type to the track. Defaults to True.
    nsidc_region (bool, optional): Whether to interpolate NSIDC sea ice regions to the track. Defaults to True.
    distance_to_coast (bool, optional): Whether to calculate the distance to the coast and add it to the track. Defaults to True.
    distance_to_ice_edge (bool, optional): Whether to calculate the distance to the ice edge and add it to the track. Defaults to True.
    handle_missing_sic_ds (str, optional): How to handle missing sea ice concentration datasets when interpolating sea ice concentration. Options are 'raise' to raise a FileNotFoundError, 'temporal_interp' to use temporal interpolation between adjacent dates. Defaults to 'raise'.
    handle_missing_edge_ds (str, optional): How to handle missing sea ice edge datasets when calculating distance to ice edge. Options are 'raise' to raise a FileNotFoundError, 'use_sic' to use the CDR sea ice concentration dataset to define the edge instead. Defaults to 'raise'.

    Returns:
    xr.Dataset: The Sentinel-3 dataset with the interpolated auxiliary data added as new variables
    """
    
    if sea_ice_concentration:
        sentinel3_ds = add_track_cdr_sea_ice_concentration(sentinel3_ds, hemisphere, measurement_frequency, handle_missing_sic_ds)

    if sea_ice_type:
        sentinel3_ds = add_track_sea_ice_type(sentinel3_ds, hemisphere, measurement_frequency)

    if nsidc_region:
        sentinel3_ds = add_track_nsidc_regions(sentinel3_ds, hemisphere, measurement_frequency)

    if distance_to_coast:
        sentinel3_ds = add_distance_to_coast(sentinel3_ds, hemisphere, measurement_frequency)

    if distance_to_ice_edge:
        sentinel3_ds = add_distance_to_ice_edge(sentinel3_ds, hemisphere, measurement_frequency, handle_missing_edge_ds)
    return sentinel3_ds

#========================================================================================================================#

def filter_sentinel3_dataset(sentinel3_ds, measurement_frequency=['01', '20_ku'], lat_range=[90, -90], min_dist_coast=None, nsidc_region=None, min_sic=None, max_t2m=None, tb_good_quality=True, drop=True):
    """
    Filters the Sentinel-3 dataset.

    Parameters:
    sentinel3_ds (xr.Dataset): The Sentinel-3 dataset to filter.
    measurement_frequency (list of str, optional): List of measurement frequencies to apply the filters to. Defaults to ['01', '20_ku'].
    lat_range (list of float, optional): The latitude range to filter the data to, in the format [max_latitude, min_latitude]. Defaults to [90, -90] - i.e., all latitudes.
    min_dist_coast (float, optional): The minimum distance to the coast in meters. If None, no filter is applied based on distance to the coast. Defaults to None.
    nsidc_region (list of int, optional): List of NSIDC sea ice region codes to filter the data to. If None, no filter is applied based on NSIDC region. Defaults to None.
    min_sic (float, optional): The minimum sea ice concentration to filter the data to. If None, no filter is applied based on sea ice concentration. Defaults to None.
    max_t2m (float, optional): The maximum 2m air temperature (K) to filter the data to. If None, no filter is applied based on 2m air temperature. Defaults to None.
    tb_good_quality (bool, optional): Whether to filter the data to only include points with good quality brightness temperatures (quality flag of 0 in the S3 L2 SI product for both 23.8 GHz and 36.5 GHz). Defaults to True.
    drop (bool, optional): Whether to drop the data points that do not meet the filter criteria. If False, the data points that do not meet the filter criteria will be kept but with NaN values for all variables. Defaults to True.

    Returns:
    xr.Dataset: The filtered Sentinel-3 dataset.
    """
    
    s3_ds = []

    for ts in measurement_frequency:
        freq_ds = sentinel3_ds[[var for var in sentinel3_ds.data_vars if ts in var]]

        filters = [(freq_ds[f'lat_{ts}'] >= min(lat_range)) & (freq_ds[f'lat_{ts}'] <= max(lat_range))]

        if ts == '01':
            if tb_good_quality:
                filters.append((freq_ds['tb_238_quality_flag_01'] == 0) & (freq_ds['tb_365_quality_flag_01'] == 0))
            if max_t2m is not None:
                filters.append(freq_ds[f't2m_01'] <= max_t2m)

        if min_dist_coast is not None:
            filters.append(freq_ds[f'dist_to_coast_{ts}'] >= min_dist_coast)
        if nsidc_region is not None:
            filters.append(freq_ds[f'region_{ts}'].isin(nsidc_region))
        if min_sic is not None:
            filters.append(freq_ds[f'ice_conc_{ts}'] >= min_sic)

        #combine all filters and apply them
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter &= f

        freq_ds = freq_ds.where(combined_filter, drop=drop)
        s3_ds.append(freq_ds)

    #merge the filtered datasets
    sentinel3_ds = xr.merge(s3_ds, join='inner')
    return sentinel3_ds

#========================================================================================================================#

def calculate_sic_corrected_tb(tb, sic, ow_tp):
    """
    Calculate the sea ice concentration corrected brightness temperature.
    
    Parameters:
    tb (array-like): The original brightness temperature to correct.
    sic (array-like): The sea ice concentration values between 0 and 1.
    ow_tp (float): The open water tie point brightness temperature.
    
    Returns:
    array-like: The sea ice concentration corrected brightness temperature.
    
    """

    if np.nanmax(sic) > 1.00001:
        sic = sic/100
    return (tb - (1-sic)*ow_tp)/sic

#========================================================================================================================#

def get_tp_ds(date, hemisphere, iteration=1):
    """
    Get the tie point dataset for a given date, hemisphere and iteration.
    
    Parameters:
    date (pd.Timestamp): The date for which to load the tie point dataset.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    iteration (int, optional): The iteration of the radiative transfer correction used in creating the Tbs. Use 1 for the standard tie points and 2 for the 'double difference' tie points. Defaults to 1.

    Returns:
    xr.Dataset: The tie point dataset.
    """
    tp_dir = os.path.join(get_local_directory_path("Tie_Points"), date.strftime('%Y/%m'))
    if iteration == 1:
        fname = f's3_mwr_tiepoints_{hemisphere}_{date.strftime("%Y%m%d")}.nc'
    elif iteration == 2: #'double difference' tie points
        fname = f's3_mwr_tiepoints_dd_{hemisphere}_{date.strftime("%Y%m%d")}.nc'
    else:
        raise ValueError("Invalid iteration value. Use 1 for standard tie points or 2 for double difference tie points.")
    tp_ds = xr.open_dataset(f'{tp_dir}/{fname}')
    return tp_ds

#========================================================================================================================#

def correct_tbs_for_ow(s3_ds, hemisphere, tp_iteration=1):
    """
    Correct theMWR brightness temperatures for surface open water contamination.

    Parameters:
    s3_ds (xr.Dataset): The Sentinel-3 dataset containing the brightness temperatures and sea ice concentration variables.
    hemisphere (str): 'nh' for Northern Hemisphere or 'sh' for Southern Hemisphere.
    tp_iteration (int, optional): The iteration of the radiative transfer correction used in creating the Tbs, which determines which tie points to use for the correction.
    
    Returns:
    xr.Dataset: The Sentinel-3 dataset with the corrected brightness temperatures added.
    """
    s3_date = pd.to_datetime(s3_ds['time_01'].mean().values).date()
    
    tp_ds = get_tp_ds(s3_date, hemisphere, iteration=tp_iteration)
    ow_tie_points = tp_ds.where(tp_ds['tp_type'] == 0, drop=True)

    for freq, name in zip(['238', '365'], ['23.8', '36.5']):
        ow_tp = np.nanmean(ow_tie_points[f'corr_tb_{freq}_01'].values)
        s3_ds[f'tb_{freq}_corrected_01'] = (('time_01'), calculate_sic_corrected_tb(s3_ds[f'corr_tb_{freq}_01'].values, s3_ds[f'ice_conc_01'].values, ow_tp))
        s3_ds[f'tb_{freq}_corrected_01'].attrs = {'units':'K', 'long_name':f'Atmospherically and SIC Corrected Brightness Temperature at {name} GHz'}
    return s3_ds  

#========================================================================================================================#

def calculate_gradient_ratio(tb_238, tb_365):
    """
    Calculate the gradient ratio between the 23.8 GHz and 36.5 GHz brightness temperatures.
    
    Parameters:
    tb_238 (array-like): The brightness temperature at 23.8 GHz.
    tb_365 (array-like): The brightness temperature at 36.5 GHz.
    
    Returns:
    array-like: The gradient ratio calculated as (tb_365 - tb_238)/(tb_238 + tb_365).
    """

    return (tb_365 - tb_238)/(tb_238 + tb_365)

#========================================================================================================================#

def get_oib_flight_df(fpath, min_snod_threshold = 0.00, min_n_atm = None):
    """
    Get a dataframe of OIB Quick Look flight data from a given txt file path.
    Quality control is applied and the lon/lat coordinates are transformed to EASE2 x/y coordinates.
    
    Parameters:
    fpath (str): The file path to the OIB Quick Look flight data text file.
    min_snod_threshold (float, optional): Minimum snow depth threshold to filter the data. Defaults to 0.00.
    min_n_atm (int, optional): Minimum number of ATM observations to filter the data. If None, no filtering will be applied based on the number of ATM observations. Defaults to None.
    
    Returns:
    pd.DataFrame: A dataframe containing the OIB flight data.

    """

    oib_flight_df = pd.read_csv(fpath, usecols = ['date','elapsed','lat','lon','thickness','n_atm', 'mean_fb', 'ATM_fb','snow_depth', 'surface_roughness'],
                                dtype={'elapsed':float,'lat':float,'lon':float,'thickness':float, 'n_atm':int, 'mean_fb':float, 
                                       'ATM_fb':float,'snow_depth':float, 'surface_roughness':float}, parse_dates=['date'])
    oib_flight_df['datetime'] = [row['date'] + pd.Timedelta(seconds=row['elapsed']) for i, row in oib_flight_df.iterrows()]
    oib_flight_df['lon'] =  Parallel(n_jobs=20, prefer="threads")(delayed(convert_lon_easting_format)(longitude) for longitude in oib_flight_df['lon'].values)
    oib_flight_df = oib_flight_df[oib_flight_df['snow_depth'] >= min_snod_threshold].reset_index(drop=True)
    if min_n_atm is not None:
        oib_flight_df = oib_flight_df[oib_flight_df['n_atm'] >= min_n_atm].reset_index(drop=True)
    oib_flight_df.loc[oib_flight_df['thickness'] < 0, 'thickness'] = np.nan  #set negative thickness values to nan
    oib_flight_df['xc'],oib_flight_df['yc'] = transform_coords(4326, 6931, oib_flight_df['lon'], oib_flight_df['lat'])
    oib_flight_df = oib_flight_df.drop(columns=['elapsed'])
    return oib_flight_df

#========================================================================================================================#

def get_season_start_yr(date):
    
    """
    Returns the starting year of the Arctic sea ice growth season for a given date.
    
    Parameters:
    date (pd.Timestamp): The date for which to determine the season start year.
    
    Returns:
    int: The starting year of the sea ice growth season.
    
    """
    
    if date.month in [8,9,10,11,12]:
        return date.year
    else:
        return date.year - 1
    
#========================================================================================================================#

def get_new_mallett_sden(date):
    """"
    Calculate the snow density using the Mallett (2025) method
    https://www.cambridge.org/core/journals/journal-of-glaciology/article/methodologically-robust-densification-function-for-snow-on-multiyear-arctic-sea-ice/430CCF225915658E8EEB9E5738595EC9
    
    Parameters:
    date (pd.Timestamp): The date for which the snow density is to be calculated

    Returns:
    float: The snow density in kg/m^3
    
    """
    season_start_yr = get_season_start_yr(date)
    season_aug_start = pd.to_datetime(f"{season_start_yr}-08-01")

    days_since_aug_first = (date - season_aug_start).days
    sden = (0.35*days_since_aug_first) + 239.78
    return sden

#========================================================================================================================#

def get_alexandrov_cice_density(ice_type, handle_ambigious='half_half'):
    """
    Get sea ice density based on the sea ice type using the Alexandrov et al. (2010) method.
    
    Parameters:
    ice_type (int): The sea ice type, where 2 is first year ice, 3 is multi-year ice and 4 is ambiguous.
    handle_ambigious (str, optional): How to handle ambiguous ice types. Options are 'half_half' to take the average of the first year and multi-year ice densities, or 'nan' to set the density to NaN for ambiguous ice types. Defaults to 'half_half'.
    
    Returns:
    float: The sea ice density in kg/m^3 based on the ice type.
    """
    
    if ice_type == 2:
        return 916.7
    if ice_type == 3:
        return 882.0
    if ice_type == 4:
        if handle_ambigious == 'half_half':
            return (916.7 + 882.0)/2
        if handle_ambigious == 'nan':
            return np.nan
    return np.nan

#========================================================================================================================#

def calc_corr_fb(fb, hs, sden):
    """
    Corrects the radar freeboard to give ice freeboard by accouting for reduced radar propogation speed in the snow layer.
    
    Parameters:
    fb (array-like): The original radar freeboard to correct.
    hs (array-like): The snow depth.
    sden (array-like): The snow density in kg/m^3.
    
    Returns:
    array-like: The ice freeboard.
    """
    # c = 299792458.
    # cs = c*(1. + 0.51*sden/1000)**-1.5
    # cf = c/cs -1
    # h_corr = hs*cf
    
    h_corr = hs*((1+(0.51*sden/1000))**1.5 - 1) 
    corr_fb = fb + h_corr
    return corr_fb

#========================================================================================================================#

def calc_sit(fb, hs, sden, iden):
    """
    Calculates sea ice thickness from radar freeboard, snow depth, snow density and ice density assuming hydrostatic equilibriumand.
    
    Parameters:
    fb (array-like): The sea ice freeboard (m).
    hs (array-like): The snow depth (m).
    sden (array-like): The snow density (kg/m^3).
    iden (array-like): The ice density in (kg/m^3).
    
    Returns:
    array-like: The sea ice thickness (m).
    """
    
    wden = 1023.9 # seawater density (Tilling, 2018 (Wadhams et al., 1992))
    sit = (fb*wden + hs*sden)/(wden-iden)
    return sit

#========================================================================================================================#

def set_axis_boundary_circular(ax):
    """
    Set the boundary of a matplotlib axis to be circular inplace.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axis to set the boundary for.
    """
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

#========================================================================================================================#

def add_map_cfeatures(ax):
    """
    Adds cartopy ocean, land and coastline to a matplotlib axis with a cartopy projection.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axis to add the features to.
    """
    ax.add_feature(cfeature.OCEAN, color = 'gray')
    ax.add_feature(cfeature.LAND, color='silver', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)

#========================================================================================================================#

def get_unique_dates_within_interval(central_dates, n_days):
    """
    Get unique dates within a specified interval around a list of dates.
        
    Parameters:
    central_dates (list of pd.Timestamp): The list of dates to get the interval around
    n_days (int): The number of days before and after each date to include in the interval.
    
    Returns:
    np.ndarray: An array of unique dates within the specified interval.

    """
    
    
    all_dates = []
    for date in central_dates:
        all_dates += [date + pd.DateOffset(days=i) for i in range(-n_days, n_days+1)]
    return np.unique(all_dates)

#========================================================================================================================#
