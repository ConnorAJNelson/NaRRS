"""
Author: Connor Nelson
Email: connor.nelson.21@ucl.ac.uk
Date: 2024-07-28
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="You will likely lose important projection information when converting to a PROJ string from another format.")
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import xarray as xr
import os
import verde as vd
from scipy.spatial import KDTree
import subprocess
import pyproj
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation


def set_axis_boundary_circular(ax):
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)


def add_map_cfeatures(ax):
    ax.add_feature(cfeature.OCEAN, color = 'silver')
    ax.add_feature(cfeature.LAND, color='gray', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)


class SeaIceDrift:
    def __init__(self):
        self.hemisphere = None
        self.drift_dataset = None
        self.concentration_dataset = None
        self.resampled_drift_dataset = None
        self.mv_time_resolution = None
        self.motion_vectors = None
        self.advection_results_df = None
        self.product_type = None

    def set_product_type(self, product_type):
        """
        Set the OSI SAF product type to be used (either '455' or '405'). 
        Note that the OSI SAF 455 product only covers the period from 1991-01-01 to 2020-12-31, while the OSI SAF 405 product covers the period from 2009-12-29 to 4 days ago.
        
        Parameters:
        product_type (str): The ice drift product type ('455' for reprocessed, '405' for NRT).
        
        """
        if product_type not in ['455', '405']:
            raise ValueError("Invalid product type. Choose '455' or '405'.")
        self.product_type = product_type


    @staticmethod
    def get_drift_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        Get a date range between two dates. An extra day is added to the start and an extra 2 days is added to the end
        such that the entire of the start and end dates are included in the drift dataset.
        Note, if start_date > end_date, the function swaps the range and applies the same logic.
        
        Parameters:
        start_date (pd.Timestamp): The starting datetime.
        end_date (pd.Timestamp): The ending datetime.
        
        Returns:
        pd.DatetimeIndex: A list of dates between the start and end dates.
        """
        if start_date > end_date:
            return pd.date_range(end_date - pd.Timedelta('1d'), start_date + pd.Timedelta('2d'))
        return pd.date_range(start_date - pd.Timedelta('1d'), end_date + pd.Timedelta('2d'))
    

    def download_motion_vectors(self, output_base_directory, start_date, end_date, hemisphere, skip_existing=True):
        """
        Download CDR low-resolution ice motion vector files from the Norwegian Meteorological Institute thredds server (https://thredds.met.no/thredds/osisaf/osisaf.html) using wget. 
        The files are saved in a directory structure based on the product type, year, and month.

        Parameters:
        output_base_directory (str): Path to the base directory where the files will be saved.
        start_date (pd.Timestamp): The starting datetime for downloading motion vectors.
        end_date (pd.Timestamp): The ending datetime for downloading motion vectors.
        hemisphere (str): The hemisphere of the ice drift files ('nh' or 'sh').
        skip_existing (bool): Whether to skip downloading files that already exist in the directory.
        """

        date_range = self.get_drift_date_range(start_date, end_date)

        if self.product_type == '455':
            if date_range[-1] > pd.to_datetime('2020-12-31') or date_range[0] < pd.to_datetime('1991-01-01'):
                raise ValueError('OSI455 Motion vectors only available from 1991-01-01 to 2020-12-31.')
        elif self.product_type == '405':
            if date_range[-1] > pd.to_datetime((datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d')) or date_range[0] < pd.to_datetime('2009-12-29'):
                raise ValueError('OSI405 Motion vectors only available from 2009-12-29 to 4 days ago.')
        else:
            raise ValueError("Invalid product type. Set product type as '455' or '405'.")

        for date in date_range:
            year = date.strftime('%Y')
            month = date.strftime('%m')
            dt_str = date.strftime('%Y%m%d1200')
            target_dir = os.path.join(output_base_directory, f"OSI{self.product_type}/{year}/{month}")
        
            if self.product_type == '455':
                product_name = f"ice_drift_{hemisphere}_ease2-750_cdr-v1p0_24h-{dt_str}.nc"
                url = f"https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/drift_455m_files/merged/{year}/{month}/{product_name}"

            elif self.product_type == '405':
                dti_str = (date - timedelta(days=2)).strftime('%Y%m%d1200')
                product_name = f"ice_drift_{hemisphere}_polstere-625_multi-oi_{dti_str}-{dt_str}.nc"
                url = f"https://thredds.met.no/thredds/fileServer/osisaf/met.no/ice/drift_lr/merged/{year}/{month}/{product_name}"

            if os.path.exists(os.path.join(target_dir, product_name)) and skip_existing:
                print(f'{product_name} exists in {target_dir}')
                continue
            
            os.makedirs(target_dir, exist_ok=True)
            cmd = f'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge -r --reject "index.html*" -nv -nc -np -nd -nH -e robots=off {url}'
            subprocess.run(f"cd {target_dir}; {cmd}", shell = True, check = True)
            print(f'{product_name} downloaded to {target_dir}')


    def load_motion_vectors(self, directory, start_date, end_date, hemisphere='nh'):
        """
        Load ice motion vectors from NetCDF files in the specified directory within the date range.

        Parameters:
        directory (str): Path to the directory containing NetCDF files with ice motion vectors.
        start_date (pd.Timestamp): The starting datetime for loading motion vectors.
        end_date (pd.Timestamp): The ending datetime for loading motion vectors.
        hemisphere (str): The hemisphere of the ice drift files ('nh' or 'sh').
        """

        if self.product_type == '455':
            self.load_motion_vectors_455(directory, start_date, end_date, hemisphere)
        elif self.product_type == '405':
            self.load_motion_vectors_405(directory, start_date, end_date, hemisphere)
        else:
            raise ValueError("Invalid product type. Set product type as '455' or '405' using set_product_type() method.")
        

    def load_motion_vectors_455(self, directory, start_date, end_date, hemisphere='nh'):
        """
        Load OSI-SAF 455 ice motion vectors covering the date range from NetCDF files in the specified directory.
        
        Parameters:
        directory (str): Path to the directory containing NetCDF files with ice motion vectors.
        start_date (pd.Timestamp): The starting datetime for loading motion vectors.
        end_date (pd.Timestamp): The ending datetime for loading motion vectors.
        hemisphere (str): The hemisphere of the ice drift files ('nh' or 'sh').
        time_resolution (str): The temporal resolution for resampling the motion vectors.
        """
        self.hemisphere = hemisphere
        date_range = self.get_drift_date_range(start_date, end_date)

        drift_ds = []
        for date in date_range:
            filename = f"ice_drift_{hemisphere}_ease2-750_cdr-v1p0_24h-{date.strftime('%Y%m%d')}1200.nc"
            file_path = os.path.join(directory, date.strftime('%Y/%m'), filename)
            try:
                ds = xr.open_dataset(file_path)
                ds['time'] = pd.to_datetime(ds['time'])
                ds['dX'] = ds['dX'] * 1e3 #convert from km to m
                ds['dY'] = ds['dY'] * 1e3
                ds['xc'] = ds['xc'] * 1e3
                ds['yc'] = ds['yc'] * 1e3
                drift_ds.append(ds)
            except FileNotFoundError:
                raise FileNotFoundError(f"File {file_path} not found.")
        drift_ds = xr.concat(drift_ds, dim='time')

        self.drift_dataset = drift_ds
        self.motion_vectors = drift_ds
        self.mv_time_resolution = '1d'


    def load_motion_vectors_405(self, directory, start_date, end_date, hemisphere='nh'):
        """
        Load OSI-SAF 405 ice motion vectors covering the date range from NetCDF files in the specified directory. 
        The motion vectors are resampled to daily time steps by averaging the overlapping 48 hour motion vectors.
        
        Parameters:
        directory (str): Path to the directory containing NetCDF files with ice motion vectors.
        start_date (pd.Timestamp): The starting datetime for loading motion vectors.
        end_date (pd.Timestamp): The ending datetime for loading motion vectors.
        hemisphere (str): The hemisphere of the ice drift files ('nh' or 'sh').
        """

        self.hemisphere = hemisphere
        date_range = self.get_drift_date_range(start_date, end_date)

        drift_ds = []
        for date in date_range:
            dt_str = date.strftime('%Y%m%d1200')
            dt_str_start = (date - timedelta(days=2)).strftime('%Y%m%d1200')
            filename = f"ice_drift_{hemisphere}_polstere-625_multi-oi_{dt_str_start}-{dt_str}.nc"
            file_path = os.path.join(directory, date.strftime('%Y/%m'), filename)
            try:
                ds = xr.open_dataset(file_path)
                ds['time'] = pd.to_datetime(ds['time'])

                xcs, ycs = self.transform_lonlat_to_ease2(ds['lon'], ds['lat'], hemisphere)
                ds['xc'] =  (('yc', 'xc'), xcs)
                ds['yc'] =  (('yc', 'xc'), ycs)
        
                lon1, lat1 = ds['lon1'].squeeze(), ds['lat1'].squeeze()
                xc1, yc1 = self.transform_lonlat_to_ease2(lon1, lat1, hemisphere)
                ds['xc1'] = (('time', 'yc', 'xc'), xc1[np.newaxis, ...])
                ds['yc1'] = (('time', 'yc', 'xc'), yc1[np.newaxis, ...])
                            
                ds['dX'] = (('time', 'yc', 'xc'), xc1.squeeze() - ds['xc'].values[np.newaxis, ...])
                ds['dY'] = (('time', 'yc', 'xc'), yc1.squeeze() - ds['yc'].values[np.newaxis, ...])

                drift_ds.append(ds)
            except FileNotFoundError:
                raise FileNotFoundError(f"File {file_path} not found.")
            
        drift_ds = xr.concat(drift_ds, dim='time')
        drift_ds = self.resample_osi405_to_daily(drift_ds)

        print(f'Motion vectors loaded from {start_date} to {end_date}')
        
        self.drift_dataset = drift_ds
        self.motion_vectors = drift_ds
        self.mv_time_resolution = '1d'

    
    def resample_osi405_to_daily(self, drift_ds):
        """
        Resample OSI405 motion vectors to daily time steps by averaging the overalpping 48 hour motion vectors.

        Parameters:
        drift_ds (xr.Dataset): Dataset containing OSI405 motion vectors with 48 hour motion vectors.
        
        Returns: 
        xr.Dataset: Dataset containing OSI405 motion vectors resampled to daily motion vectors. 
        
        Note, The first and last time steps are dropped.   
        """

        drift_ds_resampled = drift_ds.copy()
        drift_ds_resampled['dX'] = 0.5 * (drift_ds['dX'][:-1]/2 + drift_ds['dX'][1:]/2)
        drift_ds_resampled['dY'] = 0.5 * (drift_ds['dY'][:-1]/2 + drift_ds['dY'][1:]/2)

        #recalculate xc1 and yc1
        drift_ds_resampled['xc1'] = drift_ds_resampled['dX'] + drift_ds_resampled['xc']
        drift_ds_resampled['yc1'] = drift_ds_resampled['dY'] + drift_ds_resampled['yc']

        #recalculate lon1 and lat1
        lon1, lat1 = self.transform_ease2_to_lonlat(drift_ds_resampled['xc1'], drift_ds_resampled['yc1'], self.hemisphere)
        drift_ds_resampled['lon1'] = (('time', 'yc', 'xc'), lon1)
        drift_ds_resampled['lat1'] = (('time', 'yc', 'xc'), lat1)

        #drop the first and last time step from the resampled motion vectors as they are not valid
        drift_ds_resampled = drift_ds_resampled.isel(time=slice(1, -1))
        #drop invalid variables
        drift_ds_resampled = drift_ds_resampled.drop_vars(['Polar_Stereographic_Grid','time_bnds','dt0', 'dt1', 'status_flag', 'uncert_dX_and_dY'])

        return drift_ds_resampled


    def download_osisaf_cdr_sic(self, output_base_directory, start_date, end_date, hemisphere='nh', skip_existing=True, handle_download_error='ignore'):
        """
        Download OSI SAF CDR and ICDR sea ice concentration data from the Norwegian Meteorological Institute thredds server (https://thredds.met.no/thredds/osisaf/osisaf.html) using wget.
        
        Parameters:
        output_base_directory (str): Path to the base directory where the files will be saved.
        start_date (pd.Timestamp): The starting datetime for downloading sea ice concentration data.
        end_date (pd.Timestamp): The ending datetime for downloading sea ice concentration data.
        hemisphere (str): The hemisphere of the sea ice concentration files ('nh' or 'sh').
        skip_existing (bool): Whether to skip downloading files that already exist in the directory.
        handle_download_error (str): How to handle download errors ('ignore' to ignore errors and continue downloading, 'raise' to raise the error and stop downloading).
        
        """
    
        date_range = self.get_drift_date_range(start_date, end_date)
        
        for date in date_range:
            year = date.strftime('%Y')
            month = date.strftime('%m')
            dt_str = date.strftime('%Y%m%d1200')

            #if available, download Global Sea Ice Concentration climate data record (SMMR/SSMI/SSMIS), release 3, OSI-450-a
            if date < pd.to_datetime('2021-01-01'):
                target_dir = os.path.join(output_base_directory, f"CDR/{year}/{month}")
                product_name = f'ice_conc_{hemisphere}_ease2-250_cdr-v3p0_{dt_str}.nc'
                url = f'https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/conc_450a_files/{year}/{month}/{product_name}'
            
            #else, download Global Sea Ice Concentration interim climate data record, release 3, OSI-430-a
            else:
                target_dir = os.path.join(output_base_directory, f"ICDR/{year}/{month}")
                product_name = f'ice_conc_{hemisphere}_ease2-250_icdr-v3p0_{dt_str}.nc'
                url = f'https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/conc_cra_files/{year}/{month}/{product_name}'

            if os.path.exists(os.path.join(target_dir, product_name)) and skip_existing:
                print(f'{product_name} exists in {target_dir}')
                continue
            
            os.makedirs(target_dir, exist_ok=True)
            cmd = f'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge -r --reject "index.html*" -nv -nc -np -nd -nH -e robots=off {url}'
            try:
                subprocess.run(f"cd {target_dir}; {cmd}", shell = True, check = True)
            except subprocess.CalledProcessError as e:
                if handle_download_error == 'ignore':
                    print(f"Error downloading {product_name}: {e}")
                    print("Ignoring error and continuing.")
                    continue
                elif handle_download_error == 'raise':
                    raise e

    def load_concentration_dataset(self, directory, start_date, end_date, hemisphere='nh', handle_missing_dates='raise'):
        """
        Load sea ice concentration data covering the date range from NetCDF files in the specified directory.
        
        Parameters:
        directory (str): Path to the directory containing NetCDF files with sea ice concentration data.
        start_date: The starting datetime for loading concentration data.
        end_date: The ending datetime for loading concentration data.
        hemisphere (str): The hemisphere of the ice drift files ('nh' or 'sh').
        handle_missing_dates (str): How to handle missing dates in the dataset ('raise' to raise an error, 'interpolate' to interpolate the missing dates).    
        """
        
        self.hemisphere = hemisphere
        date_range = self.get_drift_date_range(start_date, end_date)

        conc_ds = []
        missing_dates = []
        for date in date_range:
            if date < pd.to_datetime('2021-01-01'):
                filename = f"ice_conc_{hemisphere}_ease2-250_cdr-v3p0_{date.strftime('%Y%m%d')}1200.nc"
                fpath = os.path.join(directory,'CDR', date.strftime('%Y/%m'), filename)
            else:
                filename = f"ice_conc_{hemisphere}_ease2-250_icdr-v3p0_{date.strftime('%Y%m%d')}1200.nc"
                fpath = os.path.join(directory,'ICDR', date.strftime('%Y/%m'), filename)

            try:
                ds = xr.open_dataset(fpath)
                ds['time'] = pd.to_datetime(ds['time'])
                ds['xc'] = ds['xc']*1e3
                ds['yc'] = ds['yc']*1e3
                conc_ds.append(ds)
            except FileNotFoundError:
                if handle_missing_dates == 'raise':
                    raise FileNotFoundError(f"File {fpath} not found.")
                elif handle_missing_dates == 'interpolate':
                    missing_dates.append(date + pd.Timedelta(hours=12))
                    continue
        if len(conc_ds) == 0:
            raise FileNotFoundError(f"No files found in {directory} for the specified date range.")
        
        conc_ds = xr.concat(conc_ds, dim='time')
        conc_ds = conc_ds.drop('time_bnds')

        if len(missing_dates) > 0:
            print(f"Files not found for the following dates: {missing_dates}")
            print(f"Handling missing dates by interpolating the concentration dataset.")
            conc_ds = self._handle_missing_dates(conc_ds, missing_dates)

        self.concentration_dataset = conc_ds
        print(f'Sea ice concentration loaded from {start_date} to {end_date}')


    def _handle_missing_dates(self, ds, missing_dates, max_gap=pd.Timedelta('5d')):
        """
        Handles missing dates in the concentration dataset by adding the missing dates to the dataset and filling the values with linear interpolation.
        
        Parameters:
        ds (xr.Dataset): The concentration dataset with missing dates.
        missing_dates (list): A list of missing dates to add to the dataset.
        max_gap (pd.Timedelta): The maximum gap to fill with interpolation. If the gap between existing dates is larger than this, the missing values will not be filled and will remain as NaN.
        
        Returns:
        xr.Dataset: The updated concentration dataset including the missing dates with values filled by linear interpolation where possible.
        """

        missing_dates_ds = ds.isel(time=slice(0, 1)).copy()
        missing_dates_ds = xr.merge([missing_dates_ds.assign_coords(time=[date]) for date in missing_dates])
        for var in missing_dates_ds.data_vars:
            missing_dates_ds[var][:] = np.nan #initially set the values for the missing dates to NaN
        ds = xr.concat([ds, missing_dates_ds], dim='time')
        ds = ds.sortby('time')
        ds = ds.interpolate_na(dim='time', method='linear', max_gap=max_gap) #fill the missing values with linear interpolation
        return ds


    def add_sic_to_motion_vectors_dataset(self):
        """
        Resamples the loaded sea ice concentration dataset to the same spatial resolution as the drift dataset and adds it to the drift dataset.

        Parameters:
        drift_dataset (xr.Dataset): The drift dataset to which to add the sea ice concentration data.
        
        Returns:
        xr.Dataset: The drift dataset with the sea ice concentration added following spatio-temporal bilinear interpolation.
        """
        if self.concentration_dataset is None:
            raise ValueError('No concentration dataset loaded to add to drift dataset. Use load_concentration_dataset() method.')
        drift_dataset = self.motion_vectors

        drift_datetimes = drift_dataset.time.values
        sic = self.concentration_dataset['ice_conc'].interp(time=drift_datetimes, method='linear') #resample the concentration dataset to the same time steps as the drift dataset using linear interpolation.
       
        nans = np.isnan(sic.values) #need to keep the nans nan (eg where there is land)
        sic = sic.where(sic >= 0.0, 0.0).where(sic <= 100.0, 100.0) #ensure SIC is between 0.0 and 100.0
        sic.values[nans] = np.nan #reapply the nans

        def interpolate_sic_to_drift_grid(sic_dataset, time):
            drift_grid_xcs, drift_grid_ycs = np.meshgrid(drift_dataset.xc.values, drift_dataset.yc.values)
            return self.linear_interpolation(sic_dataset.xc.values, sic_dataset.yc.values, sic_dataset.sel(time=time).values,
                                            drift_grid_xcs, drift_grid_ycs, max_dist=25e3)
        
        #spatially resample the concentration dataset to the same grid as the drift dataset in parallel for each time step
        resampled_sic = Parallel(n_jobs=-1)(delayed(interpolate_sic_to_drift_grid)(sic, time) for time in drift_dataset.time.values)
        resampled_sic = np.stack(resampled_sic)

        drift_dataset['ice_conc'] = (('time', 'yc', 'xc'), resampled_sic) #add SIC to the drift dataset
        drift_dataset['ice_conc'].attrs = {'long_name': ' Sea ice ice concentration', 'units': '%', 'description': 'Sea ice concentration interpolated from the OSI SAF dataset'}
        
        self.motion_vectors = drift_dataset
        print('Sea ice concentration added to motion vector dataset.')
        
    
    def resample_motion_vectors(self, resample_time=False, time_resolution='1h', resample_spatial=False, ease2_resolution=25e3, min_lat=None, max_lat=None, spatial_resampling_method='linear', max_spatial_interpolation_dist=75e3):
        """
        Resamples the loaded motion vectors to the specified time resolution and spatial resolution.

        Parameters:
        resample_time (bool): Whether to resample the motion vectors to the specified time resolution.
        time_resolution (str): The time resolution for resampling the motion vectors (default '1h' for hourly).
        resample_spatial (bool): Whether to resample the motion vectors to the specified spatial resolution.
        hemisphere (str): The hemisphere of the ice drift files ('nh' or 'sh').
        ease2_resolution (float): The resolution of the EASE2 grid (default 25km).
        min_lat (float): The minimum latitude of the target grid for resampling the motion vectors.
        max_lat (float): The maximum latitude of the target grid for resampling the motion vectors.
        spatial_resampling_method (str): The method for resampling the motion vectors ('nearest' or 'linear').
        max_spatial_interpolation_dist (float): The maximum distance from the grid point to interpolate the motion vector.
        
        
        Returns:
        xr.Dataset: A resampled Dataset of the motion vectors.
        """
        if self.drift_dataset is None:
            raise ValueError('No drift dataset loaded to resample.')

        if resample_time:
            self.mv_time_resolution = time_resolution
            resampled_drift_dataset = self._resample_motion_vectors_time(mv_time_resolution=time_resolution)
            self.motion_vectors = resampled_drift_dataset
        
        if resample_spatial:
            resampled_drift_dataset = self._resample_motion_vectors_spatial(self.hemisphere, ease2_resolution, min_lat, max_lat, spatial_resampling_method, max_spatial_interpolation_dist)
            self.motion_vectors = resampled_drift_dataset
        
        self.resampled_drift_dataset = resampled_drift_dataset
        print('Motion vectors successfully resampled.')


    def _resample_motion_vectors_time(self, mv_time_resolution):
        """
        Resamples the loaded motion vectors to the specified interval.
        
        Parameters:
        mv_time_resolution (str): The time resolution for resampling the motion vectors.
        
        Returns:
        xr.Dataset: A resampled Dataset of the motion vectors with the specified time resolution.
        """

        #Each input time represents the drift from the previous day to the current day (24 hours)
        #Therefore, we need to calculate the drift per time resolution requested

        if pd.Timedelta(mv_time_resolution) > pd.Timedelta('1d'): #ensure that the time resolution is one day or less as the input data files are drift across 1 day
            raise ValueError('Time resolution must be one day or less.')
        
        motion_vectors = self.motion_vectors
        time_res_per_day = pd.Timedelta('1d') / pd.Timedelta(mv_time_resolution)
        motion_vectors['dX_per_time_res'] = motion_vectors['dX'] / time_res_per_day
        motion_vectors['dY_per_time_res'] = motion_vectors['dY'] / time_res_per_day

        new_time_range = pd.date_range(motion_vectors['time'].values.min()-pd.Timedelta(days=1), motion_vectors['time'].values.max(), freq=mv_time_resolution)
        new_time_range = (new_time_range + pd.Timedelta(mv_time_resolution))[:-1]
        next_midday =  [self.find_next_midday(dt) for dt in new_time_range]

        #create a new dataset with the full time range at the requested time resolution
        resampled_motion_vectors = xr.Dataset({'time': ('time', new_time_range), 'xc': motion_vectors['xc'], 'yc': motion_vectors['yc']})
        resampled_motion_vectors['time'].attrs = {'long_name': 'time', 'standard_name': 'time', 'description': f'End time of drift over the last {mv_time_resolution}'}
        resampled_motion_vectors['dX'] = (('time', 'yc', 'xc'), np.zeros((len(new_time_range), len(motion_vectors['yc']), len(motion_vectors['xc']))))
        resampled_motion_vectors['dY'] = (('time', 'yc', 'xc'), np.zeros((len(new_time_range), len(motion_vectors['yc']), len(motion_vectors['xc']))))

        for i, dt in enumerate(new_time_range):
            motion_vectors_slice = motion_vectors.sel(time=next_midday[i])
            resampled_motion_vectors['dX'][i] = motion_vectors_slice['dX_per_time_res']
            resampled_motion_vectors['dY'][i] = motion_vectors_slice['dY_per_time_res']
                        
        return resampled_motion_vectors
    

    def _resample_motion_vectors_spatial(self, hemisphere, ease2_resolution, min_lat=None, max_lat=None, interpolation_method='nearest', max_dist=75e3):
        """
        Resamples the loaded motion vectors to the specified spatial resolution by interpolating the motion vectors to a new EASE2 grid with the specified resolution.
        
        Parameters:
        hemisphere (str): The hemisphere of the ice drift files ('nh' or 'sh').
        ease2_resolution (float): The resolution of the EASE2 grid (default 25km).
        min_lat (float): The minimum latitude of the target grid for resampling the motion vectors.
        max_lat (float): The maximum latitude of the target grid for resampling the motion vectors.
        interpolation_method (str): The method for resampling the motion vectors ('nearest' or 'linear').
        max_dist (float): The maximum distance from the grid point to interpolate the motion vector. If the distance from the grid point is greater than this, the interpolated value will be set to NaN.
        
        Returns:
        xr.Dataset: A resampled Dataset of the motion vectors with the specified spatial resolution.
        """
    
        target_grid = self.get_target_ease2_xarray(hemisphere, resolution=ease2_resolution, min_lat=min_lat, max_lat=max_lat)
        target_xcs, target_ycs = np.meshgrid(target_grid.xc.values, target_grid.yc.values)
        
        def perform_spatial_resampling(self, time, target_xcs, target_ycs, interpolation_method, max_dist):
            if interpolation_method == 'nearest':
                resampled_dx = self.nearest_neighbour_interpolation(self.motion_vectors.xc.values, self.motion_vectors.yc.values, self.motion_vectors.sel(time=time).dX.values,
                                                                                target_xcs, target_ycs, max_dist=max_dist)
                resampled_dy = self.nearest_neighbour_interpolation(self.motion_vectors.xc.values, self.motion_vectors.yc.values, self.motion_vectors.sel(time=time).dY.values,
                                                                                target_xcs, target_ycs, max_dist=max_dist)
            elif interpolation_method == 'linear':
                resampled_dx = self.linear_interpolation(self.motion_vectors.xc.values, self.motion_vectors.yc.values, self.motion_vectors.sel(time=time).dX.values,
                                                                                target_xcs, target_ycs, max_dist=max_dist)
                resampled_dy = self.linear_interpolation(self.motion_vectors.xc.values, self.motion_vectors.yc.values, self.motion_vectors.sel(time=time).dY.values,
                                                                                target_xcs, target_ycs, max_dist=max_dist)
            else:
                raise ValueError('Invalid interpolation method. Choose from "nearest" or "linear".')
            return resampled_dx, resampled_dy
        
        #run the spatial resampling in parallel
        dxs, dys = zip(*Parallel(n_jobs=-1)(delayed(perform_spatial_resampling)(self, time, target_xcs, target_ycs, interpolation_method, max_dist) for time in self.motion_vectors.time.values))

        resampled_dx = np.stack(dxs)
        resampled_dy = np.stack(dys)
        
        #create a new dataset with the resampled motion vectors on the target grid
        resampled_motion_vectors = xr.Dataset({'time': self.motion_vectors.time, 'xc': target_grid.xc, 'yc': target_grid.yc, 
                                              'lon': (('yc', 'xc'), target_grid.lon.data), 'lat': (('yc', 'xc'), target_grid.lat.data),
                                              'dX': (('time', 'yc', 'xc'), resampled_dx), 'dY': (('time', 'yc', 'xc'), resampled_dy)})

        return resampled_motion_vectors
    

    def set_missing_ice_drift_to_zero(self, concentration_threshold=95):
        """
        Set missing ice drift values to zero where ice concentration is greater than or equal to the specified threshold.
        
        Parameters:
        concentration_threshold: The ice concentration threshold above which to set missing ice drift values to zero.
        """

        if 'ice_conc' not in self.motion_vectors.data_vars:
            self.add_sic_to_motion_vectors_dataset()

        #set dX and dY to 0 where they are missing and ice concentration is >= threshold
        self.motion_vectors['dX'] = xr.where((self.motion_vectors['dX'].isnull()) & (self.motion_vectors['ice_conc'] >= concentration_threshold), 0, self.motion_vectors['dX'])
        self.motion_vectors['dY'] = xr.where((self.motion_vectors['dY'].isnull()) & (self.motion_vectors['ice_conc'] >= concentration_threshold), 0, self.motion_vectors['dY'])
        print(f'Missing ice drift values set to zero where ice concentration >= { concentration_threshold}%')

    
    def reset_motion_vectors(self):
        """
        Resets the motion vectors to the original dataset.
        """
        self.motion_vectors = self.drift_dataset
        self.mv_time_resolution = '1d'
        self.resampled_drift_dataset = None 


    @staticmethod
    def find_next_midday(dt):
        """
        Finds the next midday datetime from a given datetime.

        Parameters:
        dt (pd.Timestamp): The datetime from which to find the next midday.
        
        Returns:
        pd.Timestamp: The next midday datetime.
        """

        if dt.hour <= 12:
            return dt.replace(hour=12, minute=0, second=0).round('h')
        else:
            return dt.replace(hour=12, minute=0, second=0).round('h') + pd.Timedelta(days=1)


    @staticmethod
    def nearest_neighbour_interpolation(grid_xcs, grid_ycs, values, xcs, ycs, max_dist=75e3):
        """
        Interpolates values to the given coordinates using nearest neighbour interpolation.

        Parameters:
        grid_xcs (np.ndarray): The x-coordinates of the grid.
        grid_ycs (np.ndarray): The y-coordinates of the grid.
        values (np.ndarray): The values to interpolate.
        xcs (np.ndarray): The x-coordinates at which to interpolate the values.
        ycs (np.ndarray): The y-coordinates at which to interpolate the values.
        max_dist (float, optional): The maximum distance from the grid point to interpolate the values. Defaults to 75e3.
        
        Returns:
        float or np.ndarray: The interpolated values.
        """
        if np.shape(grid_xcs) != np.shape(values):
            grid_xcs, grid_ycs = np.meshgrid(grid_xcs, grid_ycs)

        #if the xcs and ycs are 2d arrays, flatten them
        grid=False
        if len(xcs.shape) == 2:
            grid=True
            grid_shape = xcs.shape
            xcs = xcs.flatten()
            ycs = ycs.flatten()

        if not isinstance(xcs, (list, np.ndarray)):
            xcs = [xcs]
            ycs = [ycs]
        coords = list(zip(xcs, ycs))   

        tree = KDTree(np.c_[grid_xcs.ravel(),grid_ycs.ravel()])

        dist, idx = tree.query(coords, k=1)
        nearest_values = values.ravel()[idx].astype(float)

        if max_dist is not None:
            nearest_values[dist > max_dist] = np.nan #set values to nan where the distance from the grid point is greater than the maximum distance
        if len(nearest_values) == 1:
            return nearest_values[0]
        
        if grid:
            return nearest_values.reshape(grid_shape)
        return nearest_values


    @staticmethod
    def linear_interpolation(grid_xcs, grid_ycs, values, xcs, ycs, max_dist=75e3):
        """
        Linearly interpolates values to the given coordinates.

        Parameters:
        grid_xcs (np.ndarray): The x-coordinates of the grid.
        grid_ycs (np.ndarray): The y-coordinates of the grid.
        values (np.ndarray): The values to interpolate.
        xcs (np.ndarray): The x-coordinates at which to interpolate the values.
        ycs (np.ndarray): The y-coordinates at which to interpolate the values.
        max_dist (float, optional): The maximum distance from the grid point to interpolate the values. Defaults to 75e3 m.
        
        Returns:
        np.ndarray: The interpolated values.
        """

        if np.shape(grid_xcs) != np.shape(values):
            grid_xcs, grid_ycs = np.meshgrid(grid_xcs, grid_ycs)

        interpolator = vd.Linear().fit((grid_xcs, grid_ycs), values)
        interped_points = interpolator.predict((xcs, ycs))

        if max_dist is not None:
            distance_mask = vd.distance_mask((grid_xcs, grid_ycs), maxdist=max_dist, coordinates=(xcs, ycs)) #get the coordinates of the non nan values
            interped_points[~distance_mask] = np.nan

        return interped_points


    @staticmethod
    def get_ease2_grid_info_df():
        """
        Returns a pd.DataFrame with information about the possible EASE2 grid configurations.
        """
        
        resolutions = [3e3, 3e3, 3.125e3, 3.125e3, 6.25e3, 6.25e3, 9e3, 9e3, 12.5e3, 12.5e3, 25e3, 25e3, 36e3, 36e3]
        hemispheres = ['nh', 'sh']*int((len(resolutions)/2))
        epsg_codes = [6931, 6932]*int((len(resolutions)/2))
        nrows = [6000, 6000, 5760, 5760, 2880, 2880, 2000, 2000, 1440, 1440, 720, 720, 500, 500]
        ncols = [6000, 6000, 5760, 5760, 2880, 2880, 2000, 2000, 1440, 1440, 720, 720, 500, 500]
        upper_left_corner_x = [-9000000.0]*len(resolutions)
        upper_left_corner_y = [9000000.0]*len(resolutions)
        info_df = pd.DataFrame({'resolution': resolutions, 'hemisphere': hemispheres, 'epsg_code': epsg_codes, 'nrows': nrows, 'ncols': ncols, 'upper_left_corner_x': upper_left_corner_x, 'upper_left_corner_y': upper_left_corner_y})
        return info_df
    

    def get_target_ease2_xarray(self, hemisphere, resolution=25e3, min_lat=None, max_lat=None):
        """
        Creates an xarray dataset for the target EASE2 grid.

        Parameters:
        hemisphere (str): The hemisphere of the grid ('nh' or 'sh').
        resolution (float): The resolution of the grid (default 25km).
        min_lat (float): The minimum latitude for the grid.
        max_lat (float): The maximum latitude for the grid.
        
        Returns:
        xr.Dataset: The target EASE2 grid.
        """

        info_df = self.get_ease2_grid_info_df()
        target_info = info_df.loc[(info_df['hemisphere'] == hemisphere) & (info_df['resolution'] == resolution)]
        
        x1 = target_info['upper_left_corner_x'].values[0]
        y1 = target_info['upper_left_corner_y'].values[0]
        x2 = x1 + (target_info['ncols'].values[0] * resolution)
        y2 = y1 - (target_info['nrows'].values[0] * resolution)

        grid = vd.grid_coordinates(region=[x1,x2,y2,y1], shape=(target_info['nrows'].values[0], target_info['ncols'].values[0]), pixel_register=True, meshgrid=False)
        grid_xr = vd.make_xarray_grid((grid[0], grid[1]), dims=['yc','xc'], data=np.arange((target_info['nrows'].values[0]*target_info['ncols'].values[0])).reshape((target_info['nrows'].values[0], target_info['ncols'].values[0])), data_names='grid_index')
        grid_xr.xc.attrs = {'long_name':f'x coordinate', 'description':f'x coordinate in Lambert Azimuthal Equal Area (espg:{target_info["epsg_code"].values[0]}) projection', 'units':'m'}
        grid_xr.yc.attrs = {'long_name':f'y coordinate', 'description':f'y coordinate in Lambert Azimuthal Equal Area (espg:{target_info["epsg_code"].values[0]}) projection', 'units':'m'}

        x_2d, y_2d = np.meshgrid(grid_xr.xc, grid_xr.yc)
        lon, lat = self.transform_ease2_to_lonlat(x_2d, y_2d, hemisphere)
        grid_xr['lon'] = (('yc', 'xc'), lon)
        grid_xr['lat'] = (('yc', 'xc'), lat)
        grid_xr.lon.attrs = {'long_name':f'longitude', 'description':f'longitude in WGS84 projection', 'units':'degrees'}
        grid_xr.lat.attrs = {'long_name':f'latitude', 'description':f'latitude in WGS84 projection', 'units':'degrees'}
        grid_xr = grid_xr.set_coords(['lon', 'lat'])

        if min_lat is not None:
            grid_xr = grid_xr.where(grid_xr.lat >= min_lat, drop=True)
        if max_lat is not None:
            grid_xr = grid_xr.where(grid_xr.lat <= max_lat, drop=True)

        return grid_xr
    

    def interpolate_motion_vectors(self, datetime, xcs, ycs, method):
        """
        Spatial interpolation of the the motion vectors at a given datetime.

        Parameters:
        datetime (pd.Timestamp): The datetime for which to interpolate the motion vector.
        xcs (np.ndarray): The x-coordinates of the parcels.
        ycs (np.ndarray): The y-coordinates of the parcels.
        method (str): The interpolation method ('nearest' or 'linear').
        
        Returns:
        tuple: Interpolated dX and dY np.ndarrays for the provided coordinates.
        """

        dX = np.full_like(xcs, np.nan, dtype=np.float64)
        dY = np.full_like(ycs, np.nan, dtype=np.float64)

        valid_indices = ~np.isnan(xcs) & ~np.isnan(ycs) #valid data mask

        if np.any(valid_indices):
            valid_xcs = xcs[valid_indices]
            valid_ycs = ycs[valid_indices]

            datetime_drift = self.motion_vectors.sel(time=datetime)

            if method == 'nearest':
                dX_valid = self.nearest_neighbour_interpolation(datetime_drift.xc.values, datetime_drift.yc.values, datetime_drift.dX.values, valid_xcs, valid_ycs)
                dY_valid = self.nearest_neighbour_interpolation(datetime_drift.xc.values, datetime_drift.yc.values, datetime_drift.dY.values, valid_xcs, valid_ycs)
            elif method == 'linear':
                dX_valid = self.linear_interpolation(datetime_drift.xc.values, datetime_drift.yc.values, datetime_drift.dX.values, valid_xcs, valid_ycs)
                dY_valid = self.linear_interpolation(datetime_drift.xc.values, datetime_drift.yc.values, datetime_drift.dY.values, valid_xcs, valid_ycs)
            else:
                raise ValueError('Invalid interpolation method. Choose from "nearest" or "linear".')

            #assign the interpolations to the appropriate indices
            dX[valid_indices] = dX_valid
            dY[valid_indices] = dY_valid

        return dX, dY

    

    def interpolate_sea_ice_concentration(self, datetime, xcs, ycs, method, max_dist=50e3):
        """
        Interpolates the sea ice concentration at a given datetime.
        
        Parameters:
        datetime (pd.Timestamp): The datetime for which to interpolate the sea ice concentration.
        xcs (np.ndarray): The x-coordinates of the parcels.
        ycs (np.ndarray): The y-coordinates of the parcels.
        method (str): The interpolation method ('nearest' or 'linear').
        max_dist (float): The maximum distance from the grid point to interpolate the sea ice concentration.
        
        Returns:
        np.ndarray: Interpolated sea ice concentration
        """

        sic = np.full_like(xcs, np.nan, dtype=np.float64)
        valid_indices = ~np.isnan(xcs) & ~np.isnan(ycs)

        if np.any(valid_indices):
            valid_xcs = xcs[valid_indices]
            valid_ycs = ycs[valid_indices]

            #check that the drift dataset has sea ice concentration data
            if 'ice_conc' not in self.concentration_dataset.data_vars:
                raise ValueError('No sea ice concentration data available to interpolate.\n Please load concentration dataset using load_concentration_dataset() method and add to drift dataset using add_sic_to_motion_vectors_dataset() method.')

            datetime_concentration = self.motion_vectors.sel(time=datetime)

            if method == 'nearest':
                sic_valid = self.nearest_neighbour_interpolation(datetime_concentration.xc.values, datetime_concentration.yc.values, datetime_concentration.ice_conc.values, valid_xcs, valid_ycs, max_dist)
            elif method == 'linear':
                sic_valid = self.linear_interpolation(datetime_concentration.xc.values, datetime_concentration.yc.values, datetime_concentration.ice_conc.values, valid_xcs, valid_ycs, max_dist)
            else:
                raise ValueError('Invalid interpolation method. Choose from "nearest" or "linear".')
            
            sic[valid_indices] = sic_valid

        return sic


    @staticmethod
    def transform_ease2_to_lonlat(x, y, hemisphere, reverse=False):
        """
        Transform coordinates from EASE2 to lon/lat.

        Parameters:
        x (float, np.ndarray, list): The x-coordinate(s) in EASE2. 
        y (float, np.ndarray, list): The y-coordinate(s) in EASE2.
        hemisphere (str): The hemisphere of the coordinates ('nh' or 'sh').
        
        Returns:
        tuple: The transformed lon and lat coordinates.

        """
        if hemisphere == 'nh':
            ease2_epsg = 6931
        elif hemisphere == 'sh':
            ease2_epsg = 6932
        else:
            raise ValueError('Invalid hemisphere. Choose from "nh" or "sh".')

        if reverse:
            raise ValueError('Reverse transformation deprecated. Use transform_lonlat_to_ease2() method.')
        return pyproj.Transformer.from_crs(ease2_epsg, 4326, always_xy=True).transform(x, y)
    
    
    @staticmethod
    def transform_lonlat_to_ease2(lon, lat, hemisphere):
        """
        Transform coordinates from lon/lat to EASE2.

        Parameters:
        lon (float, np.ndarray, list): The longitude coordinate(s).
        lat (float, np.ndarray, list): The latitude coordinate(s).
        hemisphere (str): The hemisphere of the coordinates ('nh' or 'sh').
        
        Returns:
        tuple: The transformed EASE2 x and y coordinates.

        """
        if hemisphere == 'nh':
            ease2_epsg = 6931
        elif hemisphere == 'sh':
            ease2_epsg = 6932
        else:
            raise ValueError('Invalid hemisphere. Choose from "nh" or "sh".')
        
        return pyproj.Transformer.from_crs(4326, ease2_epsg, always_xy=True).transform(lon, lat)
    
    
    def simulate_advection(self, datetimes, xcs, ycs, datetime_bound1=None, datetime_bound2=None, parcel_ids=None, spatial_interpolation='nearest', results_time_resolution=None, include_sic=False, njobs=-1):
        """
        Simulates advection of ice parcels.
        
        Parameters:
        datetimes (list, np.ndarray or pd.Series): Starting datetime objects the ice parcels.
        xcs (list, np.ndarray or pd.Series): x-coordinates of the ice parcels.
        ycs (list, np.ndarray or pd.Series): y-coordinates of the ice parcels.
        datetime_bound1 (datetime, optional): The first datetime bound for the advection simulation. If not provided, no lower bound is applied.
        datetime_bound2 (datetime, optional): The second datetime bound for the advection simulation. If not provided, no upper bound is applied.
        parcel_ids (list, optional): The IDs of the ice parcels. If not provided, IDs are automatically assigned.
        spatial_interpolation (str): The interpolation method for the motion vectors ('nearest' or 'linear').
        results_time_resolution (str): The time resolution for the results (e.g., '1d', '1h').
        include_sic (bool): Whether to include sea ice concentration data in the results. Default is False.
        njobs (int): The number of parallel jobs to run.
        
        Returns:
        pd.DataFrame with columns ['start_coords', 'parcel_id', 'datetime', 'x', 'y'] showing the coordinates at each interval.
            - 'start_coords' shows the initial coordinates and datetime for each parcel in the format '(x, y, datetime)'. 
            - If include_sic is True, an additional column 'ice_conc' is included showing the interpolated sea ice concentration at each point.
        """

        if len(datetimes) != len(xcs) or len(datetimes) != len(ycs):
            raise ValueError('Input datetimes, xcs, ycs must be of the same length')
        
        #if the time resolution is a multiple of days, floor the datetimes to the nearest day and add 12 hours
        if self.mv_time_resolution[-1].lower() == 'd':
            datetimes = pd.to_datetime(datetimes).floor('d') + pd.Timedelta(hours=12)
            if datetime_bound1 is not None:
                datetime_bound1 = pd.to_datetime(datetime_bound1).floor('d') + pd.Timedelta(hours=12)
            if datetime_bound2 is not None:
                datetime_bound2 = pd.to_datetime(datetime_bound2).floor('d') + pd.Timedelta(hours=12)
        else:
            datetimes = pd.to_datetime(datetimes).round(self.mv_time_resolution)
            if datetime_bound1 is not None:
                datetime_bound1 = pd.to_datetime(datetime_bound1).round(self.mv_time_resolution)
            if datetime_bound2 is not None:
                datetime_bound2 = pd.to_datetime(datetime_bound2).round(self.mv_time_resolution)

        if parcel_ids is None:  #if parcel ids are not provided, create them
            parcel_ids = np.arange(len(xcs))
            print('Parcel IDs not provided. Assigning unique IDs to parcels.')
        df = pd.DataFrame({'datetime': datetimes, 'x': xcs, 'y': ycs, 'parcel_id': parcel_ids})
        
        #group by unique datetimes to minimise redundant interpolation operations
        grouped = df.groupby('datetime')

        #process each datetime group in parallel
        results = Parallel(n_jobs=njobs)(
            delayed(self._process_datetime_group)(datetime, group, datetime_bound1, datetime_bound2, spatial_interpolation)
            for datetime, group in grouped)
        
        result_df = self._convert_advection_results_list_to_dataframe(results)

        if include_sic:
            result_df = self._add_sea_ice_concentration(result_df, spatial_interpolation)

        if results_time_resolution is not None:
            result_df = self._resample_results(result_df, time_resolution=results_time_resolution, hemisphere=self.hemisphere, include_sic=include_sic)
            
        result_df = result_df.sort_values(by=['parcel_id','datetime'])
        self.advection_results_df = result_df
        return result_df


    def _process_datetime_group(self, datetime, group, datetime_bound1, datetime_bound2, spatial_interpolation):
        """
        Process the advection of a group of parcels that share the same datetime.

        Parameters:
        datetime (pd.Timestamp): The shared datetime for this group.
        group (pd.DataFrame): DataFrame group subset containing the initial x, y coordinates and parcel IDs for the parcels in this group.
        datetime_bound1: The first datetime bound for the advection simulation.
        datetime_bound2: The second datetime bound for the advection simulation.
        spatial_interpolation: The interpolation method for the motion vectors ('nearest' or 'linear').
        
        Returns:
        pd.DataFrame: DataFrame with coordinates for each group member over time.
        """
        initial_xcs = group['x'].values
        initial_ycs = group['y'].values
        parcel_ids = group['parcel_id'].values

        coord_results = []

        #save the initial position of each parcel
        coord_ids = [f'({xc}, {yc}, {datetime})' for xc, yc in zip(initial_xcs, initial_ycs)]
        for xc, yc, coord_id, parcel_id in zip(initial_xcs, initial_ycs, coord_ids, parcel_ids):
            coord_results.append({'datetime': datetime, 'x': xc, 'y': yc, 'start_coords': coord_id, 'parcel_id': parcel_id})
        
        #get the bounds that are not none
        datetime_bounds = [dt for dt in [datetime_bound1, datetime_bound2] if dt is not None]
        
        #forward/backward advection logic
        for datetime_bound in datetime_bounds:
            current_datetime = datetime
            xcs = initial_xcs.copy()
            ycs = initial_ycs.copy()

            try:
                time_direction = self.get_time_direction(current_datetime, datetime_bound)
            except ValueError:
                continue

            while (current_datetime < datetime_bound and time_direction == 'forwards') or \
                (current_datetime > datetime_bound and time_direction == 'backwards'):
                
                #move to the next or previous time step
                if time_direction == 'forwards':
                    current_datetime += pd.to_timedelta(self.mv_time_resolution)
                else:
                    current_datetime -= pd.to_timedelta(self.mv_time_resolution)
                
                #obtain the relevant motion vectors for all parcels at this timestep via spatial interpolation from the motion vectors dataset
                dXs, dYs = self.interpolate_motion_vectors(current_datetime, xcs, ycs, spatial_interpolation)

                #update the positions of all parcels
                if time_direction == 'forwards':
                    xcs += dXs
                    ycs += dYs
                else:
                    xcs -= dXs
                    ycs -= dYs

                #store the results for this timestep
                for xc, yc, coord_id, parcel_id in zip(xcs, ycs, coord_ids, parcel_ids):
                    coord_results.append({'datetime': current_datetime, 'x': xc, 'y': yc, 'start_coords': coord_id, 'parcel_id': parcel_id})

        return pd.DataFrame(coord_results)


    @staticmethod
    def get_time_direction(start_datetime, end_datetime):
        """
        Determines the direction of time between two datetimes.
        
        Parameters:
        start_datetime (pd.Timestamp): The starting datetime.
        end_datetime (pd.Timestamp): The ending datetime.
        
        Returns:
        str: indication of the direction of time ('forwards' or 'backwards').
        """
        if start_datetime < end_datetime:
            return 'forwards'
        elif start_datetime > end_datetime:
            return 'backwards'
        else:
            raise ValueError('Start and end datetimes are the same.')
    

    def _convert_advection_results_list_to_dataframe(self, advection_results):
        """
        Converts a list of advection results (DataFrames) to a single DataFrame.
        
        Parameters:
        advection_results (list): list of pd.DataFrames containing advection results following advection processing by datetime group.
        
        Returns:
        pd.DataFrame: A single DataFrame containing all advection results with columns ['start_coords', 'parcel_id', 'datetime', 'x', 'y'] and, if include_sic is True, an additional column 'ice_conc' for sea ice concentration.
        """
        
        result_df = pd.concat(advection_results).reset_index(drop=True)
        result_df['datetime'] = pd.to_datetime(result_df['datetime'])
        result_df['x'] = result_df['x'].astype(float)
        result_df['y'] = result_df['y'].astype(float)
        result_df['lon'], result_df['lat'] = self.transform_ease2_to_lonlat(result_df['x'], result_df['y'], self.hemisphere)
        result_df = result_df.set_index(['start_coords', 'datetime']).sort_index()
        return result_df

    
    def _add_sea_ice_concentration(self, result_df, spatial_interpolation):
        """
        Adds sea ice concentration to the advected coordinates by interpolating after advection.
        
        Parameters:
        result_df (pd.DataFrame): DataFrame with advected coordinates.
        spatial_interpolation (str): The interpolation method for the sea ice concentration ('nearest' or 'linear').
        
        Returns:
        pd.DataFrame: DataFrame updated to include interpolated sea ice concentration values in a new column 'ice_conc'.
        """
        grouped = result_df.groupby('datetime')
        sic_results = Parallel(n_jobs=-1)(
            delayed(self._interpolate_sic_for_group)(datetime, group, spatial_interpolation)
            for datetime, group in grouped
        )

        updated_results = pd.concat(sic_results).reset_index()
        updated_results.set_index(['start_coords', 'datetime'], inplace=True)
        #ensure that the index follows the same order as the original result_df
        updated_results = updated_results.reindex(result_df.index)
        
        return updated_results


    def _interpolate_sic_for_group(self, datetime, group, spatial_interpolation):
        """
        Interpolates sea ice concentration for a specific datetime group.
        
        Parameters:
        datetime (pd.Timestamp): The shared datetime for the group.
        group (pd.DataFrame): Group of parcels with their x, y coordinates.
        spatial_interpolation (str): The interpolation method for sea ice concentration ('nearest' or 'linear').
        
        Returns:
        pd.DataFrame: Group DataFrame with interpolated sea ice concentration.
        """
        xcs = group['x'].values
        ycs = group['y'].values
        sic_values = self.interpolate_sea_ice_concentration(datetime, xcs, ycs, method=spatial_interpolation)
        group['ice_conc'] = sic_values
        return group


    def _resample_results(self, result_df, time_resolution, hemisphere, include_sic=False):
        """
        Resamples the results DataFrame to the specified time resolution using linear interpolation.

        Parameters:
        result_df (pd.DataFrame): A DataFrame containing the results of the parcel trajectories.
        time_resolution (str): The time resolution for resampling the results (e.g., '1d' for daily, '1h' for hourly).
        hemisphere (str): The hemisphere of the ice drift files ('nh' or 'sh').
        include_sic (bool): Whether to include sea ice concentration in the resampled results.
        
        Returns: 
        pd.DataFrame: DataFrame with the results resampled to the specified time resolution.
        """

        #check if the frequency is already the same as the time resolution (we don't need to resample if it is)
        # if pd.Timedelta(time_resolution) == self._find_datetime_frequency(result_df.index.get_level_values('datetime')):
        if self.mv_time_resolution == time_resolution:
            return result_df

        def resample_group(start_coords, group, time_resolution, transform_func, hemisphere):

            if time_resolution[-1] == 'd' or time_resolution[-1] == 'D':
                offset = pd.Timedelta(hours=12)
            else:
                offset = None

            group.set_index('datetime', inplace=True)
            cols_to_resample = ['x', 'y', 'parcel_id'] if not include_sic else ['x', 'y', 'ice_conc', 'parcel_id']
            group = group[cols_to_resample].resample(time_resolution, offset=offset).interpolate(method='linear')
            group['lon'], group['lat'] = transform_func(group['x'], group['y'], hemisphere)
            group['start_coords'] = start_coords
            group['parcel_id'] = group['parcel_id'].astype(int)
            return group.reset_index()
        
        result_df = result_df.reset_index()
        groups = result_df.groupby('start_coords')
        
        resampled_results = Parallel(n_jobs=-1)(delayed(resample_group)(sc, group, time_resolution, self.transform_ease2_to_lonlat, hemisphere) for sc, group in groups)
        
        result_df = pd.concat(resampled_results)
        result_df = result_df.set_index(['start_coords', 'datetime'])

        return result_df
    

    def _find_datetime_frequency(self, datetimes):
        """
        Checks that the datetimes are of the same frequency and returns that frequency.
        
        Parameters:
        datetimes(np.ndarray, pd.Series or list): A list of datetimes.

        Returns:
        pd.Timedelta: The frequency of the datetimes.
        """
        if (datetimes[1] - datetimes[0]) == (datetimes[-1] - datetimes[-2]):
            return datetimes[1] - datetimes[0]
        else:
            raise ValueError('Datetimes are not of the same frequency.')
        
        
    def get_coords_at_datetime(self, result_df, datetime):
        """
        Retrieves the coordinates of parcels at a specific datetime from the results DataFrame.
        
        Parameters:
        result_df (pd.DataFrame): A DataFrame containing the results of the parcel trajectories.
        datetime (pd.Timestamp): The datetime at which to get the coordinates.

        Returns:
        pd.DataFrame: A DataFrame with the coordinates of parcels at the specified datetime.
        """
        return result_df.xs(datetime, level='datetime')


    def plot_parcel_trajectories(self, result_df, hemisphere, ax=None, projection=None, every_n_parcel=None):
        """
        Plots the trajectories of parcels on a map.
        
        Parameters:
        result_df (pd.DataFrame): A DataFrame containing the results of the parcel trajectories.
        hemisphere (str): The hemisphere of the map ('nh' or 'sh').
        ax (matplotlib.axes.Axes, optional): The axis on which to plot the trajectories.If not provided, a new figure and axis will be created.
        projection (cartopy.crs.CRS, optional): The projection of the map.
        every_n_parcel (int, optional): The interval at which to plot parcels. If not provided, all parcels will be plotted.

        Returns:
        matplotlib.axes.Axes: The axis with the plotted trajectories.
        """

        if ax is None:
            fig = plt.figure(figsize=(7.5,7.5))
            ax = fig.add_subplot(111, projection=projection)
            ax = self.format_map_starting_ax(ax, result_df, projection)

        if every_n_parcel is not None:
            start_coords = result_df.reset_index().dropna(subset=['x','y',])['start_coords'].unique()
            start_coords = start_coords[::every_n_parcel]
            result_df = result_df.reset_index().loc[result_df.reset_index()['start_coords'].isin(start_coords)].set_index(['start_coords', 'datetime'])
            ax.set_title(f'Advection of every {every_n_parcel}th datapoint')

        if projection is None:
            transform = ax.transData
        else:
            transform = ccrs.epsg(self.get_hemisphere_epsg_code(hemisphere))

        for start_coords, group in result_df.groupby('start_coords'):
            ax.plot(group['x'], group['y'], label=start_coords, transform=transform)
            ax.quiver(group['x'], group['y'], group['x'].diff(), group['y'].diff(), scale=1, scale_units='xy', angles='xy', color='r', transform=transform)
        return ax
    

    def plot_track_start_and_end_coords(self, result_df, hemisphere, ax=None, projection=None, plot_style='scatter'):
        """
        Plots the start and end coordinates of parcels on a map.
        
        Parameters:
        result_df (pd.DataFrame): A DataFrame containing the results of the parcel trajectories.
        hemisphere (str): The hemisphere of the map ('nh' or 'sh').
        ax (matplotlib.axes.Axes, optional): The axis on which to plot the trajectories. If not provided, a new figure and axis will be created.
        projection (cartopy.crs.CRS, optional): The projection of the map.
        plot_style (str, optional): The style of plotting ('scatter' or 'line'). Default is 'scatter'.

        Returns:
        matplotlib.axes.Axes: The axis with the plotted trajectories.
        """
        if ax is None:
            fig = plt.figure(figsize=(7.5,7.5))
            ax = fig.add_subplot(111, projection=projection)
            ax = self.format_map_starting_ax(ax, result_df, projection)

        if projection is None:
            transform = ax.transData
        else:
            transform = ccrs.epsg(self.get_hemisphere_epsg_code(hemisphere))
        start_coords = result_df.groupby('start_coords').first()
        end_coords = result_df.groupby('start_coords').last()
        if plot_style == 'scatter':
            ax.scatter(start_coords['x'], start_coords['y'], color='r', label='Start', transform=transform, s=2)
            ax.scatter(end_coords['x'], end_coords['y'],  color='b', label='End', transform=transform, s=2)
        elif plot_style == 'line':
            ax.plot(start_coords['x'], start_coords['y'], color='r', label='Start', transform=transform)
            ax.plot(end_coords['x'], end_coords['y'],  color='b', label='End', transform=transform)
        ax.legend()
        return ax
    

    def plot_track_at_time(self, result_df, datetime, hemisphere, ax=None, projection=None, plot_style='scatter'):
        """
        Plots the track of parcels at a specific time.
        
        Parameters:
        result_df (pd.DataFrame): A DataFrame containing the results of the parcel trajectories.
        datetime (datetime): The datetime at which to plot the track.
        hemisphere (str): The hemisphere of the map ('nh' or 'sh').
        ax (matplotlib.axes.Axes, optional): The axis on which to plot the trajectories. If not provided, a new figure and axis will be created.
        projection (cartopy.crs.CRS, optional): The projection of the map.
        plot_style (str, optional): The style of plotting ('scatter' or 'line'). Default is 'scatter'.

        Returns:
        matplotlib.axes.Axes: The axis with the plotted trajectories.
        """
        if ax is None:
            fig = plt.figure(figsize=(7.5,7.5))
            ax = fig.add_subplot(111, projection=projection)
            ax = self.format_map_starting_ax(ax, result_df, projection)

        if projection is None:
            transform = ax.transData
        else:
            transform = ccrs.epsg(self.get_hemisphere_epsg_code(hemisphere))
        selected_results = result_df.xs(datetime, level='datetime')
        if plot_style == 'scatter':
            ax.scatter(selected_results['x'], selected_results['y'], transform=transform, s=2)
        elif plot_style == 'line':
            ax.plot(selected_results['x'], selected_results['y'], transform=transform)
        ax.set_title(datetime)
        return ax
    

    def animate_track_ice_drift(self, result_df, hemisphere, projection=None, plot_style='scatter'):
        """
        Animates the track of ice drift over time.
        
        Parameters:
        result_df (pd.DataFrame): A DataFrame containing the results of the parcel trajectories.
        hemisphere (str): The hemisphere of the ice drift ('nh' or 'sh').
        projection (cartopy.crs.CRS, optional): The projection of the map.
        plot_style (str, optional): The style of plotting ('scatter' or 'line'). Default is 'scatter'.

        Returns:
        matplotlib.animation.FuncAnimation: The animation of the ice drift.
        """
        
        def update(frame, ax, result_df, projection):
            ax.clear()
            ax = self.format_map_starting_ax(ax, result_df, projection)
            ax = self.plot_track_at_time(result_df, frame, hemisphere, ax=ax, projection=projection, plot_style=plot_style)
            return ax.artists
        
        fig = plt.figure(figsize=(7.5,7.5))
        ax = fig.add_subplot(111, projection=projection)
        ax = self.format_map_starting_ax(ax, result_df, projection)
        ax.set_title(' ')
        plt.tight_layout()
        
        ani = FuncAnimation(fig, update, frames=result_df.index.get_level_values('datetime').unique(), fargs=(ax, result_df, projection),blit=True)
        plt.close()
        return ani
    

    @staticmethod
    def format_map_starting_ax(ax, result_df, projection):
        if projection is not None:
            ax.set_extent([-180, 180, 62, 90], ccrs.PlateCarree())
            SeaIceDrift.set_axis_boundary_circular(ax)
            SeaIceDrift.add_map_cfeatures(ax)
        else:
            ax.set_xlim(result_df['x'].min(), result_df['x'].max())
            ax.set_ylim(result_df['y'].min(), result_df['y'].max())
        return ax

    @staticmethod
    def get_hemisphere_epsg_code(hemisphere):
        if hemisphere == 'nh':
            return 6931
        elif hemisphere == 'sh':
            return 6932
        else:
            raise ValueError('Invalid hemisphere. Choose from "nh" or "sh".')
    