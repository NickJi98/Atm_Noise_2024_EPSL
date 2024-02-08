#!/usr/bin/env python3

"""
Functions for general analysis: Tropical cyclone info & Seismic stations

Author: Qing Ji
"""

# Load python packages
import os, re
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from scipy.interpolate import interp1d

# Obspy client
client_default = Client("IRIS")

# Tropical cyclone database
base_dir = '/Users/JQ/Desktop/Hurricane_Wind/Noise_2024_EPSL'
tc_dataset = nc.Dataset(os.path.join(base_dir, 'Data/IBTrACS.ALL.v04r00.nc'))


# Search TC in the dataset
def get_tc_info(year, name=None, number=None, agency='USA'):
    if (name is None) and (number is None):
        raise ValueError('Either TC name or number for that year \
                          must be provided!')
    if (name is not None) and (number is not None):
        raise Warning('Receive both TC name and number for that year. \
                       Only searching for TC name!')

    # All TC in the search year
    year_mask = (tc_dataset['season'][:] == year)
    tc_search_list = []
    for obj in tc_dataset['name'][year_mask, :]:
        tc_name = "".join(obj[~obj.mask].astype('str'))
        tc_search_list.append(tc_name)

    # Array index of target TC name
    if (name is not None):
        try:
            tc_ind = np.flatnonzero(year_mask)[tc_search_list.index(name.upper())]
        except Exception:
            raise ValueError('No TC matches the name: %s %d' % (name, year))

    # Array index of target TC number in that year
    else:
        tc_ind = np.flatnonzero(year_mask)[number-1]

    # Create TC info dict
    tc_info = {}

    # ISO time
    time_arr = np.array([])
    for obj in tc_dataset['iso_time'][tc_ind]:
        tc_time = "".join(obj[~obj.mask].astype('str'))
        if tc_time != "":
            time_arr = np.append(time_arr,
                                 UTCDateTime(tc_time, precision=3).datetime)
    tc_info['time'] = time_arr

    # TC track
    if agency == 'USA':
        tc_info['lon'] = tc_dataset['usa_lon'][tc_ind].compressed()
        tc_info['lat'] = tc_dataset['usa_lat'][tc_ind].compressed()

    # Hurricane scale
        tc_info['scale'] = tc_dataset['usa_sshs'][tc_ind].compressed()
    # Max sustained wind speed [m/s]
        tc_info['sus_wind'] = tc_dataset['usa_wind'][tc_ind].compressed() * 0.514
    # Min sea level pressure [kPa]
        tc_info['pres'] = tc_dataset['usa_pres'][tc_ind].compressed() * 0.1

    # Radius of max winds [km] (NOT re-analyzed)
        tc_info['rmw'] = tc_dataset['usa_rmw'][tc_ind].compressed() * 1.852
    # Eye diameter [km] (NOT re-analyzed)
        tc_info['eye'] = tc_dataset['usa_eye'][tc_ind].compressed() * 1.852

    else:
        tc_info['lon'] = tc_dataset['lon'][tc_ind].compressed()
        tc_info['lat'] = tc_dataset['lat'][tc_ind].compressed()

        nature_arr = np.array([])
        for obj in tc_dataset['nature'][tc_ind]:
            nature = "".join(obj[~obj.mask].astype('str'))
            if nature != "":
                nature_arr = np.append(nature_arr, nature)
        tc_info['nature'] = nature_arr

        tc_info['scale'] = tc_dataset['usa_sshs'][tc_ind].compressed()
        tc_info['sus_wind'] = tc_dataset['wmo_wind'][tc_ind].compressed() * 0.514
        tc_info['pres'] = tc_dataset['wmo_pres'][tc_ind].compressed() * 0.1

    return tc_info


# Get interpolation function of TC track
def get_interp_func(tc_info):
    
    t0 = tc_info['time'][0]
    t_rel = np.array([(obj-t0).total_seconds() for obj in tc_info['time']])
    f_lon = interp1d(t_rel, tc_info['lon'], bounds_error=False)
    f_lat = interp1d(t_rel, tc_info['lat'], bounds_error=False)
    
    return [f_lon, f_lat], t0


# Interpolate TC track
def interp_track(time_range, f_loc, t0):
    
    # Convert to UTCDateTime
    t0 = UTCDateTime(t0)
    
    # Timestamp to evaluate TC locations
    tc_reftime = np.arange(0, time_range[1]-time_range[0], 60*60, dtype=int) \
        + (time_range[0] - t0)
    tc_timestamp = [(t0 + obj).datetime for obj in tc_reftime]
    
    # Interpolate TC locations
    tc_lon, tc_lat = f_loc[0](tc_reftime), f_loc[1](tc_reftime)
    
    return tc_timestamp, tc_lon, tc_lat


# Plot TC track (and seismic stations)
def plot_tc_track(tc_info, extent=None, category=False, sta_info=None,
                  client=None, plot_sta=False, plot_stnm=False, aspect=None):

    # IRIS client
    if client is None:
        client = client_default

    # Map region
    if extent is None:
        margin_in_deg = [10, 5]
        extent = [np.min(tc_info['lon'])-margin_in_deg[0],
                  np.max(tc_info['lon'])+margin_in_deg[0],
                  np.min(tc_info['lat'])-margin_in_deg[1],
                  np.max(tc_info['lat'])+margin_in_deg[1]]

    # Map projection
    proj = ccrs.Mercator()
    proj_data = ccrs.PlateCarree()

    fig = plt.figure()
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj_data)
    ax.add_feature(cfeature.STATES, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE)

    # Plot TC track
    ax.plot(tc_info['lon'], tc_info['lat'], 'b-', transform=proj_data)

    if category is False:
        ax.scatter(tc_info['lon'], tc_info['lat'], s=60, c='blue', marker='o',
                   transform=proj_data, label='Hurricane Track')
    else:
        ax.scatter(tc_info['lon'][tc_info['scale'] >= 1],
                   tc_info['lat'][tc_info['scale'] >= 1],
                   s=60, c='orange', marker='o',
                   transform=proj_data, label='Hurricane')
        ax.scatter(tc_info['lon'][tc_info['scale'] == 0],
                   tc_info['lat'][tc_info['scale'] == 0],
                   s=60, c='green', marker='o',
                   transform=proj_data, label='Tropical Storm')
        ax.scatter(tc_info['lon'][tc_info['scale'] == -1],
                   tc_info['lat'][tc_info['scale'] == -1],
                   s=60, c='blue', marker='o',
                   transform=proj_data, label='Tropical Depression')
        ax.scatter(tc_info['lon'][tc_info['scale'] < -1],
                   tc_info['lat'][tc_info['scale'] < -1],
                   s=60, c='gray', marker='o', transform=proj_data)

    # Plot stations
    if plot_sta:
        if sta_info is None:
            print('Download seismic station information ...')
            sta_inv = client.get_stations(starttime=UTCDateTime(tc_info['time'][0]),
                                          endtime=UTCDateTime(tc_info['time'][-1]),
                                          minlongitude=extent[0],
                                          maxlongitude=extent[1],
                                          minlatitude=extent[2],
                                          maxlatitude=extent[3],
                                          channel='LHZ,BHZ',
                                          level='channel')
            sta_info = get_sta_info(sta_inv)
            print('Finish.')

        ax.scatter(sta_info['lon'], sta_info['lat'],
                   s=60, c='k', marker='^', edgecolors='none',
                   transform=proj_data, label='Station')

    # Plot station name
    if plot_stnm:
        transform = proj_data._as_mpl_transform(ax)
        for i, stnm in enumerate(sta_info['stnm']):
            if (sta_info['lon'][i] < extent[0]) \
                    or (sta_info['lon'][i] > extent[1]) \
                    or (sta_info['lat'][i] < extent[2]) \
                    or (sta_info['lat'][i] > extent[3]):
                continue
            ax.annotate(stnm, (sta_info['lon'][i], sta_info['lat'][i]),
                        xycoords=transform)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--')
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    gl.top_labels = False
    gl.right_labels = False
    ax.legend()

    if aspect is None:
        aspect = 1.0

    ax.set_aspect(aspect)
    fig.set_size_inches(10/aspect, 10)

    return [fig, ax], sta_info


# Plot station distribution
def plot_stations(sta_info, extent=None, plot_stnm=False, aspect=None,
                  markersize=60, fontsize=18):

    # Map region
    if extent is None:
        margin_in_deg = [10, 5]
        extent = [np.min(sta_info['lon'])-margin_in_deg[0],
                  np.max(sta_info['lon'])+margin_in_deg[0],
                  np.min(sta_info['lat'])-margin_in_deg[1],
                  np.max(sta_info['lat'])+margin_in_deg[1]]

    # Map projection
    proj = ccrs.Mercator()
    proj_data = ccrs.PlateCarree()

    fig = plt.figure()
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj_data)
    ax.add_feature(cfeature.STATES, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE)

    ax.scatter(sta_info['lon'], sta_info['lat'],
               s=markersize, c='k', marker='^', edgecolors='none',
               transform=proj_data, label='Station')

    if plot_stnm:
        transform = proj_data._as_mpl_transform(ax)
        sta_info = sta_info.reset_index()
        for i, stnm in enumerate(sta_info['stnm']):
            if (sta_info['lon'][i] < extent[0]) \
                    or (sta_info['lon'][i] > extent[1]) \
                    or (sta_info['lat'][i] < extent[2]) \
                    or (sta_info['lat'][i] > extent[3]):
                continue
            ax.annotate(stnm, (sta_info['lon'][i], sta_info['lat'][i]),
                        xycoords=transform, fontsize=fontsize)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--')
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    gl.top_labels = False
    gl.right_labels = False
    ax.legend()

    if aspect is None:
        aspect = 1.0

    ax.set_aspect(aspect)
    fig.set_size_inches(10/aspect, 10)

    return [fig, ax]


# Obtain station DataFrame based on Inventory
def get_sta_info(sta_inv):

    # Lists that store information
    net_list = []
    stnm_list = []
    lat_list = []
    lon_list = []
    cha_list = []

    for network in sta_inv:
        net = network.code
        for station in network:
            sta = station.code
            for channel in station:
                net_list.append(net)
                stnm_list.append(sta)
                lat_list.append(channel.latitude)
                lon_list.append(channel.longitude)
                cha_list.append(channel.code)

    # Create dataframe
    sta_info = pd.DataFrame({'stnm': stnm_list,
                             'net': net_list,
                             'lon': np.array(lon_list),
                             'lat': np.array(lat_list),
                             'channel':np.array(cha_list)})
    return sta_info


# Pick out stations having co-located channels
def merge_sta_info(sta_info1, sta_info2):

    sta_list = sta_info2['stnm'].tolist()
    ind_list = []

    for ind in range(len(sta_info1)):
        sta = sta_info1.loc[ind, 'stnm']
        if sta in sta_list:
            ind_list.append(ind)

    sta_info_merge = sta_info1.iloc[ind_list, :4]
    sta_info_merge = sta_info_merge.reset_index(drop=True)

    return sta_info_merge


# Plot TC translation speed
def plot_tc_trans(tc_info):

    # TC track segment length
    geod = Geod(ellps='WGS84')
    _, _, seg_dist = geod.inv(tc_info['lon'][:-1], tc_info['lat'][:-1],
                              tc_info['lon'][1:], tc_info['lat'][1:])

    # Mid-time and time interval
    dt = tc_info['time'][1:] - tc_info['time'][:-1]
    dt_int = [obj.total_seconds() for obj in dt]
    timestamp = tc_info['time'][1:] + dt/2

    # Translation speed in m/s
    speed_trans = seg_dist / dt_int

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(timestamp, speed_trans, 'k-')
    ax.xaxis_date()
    ax.set_ylabel('Speed [m/s]')
    ax.set_ylim(0, np.max(speed_trans))
    ax.set_title('Hurricane Translation Speed')
    ax.grid()
    fig.show()

    return timestamp, speed_trans


# Calculate station distance from the hurricane center in km
# Unit in km
def get_station_dist(sta_loc, tc_loc):

    geod = Geod(ellps='WGS84')
    Npts = tc_loc.shape[1]
    _, _, dist = geod.inv([sta_loc[0]]*Npts, [sta_loc[1]]*Npts,
                          tc_loc[0, :], tc_loc[1, :])
    dist = dist / 1e3

    return dist


# Calculate station location under the hurricane center polar coordinate
# Unit in km
def station_geo2polar(sta_info, tc_loc):

    geod = Geod(ellps='WGS84')
    Npts = sta_info.shape[0]
    az, baz, dist = geod.inv([tc_loc[0]]*Npts, [tc_loc[1]]*Npts,
                             sta_info['lon'], sta_info['lat'])
    dist = dist / 1e3

    # Add new columns to station DataFrame
    sta_info_new = sta_info.assign(azimuth=az, dist=dist, backazimuth=baz)
    return sta_info_new


# Plot station distance from the hurricane center
def plot_station_dist(timestamp, dist, title='Title'):

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(timestamp, dist, 'k-')
    ax.xaxis_date()
    ax.set_ylabel('Distance [km]')
    ax.set_ylim(0, np.nanmax(dist))
    ax.set_title('Distance from Hurricane Center, %s' % title)
    ax.grid()


# Read ASOS data
def read_ASOS_data():
    
    asos_raw = pd.read_csv(os.path.join(base_dir, 'Data/ASOS/645A_HUM_tmpc.csv'))
    asos_raw = asos_raw[asos_raw['tmpc'] != 'M']
    
    asos_T_time = np.array(asos_raw['valid'].transform(lambda x: UTCDateTime(x).datetime).tolist())
    asos_T = np.array(asos_raw['tmpc'].transform(float).tolist())
    asos_T_info = {'time': asos_T_time, 'T': asos_T}
    
    asos_raw = pd.read_csv(os.path.join(base_dir, 'Data/ASOS/645A_HUM_wind.csv'))
    asos_raw = asos_raw[asos_raw['sped'] != 'M']
    
    asos_wind_time = np.array(asos_raw['valid'].transform(lambda x: UTCDateTime(x).datetime).tolist())
    asos_wind = np.array(asos_raw['sped'].transform(float).tolist()) * 0.44704
    asos_wind_info = {'time': asos_wind_time, 'wind': asos_wind}
    
    return asos_T_info, asos_wind_info


# Start time of PSD database
ts_db = UTCDateTime('20120829-00')


# Read PSD database (comp: 'prs', 'uz', 'HV', 'coh')
def read_psd_db(db_dir=os.path.join(base_dir, 'Data/psd_data/'), 
                comp='prs'):
    
    val_med = pd.read_csv(os.path.join(db_dir, '%s_med_20-100s.csv' %comp))
    val_quar1 = pd.read_csv(os.path.join(db_dir, '%s_quar1_20-100s.csv' %comp))
    val_quar3 = pd.read_csv(os.path.join(db_dir, '%s_quar3_20-100s.csv' %comp))
    
    return val_med, val_quar1, val_quar3


# Extract PSD snapshot at particular time (comp: 'prs', 'uz')
def extract_psd_snapshot(snapshot_time, comp='prs', sta_info=None, 
                         db_dir=os.path.join(base_dir, 'Data/psd_data/')):
    
    if sta_info is None:
        pd.read_csv(os.path.join(base_dir, '/Data/station_list.csv'))
    
    # Hurricane track
    f_loc, t0 = get_interp_func(get_tc_info(year=2012, name='Isaac', agency='USA'))
    
    # TC location at snapshot time
    tc_loc = [f_loc[0](snapshot_time - UTCDateTime(t0)), 
              f_loc[1](snapshot_time - UTCDateTime(t0))]
    
    # Calculate station distance to hurricane center
    psd_db = station_geo2polar(sta_info, tc_loc)
    
    # Read PSD database
    val_med, val_quar1, val_quar3 = read_psd_db(db_dir, comp)
    
    # Find corresponding time in the database
    ind_col = round((snapshot_time - ts_db)/3600 * 2)/2
    if ind_col <= 0:
        raise ValueError('Earliest time is %s' %(ts_db + 0.5*3600))
    elif ind_col >= 48:
        raise ValueError('Latest time is %s' %(ts_db + 47.5*3600))
    else:
        t1, t2 = ts_db + (ind_col-0.5)*3600, ts_db + (ind_col+0.5)*3600
        print('Time interval: %s to %s' %(t1.strftime('%m-%d %H:%M'), 
                                         t2.strftime('%m-%d %H:%M')))
    
    # Extract PSD data
    psd_db = psd_db.merge(val_med[['stnm', '%.1f' %ind_col]], on="stnm", how='left')
    psd_db.rename(columns = {'%.1f' %ind_col: '%s_med' %comp}, inplace = True)
    psd_db = psd_db.merge(val_quar1[['stnm', '%.1f' %ind_col]], on="stnm", how='left')
    psd_db.rename(columns = {'%.1f' %ind_col: '%s_quar1' %comp}, inplace = True)
    psd_db = psd_db.merge(val_quar3[['stnm', '%.1f' %ind_col]], on="stnm", how='left')
    psd_db.rename(columns = {'%.1f' %ind_col: '%s_quar3' %comp}, inplace = True)
    
    return psd_db


# Read H*Wind data
def read_Hwind_data(filename):
    
    # Grid data
    df_wind = pd.read_csv(filename, header=3, index_col=0)
    Npts = df_wind.shape[0]

    # Get storm center location
    with open(filename, 'r') as file:
        line = file.readlines()
    line = line[2]
    _lon = re.search(r'(-?\d+\.\d+)', line)
    _lat = re.search(r'(-?\d+\.\d+)', line.split(' and ')[1])

    tc_lon = float(_lon.group()) if _lon else None
    tc_lat = float(_lat.group()) if _lat else None
    
    # Polar coordinate (w.r.t. TC center)
    geod = Geod(ellps='WGS84')
    az, _, dist = geod.inv([tc_lon]*Npts, [tc_lat]*Npts, 
                           df_wind['lon(deg)'], df_wind['lat(deg)'])
    az_rad = np.deg2rad(az)
    
    # Update dataframe
    df_wind['dist(km)'] = dist / 1e3
    df_wind['az(deg)'] = az
    df_wind['R Comp'] = df_wind['U Comp'] * np.sin(az_rad) \
        + df_wind['V Comp'] * np.cos(az_rad)
    df_wind['T Comp'] = -df_wind['V Comp'] * np.cos(az_rad) \
        + df_wind['V Comp'] * np.sin(az_rad)
    
    return df_wind, [tc_lon, tc_lat]