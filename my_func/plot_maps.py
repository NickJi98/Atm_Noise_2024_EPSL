#!/usr/bin/env python3

"""
Functions for making (geographical) plots

Author: Qing Ji
"""

# Load python packages
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from obspy import UTCDateTime


# Plot TC track and seismic stations
def make_plot_map(tc_info, sta_info, f_loc):
    
    extent = [-101, -79, 22.5, 40]

    # Map projection
    proj = ccrs.Mercator()
    proj_data = ccrs.PlateCarree()
    fig = plt.figure(dpi=300)
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj_data)
    ax.add_feature(cfeature.STATES, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray')
    
    # Plot stations
    obj_sta, = ax.plot(sta_info['lon'], sta_info['lat'], ls='None',
                       ms=np.sqrt(60), c='k', marker='^', mec='None',
                       transform=proj_data, label='Stations', zorder=10)
    
    # Plot TC track
    marker_size = np.sqrt(40)
    obj_tc_track, = ax.plot(tc_info['lon'], tc_info['lat'], 'b-', 
                            transform=proj_data, label='Hurricane Track', zorder=1)
    obj_dot_1, = ax.plot(tc_info['lon'][tc_info['scale'] == 1],
                         tc_info['lat'][tc_info['scale'] == 1],
                         ms=marker_size, c='orange', marker='o', mec='None', ls='None',
                         transform=proj_data, label='Category 1 Hurricane')
    obj_dot_2, = ax.plot(tc_info['lon'][tc_info['scale'] == 0],
                         tc_info['lat'][tc_info['scale'] == 0],
                         ms=marker_size, c='green', marker='o', mec='None', ls='None',
                         transform=proj_data, label='Tropical Storm')
    obj_dot_3, = ax.plot(tc_info['lon'][tc_info['scale'] == -1],
                         tc_info['lat'][tc_info['scale'] == -1],
                         ms=marker_size, c='blue', marker='o', mec='None', ls='None',
                         transform=proj_data, label='Tropical Depression')
    
    # Plot specific station
    one_info = sta_info[sta_info['stnm']=='645A']
    ax.plot(one_info['lon'], one_info['lat'],
            ms=np.sqrt(120), c='magenta', marker='^', mec='None', ls='None', 
            transform=proj_data, zorder=10)
    ax.text(one_info['lon']-0.3, one_info['lat']-0.05, 'TA.645A', color='magenta', 
            transform=proj_data, horizontalalignment='right', 
            verticalalignment='top', fontsize=18, fontweight='bold')
    
    # Plot date marker
    t0 = tc_info['time'][0]
    for tc_time in [UTCDateTime('20120826-00'), UTCDateTime('20120827-00'), 
                    UTCDateTime('20120828-00'), UTCDateTime('20120829-00'), 
                    UTCDateTime('20120830-00'), UTCDateTime('20120831-00'), 
                    UTCDateTime('20120901-00')]:
        tc_lon = f_loc[0](tc_time-UTCDateTime(t0))
        tc_lat = f_loc[1](tc_time-UTCDateTime(t0))
        obj_date, = ax.plot(tc_lon, tc_lat, ms=marker_size, c='r', marker='o', 
                            mec='None', ls='None', transform=proj_data, 
                            zorder=16, label='Date Marker')
    
    for tc_time in [UTCDateTime('20120827-00'), UTCDateTime('20120828-00'), 
                    UTCDateTime('20120829-00'), UTCDateTime('20120830-00'), 
                    UTCDateTime('20120831-00'), UTCDateTime('20120901-00')]:
        tc_lon = f_loc[0](tc_time-UTCDateTime(t0))
        tc_lat = f_loc[1](tc_time-UTCDateTime(t0))
        obj_text = ax.text(tc_lon+0.1, tc_lat+0.1, r'%02d/%02d' 
                           %(tc_time.month, tc_time.day), color='red', 
                           transform=proj_data, horizontalalignment='left',
                           verticalalignment='bottom', fontsize=14, 
                           fontweight='bold', zorder=15)
        obj_text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='None'))
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle='--')
    gl.xlines = False
    gl.ylines = False
    
    xtick_loc = np.arange(-100, -75, 5)
    ytick_loc = np.arange(22.5, 42.5, 2.5)
    gl.xlocator = mticker.FixedLocator(xtick_loc)
    gl.ylocator = mticker.FixedLocator(ytick_loc)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(xtick_loc, crs=proj_data)
    ax.set_yticks(ytick_loc, crs=proj_data)
    ax.tick_params(axis="both", direction='out', length=5)
    
    ax.add_artist(ax.legend(handles=[obj_date, obj_sta, obj_tc_track], 
                            loc='lower center'))
    ax.add_artist(ax.legend(handles=[obj_dot_1, obj_dot_2, obj_dot_3], 
                            loc='lower left'))
    
    ax.set_aspect('equal', 'box')
    fig.set_size_inches(15,10)
    
    return fig, ax


# Plot HWind snapshot
def plot_Hwind_snapshot(df_wind):
    
    # Reshape data
    Nx = np.sqrt(df_wind.shape[0]).astype(int)
    lon_mat = df_wind['lon(deg)'].to_numpy().reshape(Nx, Nx)
    lat_mat = df_wind['lat(deg)'].to_numpy().reshape(Nx, Nx)
    ws = df_wind['ws(m/s)'].to_numpy().reshape(Nx, Nx)
    
    fig, ax = plt.subplots(figsize=(8,6))
    obj = ax.pcolormesh(lon_mat, lat_mat, ws, cmap='jet', shading='gouraud')
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    fig.colorbar(obj, ax=ax, label='Wind Speed (m/s)')
    
    return fig, ax


# Plot HWind profile
def plot_Hwind_profile(dist, wind, az='k'):
    
    fig, ax = plt.subplots(figsize=(10,10))
    obj = ax.scatter(dist, wind, c=az, s=2, cmap='hsv', alpha=0.75)
    ax.set_xlabel('Distance from Hurricane Center (km)')
    ax.set_ylabel('Surface Wind (m/s)')
    ax.grid()
    fig.colorbar(obj, ax=ax, label='Azimuth (deg)', orientation='horizontal')
    
    return fig, ax


# Plot seismic modeling result
def plot_model_result(mat_data, ax=None, **kwargs):
    
    # Reshape data
    dist_arr = np.squeeze(mat_data['dist_list'])
    PSD_arr = np.squeeze(mat_data['sta_PSD_list'] * 1e12)
    mask = (dist_arr > 0)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))
        
    ax.plot(dist_arr[mask], PSD_arr[mask], **kwargs)
    
    if ax is None:
        return fig, ax
    else:
        return ax