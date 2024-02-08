#!/usr/bin/env python3

"""
Functions for plotting wavelet spectra

Author: Qing Ji
"""

# Load python packages
import numpy as np
import fnmatch

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmcrameri import cm


# Plot one trace
def plot_one_trace(trace_in, filt=None, **kwargs):

    trace = trace_in.copy()
    # Band-pass filt
    if filt is not None:
        trace.filter(type='bandpass', freqmin=filt[0], freqmax=filt[1])

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(trace.times("matplotlib"), trace.data,
            "k-", **kwargs)
    ax.xaxis_date()
    ax.set_title('%s' % ('.'.join((trace.id).split('.')[:2])))
    low, high = ax.get_ylim()
    bound = max(abs(low), abs(high))
    ax.set_ylim(-bound, bound)

    if fnmatch.fnmatch(trace.stats.channel, '*D[FH]'):
        ax.set_ylabel('Pressure [Pa]')
    else:
        ax.set_ylabel('Displacement [$\mu$m]')


# Plot two records in the same panel
def plot_two_traces(trace1_in, trace2_in, filt=None, **kwargs):

    trace1 = trace1_in.copy()
    trace2 = trace2_in.copy()
    # Band-pass filt
    if filt is not None:
        trace1.filter(type='bandpass', freqmin=filt[0], freqmax=filt[1])
        trace2.filter(type='bandpass', freqmin=filt[0], freqmax=filt[1])

    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(trace1.times("matplotlib"), trace1.data,
             "k-", **kwargs)
    ax1.xaxis_date()
    ax1.set_title('%s' % ('.'.join((trace1.id).split('.')[:2])))
    low, high = ax1.get_ylim()
    bound = max(abs(low), abs(high))
    ax1.set_ylim(-bound, bound)

    ax_dist = ax1.twinx()
    ax_dist.plot(trace2.times("matplotlib"), trace2.data, "r-",
                 alpha=0.9, **kwargs)
    ax_dist.tick_params(axis='y', colors='r')
    low, high = ax_dist.get_ylim()
    bound = max(abs(low), abs(high))
    ax_dist.set_ylim(-bound, bound)

    if fnmatch.fnmatch(trace1.stats.channel, '*D[FH]'):
        ax1.set_ylabel('Pressure [Pa]')
    else:
        ax1.set_ylabel('Displacement [$\mu$m]')

    if fnmatch.fnmatch(trace2.stats.channel, '*D[FH]'):
        ax_dist.set_ylabel('Pressure [Pa]', color='r')
    else:
        ax_dist.set_ylabel('Displacement [$\mu$m]', color='r')

    return fig, [ax1, ax_dist]


# Plot wavelet transform results
# Defualt figure properties
cb_pad = 0.08
line_width = 2.0
dist_color = 'white'


# Plot wavelet PSD
def plot_wt_psd(timestamp, freqs, cfs, coi=None, channel='LHZ', decimate=50,
                cmap=cm.vik, freq_scale='log', value_range=[-4,4], title='Title',
                tc_time=None, tc_dist=None, dist_range=[0,1e3], xaxis_date=True, 
                cb_label=None):
   
    # Wavelet PSD
    fig, ax = plt.subplots(figsize=(20,5))
    obj = ax.pcolormesh(timestamp[::decimate], freqs, np.log10(cfs[:, ::decimate]), 
                        cmap=cmap, edgecolors='none',
                        vmin=value_range[0], vmax=value_range[1])
    if channel is None:
        label = cb_label
    else:
        if fnmatch.fnmatch(channel, '*D[FH]'):
            label = 'Log Pressure PSD [Pa$^2$/Hz]'
            title = 'Wavelet Pressure PSD, %s' %(title)
        elif fnmatch.fnmatch(channel, '*HZ'):
            label = 'Log Disp. PSD [($\mu$m)$^2$/Hz]'
            title = 'Wavelet Vertical Disp. PSD, %s' %(title)
        elif fnmatch.fnmatch(channel, '*HN'):
            label = 'Log Disp. PSD [($\mu$m)$^2$/Hz]'
            title = 'Wavelet N-S Disp. PSD, %s' %(title)
        elif fnmatch.fnmatch(channel, '*HE'):
            label = 'Log Disp. PSD [($\mu$m)$^2$/Hz]'
            title = 'Wavelet E-W Disp. PSD, %s' %(title)
    fig.colorbar(obj, ax=ax, pad=cb_pad, label=label)
   
    # Station distance from hurricane
    if tc_time is not None:
        ax_dist = ax.twinx()
        ax_dist.plot(tc_time, tc_dist, color=dist_color, linewidth=line_width)
        ax_dist.tick_params(axis='y', colors='r')
        ax_dist.set_ylabel('Distance [km]', color='r')
        ax_dist.set_ylim(dist_range[0], dist_range[1])
        ax_dist.tick_params(axis="y", length=8, width=2)
   
    # Cone of influence (COI)
    if coi is not None:
        ax.plot(timestamp[::decimate], 1/coi[::decimate], 'w--', linewidth=line_width)

    if xaxis_date:
        ax.xaxis_date()
        # ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        #     mdates.AutoDateLocator(minticks=6, maxticks=10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xlim(timestamp[0], timestamp[-1])
    ax.set_ylabel('Frequency [Hz]')
    ax.set_yscale(freq_scale)
    ax.set_ylim(freqs[-1], freqs[0])
    ax.set_title(title)
    ax.tick_params(axis="both", length=10, width=2)
    ax.tick_params(which='minor', length=6, width=1.5)
    fig.show()

    if tc_time is not None:
        return [fig, ax, ax_dist]
    else:
        return [fig, ax]


# Plot wavelet CSD amplitude
def plot_wt_csd_amp(timestamp, freqs, WX, coi=None, decimate=50,
                    cmap=cm.vik, freq_scale='log', title='Title',
                    tc_time=None, tc_dist=None, dist_range=[0,1e3], xaxis_date=True):

    # CSD amplitude
    fig, ax = plt.subplots(figsize=(20,5))
    obj = ax.pcolormesh(timestamp[::decimate], freqs, np.log10(np.abs(WX[:, ::decimate])), 
                        cmap=cmap, edgecolors='none')
    fig.colorbar(obj, ax=ax, pad=cb_pad, label='Log CSD Amplitude')

    # Station distance from hurricane
    if tc_time is not None:
        ax_dist = ax.twinx()
        ax_dist.plot(tc_time, tc_dist, color=dist_color, linewidth=line_width)
        ax_dist.tick_params(axis='y', colors='r')
        ax_dist.set_ylabel('Distance [km]', color='r')
        ax_dist.set_ylim(dist_range[0], dist_range[1])
        ax_dist.tick_params(axis="y", length=8, width=2)
        
    # Cone of influence (COI)
    if coi is not None:
        ax.plot(timestamp[::decimate], 1/coi[::decimate], 'w--', linewidth=line_width)
        
    if xaxis_date:
        ax.xaxis_date()
        # ax.xaxis.set_major_formatter(
        #     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xlim(timestamp[0], timestamp[-1])
    ax.set_ylabel('Frequency [Hz]')
    ax.set_yscale(freq_scale)
    ax.set_ylim(freqs[-1], freqs[0])
    ax.set_title('Wavelet CSD Amplitude, %s' %title)
    ax.tick_params(axis="both", length=10, width=2)
    ax.tick_params(which='minor', length=6, width=1.5)
    fig.show()

    if tc_time is not None:
        return [fig, ax, ax_dist]
    else:
        return [fig, ax]


# Plot wavelet CSD phase
def plot_wt_csd_phase(timestamp, freqs, WX, coi=None, decimate=50,
                      cmap=cm.vik, freq_scale='log', title='Title',
                      tc_time=None, tc_dist=None, dist_range=[0,1e3], xaxis_date=True):
    # CSD phase
    fig, ax = plt.subplots(figsize=(20,5))
    obj = ax.pcolormesh(timestamp[::decimate], freqs, np.rad2deg(np.angle(WX[:, ::decimate])), 
                        cmap=cmap, edgecolors='none', vmin=-180, vmax=180)
    fig.colorbar(obj, ax=ax, pad=cb_pad, label='CSD Phase [deg]', ticks=[-180, -90, 0, 90, 180])

    # Station distance from hurricane
    if tc_time is not None:
        ax_dist = ax.twinx()
        ax_dist.plot(tc_time, tc_dist, color=dist_color, linewidth=line_width)
        ax_dist.tick_params(axis='y', colors='r')
        ax_dist.set_ylabel('Distance [km]', color='r')
        ax_dist.set_ylim(dist_range[0], dist_range[1])
        ax_dist.tick_params(axis="y", length=8, width=2)

    # Cone of influence (COI)
    if coi is not None:
        ax.plot(timestamp[::decimate], 1/coi[::decimate], 'w--', linewidth=line_width)

    if xaxis_date:
        # ax.xaxis.set_major_formatter(
        #     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xlim(timestamp[0], timestamp[-1])
    ax.set_ylabel('Frequency [Hz]')
    ax.set_yscale(freq_scale)
    ax.set_ylim(freqs[-1], freqs[0])
    ax.set_title('Wavelet CSD Phase, %s' %title)
    ax.tick_params(axis="both", length=10, width=2)
    ax.tick_params(which='minor', length=6, width=1.5)
    fig.show()

    if tc_time is not None:
        return [fig, ax, ax_dist]
    else:
        return [fig, ax]


# Plot coherence of two traces
def plot_wt_coh(timestamp, freqs, Wcoh, coi=None, decimate=50,
                cmap=cm.vik, freq_scale='log', title='Title',
                tc_time=None, tc_dist=None, dist_range=[0,1e3], xaxis_date=True):

    # Coherence
    fig, ax = plt.subplots(figsize=(20, 5))
    obj = ax.pcolormesh(timestamp[::decimate], freqs, Wcoh[:, ::decimate],
                        cmap=cmap, edgecolors='none', vmin=0, vmax=1)
    fig.colorbar(obj, ax=ax, pad=cb_pad, label='Coherence')

    # Station distance from hurricane
    if tc_time is not None:
        ax_dist = ax.twinx()
        ax_dist.plot(tc_time, tc_dist, color=dist_color, linewidth=line_width)
        ax_dist.tick_params(axis='y', colors='r')
        ax_dist.set_ylabel('Distance [km]', color='r')
        ax_dist.set_ylim(dist_range[0], dist_range[1])
        ax_dist.tick_params(axis="y", length=8, width=2)

    # Cone of influence (COI)
    if coi is not None:
        ax.plot(timestamp[::decimate], 1/coi[::decimate], 'w--', linewidth=line_width)

    if xaxis_date:
        ax.xaxis_date()
        # ax.xaxis.set_major_formatter(
        #     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xlim(timestamp[0], timestamp[-1])
    ax.set_ylabel('Frequency [Hz]')
    ax.set_yscale(freq_scale)
    ax.set_ylim(freqs[-1], freqs[0])
    ax.set_title('Coherence, %s' %(title))
    ax.tick_params(axis="both", length=10, width=2)
    ax.tick_params(which='minor', length=6, width=1.5)
    fig.show()

    if tc_time is not None:
        return [fig, ax, ax_dist]
    else:
        return [fig, ax]


# Plot PSD time evolution at particular frequencies
def plot_time_evolution(timestamp, freqs, cfs, freq_samples, channel,
                        title='Title', legend_unit='period'):

    fig, ax = plt.subplots(figsize=(20, 5))
    if fnmatch.fnmatch(channel, '*D[FH]'):
        label = 'Pressure PSD [Pa$^2$/Hz]'
        title = 'Wavelet Pressure PSD, %s' %title
    else:
        label = 'Disp. PSD [($\mu$m)$^2$/Hz]'
        title = 'Wavelet Vertical Disp. PSD, %s' %title

    for freq_point in freq_samples:
        freq_ind = (np.abs(freqs - freq_point)).argmin()
        freq_trace = cfs[freq_ind, :]
        if legend_unit == 'period':
            ax.plot(timestamp, freq_trace, label='%d s' %(1/freq_point))
        else:
            ax.plot(timestamp, freq_trace, label='%.2f Hz' %freq_point)

    ax.set_xlim(timestamp[0], timestamp[-1])
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=5)
    ax.grid()
    fig.show()

    return [fig, ax]


# Plot PSD snapshot
def plot_psd_snapshot(psd_db, comp='prs', snapshot_time=None, 
                      az_color=False, show_cb=True):
    
    if show_cb:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        
    # Inter-quartile range over 1-hr
    ax.errorbar(psd_db['dist'], psd_db[f'{comp}_med'], 
                yerr=[psd_db[f'{comp}_quar1'], psd_db[f'{comp}_quar3']], 
                fmt='none', c='k', alpha=0.3)
    
    # Median PSD level
    if az_color:
        obj = ax.scatter(psd_db['dist'], psd_db[f'{comp}_med'], 
                         c=psd_db['azimuth'].tolist(), 
                         cmap='hsv', vmin=-180, vmax=180)
    else:
        ax.scatter(psd_db['dist'], psd_db[f'{comp}_med'], c='k')
        
    if comp == 'prs':
        ax.set_ylabel('Pressure PSD [Pa$^2$/Hz]')
        ax.set_ylim([1e-3, 1e4])
        
    elif comp == 'uz':
        ax.set_ylabel('Displacement PSD [($\mu$m)$^2$/Hz]')
        ax.set_ylim([1e-4, 1e3])
        
    ax.set_xlabel('Distance from Hurricane Center [km]')
    ax.set_yscale('log')
    ax.set_xlim([0, 1e3])
    ax.grid()
    
    # Show colorbar for azimuthal range
    if az_color and show_cb:
        fig.colorbar(obj, label='Azimuth [deg]', orientation='horizontal')
    
    if snapshot_time is not None:
        ax.set_title(snapshot_time.strftime('%m-%d %H:%M'))
        
    return fig, ax