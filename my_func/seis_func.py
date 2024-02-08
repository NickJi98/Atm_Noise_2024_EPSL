#!/usr/bin/env python3

"""
Functions for seismic data and plots

Author: Qing Ji
"""

# Load python packages
import numpy as np
import fnmatch

from obspy.clients.fdsn import Client
client = Client("IRIS")


# Download pressure or seismic record (one trace)
# Instrument response is removed
# Pressure record in Pa, Displacement record in μm (positive for downward)
def download_trace(sta_char, time_range, channel='LHZ',
                   pre_filt='default', extra_portion=0.1, client=Client("IRIS")):

    if pre_filt == 'default':
        pre_filt = [5e-4, 1e-3, 45, 50]

    # Time range
    ts = time_range[0]
    te = time_range[1]

    # Obtain station location and polarity
    sta_inv = client.get_stations(starttime=ts, endtime=te, level='channel',
                                  network=sta_char[0], station=sta_char[1],
                                  channel=channel)
    cha_info = sta_inv[0][0][0]
    sta_loc = np.array([cha_info.longitude, cha_info.latitude])

    # Pressure data
    if fnmatch.fnmatch(channel, '*D[FH]'):
        trace = client.get_waveforms(sta_char[0], sta_char[1], '*', channel,
                                     attach_response=True,
                                     starttime=ts-(te-ts)*extra_portion,
                                     endtime=te+(te-ts)*extra_portion)
        trace = trace.merge()[0]
        trace.remove_response(pre_filt=pre_filt)

    # Seismic data
    elif fnmatch.fnmatch(channel, '*H[EN12Z]'):
        trace = client.get_waveforms(network=sta_char[0], station=sta_char[1],
                                     location='*', channel=channel,
                                     attach_response=True,
                                     starttime=ts-(te-ts)*extra_portion,
                                     endtime=te+(te-ts)*extra_portion)
        trace = trace.merge()[0]
        trace.remove_response(output='VEL', pre_filt=pre_filt)

        # Integration from velocity to displacement
        trace.integrate()

        # Conversion to μm and flip polarity (positive downward)
        trace.stats.dip = cha_info.dip
        if fnmatch.fnmatch(channel, '*HZ'):
            if trace.stats.dip == -90.0:
                trace.data = -trace.data * 1e6
            elif trace.stats.dip == 90.0:
                trace.data = trace.data * 1e6
            else:
                raise ValueError('Weird dip of the vertical channel: %.1f'
                                 % trace.stats.dip)
        
        if fnmatch.fnmatch(channel, '*H[EN12]'):
            trace.data = trace.data * 1e6

    return trace, sta_loc


# Check number of data points
def check_npts(stream):
    
    npts_list = [trace.stats.npts for trace in stream]
    
    if len(set(npts_list)) > 1:
        min_npts = min(npts_list)
        
        for trace, npts in zip(stream, npts_list):
            if npts == min_npts:
                pass
            
            # Infrasound LDF channel sometimes have 1 data point less than
            # seismic channels LH*
            elif npts == min_npts + 1:
                trace.data = trace.data[1:]
                trace.stats.starttime += trace.stats.delta
            
            else:
                raise Exception('npts differ more than 1 among traces!')

    return stream
