# Standard
import os
import sys
import copy
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from dateutil import tz
import gzip
import h5py
import logging
import numpy as np
import pdb
import pickle
import re
import scipy
import scipy.io
import shutil
import subprocess
from matplotlib.dates import date2num, num2date
from glob import iglob
import json
import urllib
from sunpy.time import parse_time
import pandas as pd
import matplotlib.dates as mdates

# External
import astropy.time
import astropy.units as u

# Local
#from .predict import calc_dst_temerin_li
from .config.constants import AU, dist_to_L1

logger = logging.getLogger(__name__)

def interp_nans(sc_in, single=False):
    sc = copy.deepcopy(sc_in)
    
    n = len(sc)

    # Use index positions as the x-axis (since time steps are regular)
    x = np.arange(n)
    
    if single:
        mask = ~np.isnan(sc)
        
        if np.sum(mask) >= 2:  # Need at least 2 points to interpolate
            # Interpolate over valid points
            sc = np.interp(x, x[mask], sc[mask])
        else:
            print(f"Skipping '{key}': not enough valid data to interpolate.")
            
    else:
        keys_to_interp = [name for name in sc.dtype.names if name != 'time']

        for key in keys_to_interp:
            y = sc[key]

            if y.dtype.kind in 'f':  # Only interpolate float fields
                mask = ~np.isnan(y)

                if np.sum(mask) >= 2:  # Need at least 2 points to interpolate
                    # Interpolate over valid points
                    sc[key] = np.interp(x, x[mask], y[mask])
                else:
                    print(f"Skipping '{key}': not enough valid data to interpolate.")
  
    return sc


def interp_to_grid_sc(sc_in):
    sc_input1 = copy.deepcopy(sc_in)
    #2025-11-09 09:51:00 2025-11-12 04:04:51.593245
    t_start=sc_input1.time_shifted_exp[0] #datetime(2025,11,9,9,51)
    t_end=sc_input1.time_shifted_exp.iloc[-1] #datetime(2025,11,12,4,4)

    t1=sc_input1.time_shifted_exp

    time_all = [ t_start + timedelta(minutes=1*n) for n in range(int ((t_end - t_start).total_seconds()/60))]  

    time_mat=mdates.date2num(time_all) 

    sc_input = np.zeros(np.size(time_all), dtype=[('time', object),\
                ('bt', float), ('bt_err', float), ('bx',float), ('bx_err', float), ('by',float), ('by_err',float),\
                ('bz', float), ('bz_err', float), ('vt', float), ('np', float)])

    sc_input =sc_input.view(np.recarray)  

    time_m_num=mdates.date2num(t1) #make date number

    sc_input.time=time_all
    sc_input.np=np.interp(time_mat, time_m_num, sc_input1.np)
    sc_input.vt=np.interp(time_mat, time_m_num, sc_input1.vt)
    sc_input.bx=np.interp(time_mat, time_m_num, sc_input1.bx_scaled)
    sc_input.bx_err = np.interp(time_mat, time_m_num, np.abs(sc_input1.bx_scaled_ub-sc_input1.bx_scaled_lb))
    sc_input.by=np.interp(time_mat, time_m_num, sc_input1.by_scaled)
    sc_input.by_err = np.interp(time_mat, time_m_num, np.abs(sc_input1.by_scaled_ub-sc_input1.by_scaled_lb))
    sc_input.bz=np.interp(time_mat, time_m_num, sc_input1.bz_scaled)
    sc_input.bz_err = np.interp(time_mat, time_m_num, np.abs(sc_input1.bz_scaled_ub-sc_input1.bz_scaled_lb))
    sc_input.bt=np.interp(time_mat, time_m_num, sc_input1.bt_scaled)
    sc_input.bt_err = np.interp(time_mat, time_m_num, np.abs(sc_input1.bt_scaled_ub-sc_input1.bt_scaled_lb))
    
    print(sc_input.time[0], sc_input.time[-1])
    return sc_input



def interp_to_grid_l1(sc_in):
    sc_input1 = copy.deepcopy(sc_in)
    #2025-11-04 15:20:00 2025-11-11 15:17:00
    if type(sc_in.time) == pd.core.series.Series:
        t_start=sc_input1.time.iloc[0].round('min')
        t_end=sc_input1.time.iloc[-1].round('min')
    else:
        t_start=sc_input1.time[0]
        t_end=sc_input1.time[-1]

    t1=sc_input1.time

    #hour_res:
    #time_all = [ t_start + timedelta(hours=1*n) for n in range(int ((t_end - t_start).total_seconds()/60./60.))]
    
    #min res:
    res =round((sc_in.time[1]-sc_in.time[0]).total_seconds()/60.,0)
    time_all = [ t_start + timedelta(minutes=res*n) for n in range(int ((t_end - t_start).total_seconds()/60.))]
    
    time_mat=mdates.date2num(time_all) 

    sc_input = np.zeros(np.size(time_all), dtype=[('time', object),\
                ('bt', float), ('bx',float), ('by',float),\
                ('bz', float), ('vt', float), ('np', float),\
                ('r', float), ('lon',float), ('lat',float),\
                ('x', float), ('y',float), ('z',float)])

    sc_input =sc_input.view(np.recarray)  

    time_m_num=mdates.date2num(t1) #make date number

    sc_input.time=time_all
    sc_input.np=np.interp(time_mat, time_m_num, sc_input1.np)
    sc_input.vt=np.interp(time_mat, time_m_num, sc_input1.vt)
    sc_input.bx=np.interp(time_mat, time_m_num, sc_input1.bx)
    sc_input.by=np.interp(time_mat, time_m_num, sc_input1.by)
    sc_input.bz=np.interp(time_mat, time_m_num, sc_input1.bz)
    sc_input.bt=np.interp(time_mat, time_m_num, sc_input1.bt)
    sc_input.r=np.interp(time_mat, time_m_num, sc_input1.r)
    sc_input.lon=np.interp(time_mat, time_m_num, sc_input1.lon)
    sc_input.lat=np.interp(time_mat, time_m_num, sc_input1.lat)
    sc_input.x=np.interp(time_mat, time_m_num, sc_input1.x)
    sc_input.y=np.interp(time_mat, time_m_num, sc_input1.y)
    sc_input.z=np.interp(time_mat, time_m_num, sc_input1.z)
    
    
    sc_input_interp_nan = interp_nans(sc_input)
    
    #print(sc_input.time[0], sc_input.time[-1])
    return sc_input_interp_nan

def interp_to_grid_solo_mag(t_start, t_end, sc_in):
    sc_input1 = copy.deepcopy(sc_in)
    #2025-11-04 15:20:00 2025-11-11 15:17:00
    #t_start=datetime(2026,1,18,2) #sc_input1.time[0]
    #t_end=datetime(2026,1,21) #sc_input1.time[-1]

    t1=sc_input1.time

    #hour_res:
    #time_all = [ t_start + timedelta(hours=1*n) for n in range(int ((t_end - t_start).total_seconds()/60./60.))]
    
    #min res:
    time_all = [ t_start + timedelta(minutes=1*n) for n in range(int ((t_end - t_start).total_seconds()/60.))]
    
    time_mat=mdates.date2num(time_all) 

    sc_input = np.zeros(np.size(time_all), dtype=[('time', object),\
                ('bt', float), ('bx',float), ('by',float),\
                ('bz', float)])

    sc_input =sc_input.view(np.recarray)  

    time_m_num=mdates.date2num(t1) #make date number

    sc_input.time=time_all
    sc_input.bx=np.interp(time_mat, time_m_num, sc_input1.bx)
    sc_input.by=np.interp(time_mat, time_m_num, sc_input1.by)
    sc_input.bz=np.interp(time_mat, time_m_num, sc_input1.bz)
    sc_input.bt=np.interp(time_mat, time_m_num, sc_input1.bt)
    
    sc_input_interp_nan = interp_nans(sc_input)
    
    print(sc_input.time[0], sc_input.time[-1])
    return sc_input_interp_nan

def interp_to_grid_solo_plas(t_start, t_end, sc_in):
    sc_input1 = copy.deepcopy(sc_in)
    #2025-11-04 15:20:00 2025-11-11 15:17:00
    #t_start=datetime(2026,1,18,2) #sc_input1.time[0]
    #t_end=datetime(2026,1,21) #sc_input1.time[-1]

    t1=sc_input1.time

    #hour_res:
    #time_all = [ t_start + timedelta(hours=1*n) for n in range(int ((t_end - t_start).total_seconds()/60./60.))]
    
    #min res:
    time_all = [ t_start + timedelta(minutes=1*n) for n in range(int ((t_end - t_start).total_seconds()/60.))]
    
    time_mat=mdates.date2num(time_all) 

    sc_input = np.zeros(np.size(time_all), dtype=[('time', object),\
                                                  ('vt', float), ('np', float), ('tp', float),\
                                                  ('vx', float),('vy', float),('vz', float)])

    sc_input =sc_input.view(np.recarray)  

    time_m_num=mdates.date2num(t1) #make date number

    sc_input.time=time_all
    sc_input.np=np.interp(time_mat, time_m_num, sc_input1.np)
    sc_input.vt=np.interp(time_mat, time_m_num, sc_input1.vt)
    sc_input.tp=np.interp(time_mat, time_m_num, sc_input1.tp)
    sc_input.vx=np.interp(time_mat, time_m_num, sc_input1.vx)
    sc_input.vy=np.interp(time_mat, time_m_num, sc_input1.vy)
    sc_input.vz=np.interp(time_mat, time_m_num, sc_input1.vz)
    
    sc_input_interp_nan = interp_nans(sc_input)
    
    print(sc_input.time[0], sc_input.time[-1])
    return sc_input_interp_nan


#------ ELEvo --------

#### Update initial parameter for ELEvo according to actual arrival time of CME ###
#start_time = time.time()

def true_prop_params(sc_measure, t0_num, data_donki, sc_icme_start_time):

    #if sc_measure_name == 'PSP':
    #    sc_measure = psp
    #if sc_measure_name == 'SolarOrbiter':
    #    sc_measure = solo
    #if sc_measure_name == 'STEREO-A':
    #    sc_measure = sc
    #if sc_measure_name == 'BepiColombo':
    #    sc_measure = bepi
    #if sc_measure_name == 'Wind':
    #    sc_measure = wind

    sc_ind_measure = np.argmin(np.abs(t0_num-sc_measure.time))


    index_sc_heliodistance = np.argmin(np.abs(mdates.date2num(sc_icme_start_time)-sc_measure.time))
    r_sc=sc_measure.r[index_sc_heliodistance]
    cme_prop_time = (sc_icme_start_time-data_donki.time21_5).total_seconds()

    if np.abs(np.deg2rad(data_donki.longitude)) + np.abs(sc_measure.lon[index_sc_heliodistance]) > np.pi and np.sign(np.deg2rad(data_donki.longitude)) != np.sign(sc_measure.lon[index_sc_heliodistance]): 
        delta_sc_measure = np.deg2rad(data_donki.longitude) - (sc_measure.lon[index_sc_heliodistance] + 2 * np.pi * np.sign(np.deg2rad(data_donki.longitude)))

    else:
        delta_sc_measure = np.deg2rad(data_donki.longitude) - sc_measure.lon[index_sc_heliodistance]
        
    return cme_prop_time, r_sc, delta_sc_measure

    
def cme_r_prop(x, time_diff, distance0):
    if x[0] >= x[1]:
        accsign = 1.

    else:
        accsign = -1.

    rdrag = (accsign / (x[2] * 1e-7)) * np.log(1 + (accsign * x[2] * 1e-7) * ((x[0] - x[1]) * time_diff)) + x[1] * time_diff + distance0
    vdrag = (x[0] - x[1]) / (1 + (accsign * (x[2] * 1e-7) * (x[0] - x[1]) * time_diff)) + x[1]

    return rdrag, vdrag


def rmse(x, time_diff, r_sc, cme_delta, f, halfwidth, distance0): #v_sc=speed_ic, 

    
    if x[0] >= x[1]:
        accsign = 1.

    else:
        accsign = -1.

    r_at_sc = (accsign / (x[2] * 1e-7)) * np.log(1 + (accsign * x[2] * 1e-7) * ((x[0] - x[1]) * time_diff)) + x[1] * time_diff + distance0
    #v_at_sc= (x[2] - x[1]) / (1 + (accsign * (x[2] * 1e-7) * (x[0] - x[1]) * time_diff)) + x[1]

    cme_r_au = r_at_sc*u.km.to(u.au)

    theta_monitor = np.arctan(f**2 * np.tan(halfwidth))
    omega_monitor = np.sqrt(np.cos(theta_monitor)**2 * (f**2 - 1) + 1)   
    cme_b_monitor = cme_r_au * omega_monitor * np.sin(halfwidth) / (np.cos((halfwidth) - theta_monitor) + omega_monitor * np.sin(halfwidth))    
    cme_a_monitor = cme_b_monitor / f
    cme_c_monitor = cme_r_au - cme_b_monitor

    root = np.sin(cme_delta)**2 * f**2 * (cme_b_monitor**2 - cme_c_monitor**2) + np.cos(cme_delta)**2 * cme_b_monitor**2
    discnce_sc_monitor = (cme_c_monitor * np.cos(cme_delta) + np.sqrt(root)) / (np.sin(cme_delta)**2 * f**2 + np.cos(cme_delta)**2)

    rmse = np.sqrt((r_sc - discnce_sc_monitor)**2)

    return rmse



def check(x, time_diff, cme_prop_time, cme_delta, distance0, f, halfwidth): #v_sc=speed_ic, 

    r_at_sc, v_at_sc = cme_r_prop(x, cme_prop_time, distance0)

    cme_r_au = r_at_sc*u.km.to(u.au)
    cme_v_au = v_at_sc

    theta_monitor = np.arctan(f**2 * np.tan(halfwidth))
    omega_monitor = np.sqrt(np.cos(theta_monitor)**2 * (f**2 - 1) + 1)   
    cme_b_monitor = cme_r_au * omega_monitor * np.sin(halfwidth) / (np.cos((halfwidth) - theta_monitor) + omega_monitor * np.sin(halfwidth))    
    cme_a_monitor = cme_b_monitor / f
    cme_c_monitor = cme_r_au - cme_b_monitor

    root = np.sin(cme_delta)**2 * f**2 * (cme_b_monitor**2 - cme_c_monitor**2) + np.cos(cme_delta)**2 * cme_b_monitor**2
    distance_sc_monitor = (cme_c_monitor * np.cos(cme_delta) + np.sqrt(root)) / (np.sin(cme_delta)**2 * f**2 + np.cos(cme_delta)**2)

    return distance_sc_monitor, cme_v_au, cme_r_au


#gamma_init = [0.1, gamma_update]
#ambient_wind_init = [400., ambient_wind_update]
#speed_init = [data_donki.speed[index], speed_update]

def donki_kinematics(earth, wind, solo, data_donki, params_opt, distance0):
    
    print('   ')
    print('Processing..')
    
    #print(data_donki['associatedCMEID'][index])
    
    #index = data_donki.associatedCMEID[data_donki.associatedCMEID == '2024-02-27T11:36:00-CME-001'].index[0]
    distance0 = distance0 #21.5*u.solRad.to(u.km)
    t0 = data_donki.time21_5
    t0_num = mdates.date2num(t0)

    kindays = 15
    n_ensemble = 100000
    halfwidth = np.deg2rad(35.)
    #halfwidth = np.deg2rad(data_donki.halfAngle[index])
    #print(np.rad2deg(halfwidth))
    res_in_min = 10
    f = 0.7
    kindays_in_min = int(kindays*24*60/res_in_min)
    
    gamma_update = params_opt.x[2]
    ambient_wind_update = params_opt.x[1]
    speed_update = params_opt.x[0]

    #for sc in ['earth', 'solo']:
    t0_num_kindays = mdates.date2num(mdates.num2date(t0_num)+timedelta(days=kindays))
    
    dct_earth = t0_num-earth.time
    earth_ind = np.argmin(np.abs(dct_earth))
    
    dct0_earth = t0_num_kindays-earth.time
    earth_ind2 = np.argmin(np.abs(dct0_earth))
    
    dct = t0_num-wind.time
    wind_ind = np.argmin(np.abs(dct))
    
    dct0 = t0_num_kindays-wind.time
    wind_ind2 = np.argmin(np.abs(dct0))
        
    dct1 = t0_num-solo.time
    solo_ind = np.argmin(np.abs(dct1))
    
    dct2 = t0_num_kindays-solo.time
    solo_ind2 = np.argmin(np.abs(dct2))

    delta_earth_list = []
    for j in range(kindays_in_min):
        delta_earth = (np.deg2rad(data_donki.longitude) - earth.lon[earth_ind+j:earth_ind+(j+1)])[0]
        delta_earth_list.append(delta_earth)

    
    delta_wind_list = []
    for j in range(kindays_in_min):
        delta_wind = (np.deg2rad(data_donki.longitude) - wind.lon[wind_ind+j:wind_ind+(j+1)])[0]
        delta_wind_list.append(delta_wind)
        
        
        
    #    if np.abs(np.deg2rad(data_donki.longitude)) + np.abs(wind.lon[wind_ind+j:wind_ind+(j+1)]) > np.pi and np.sign(np.deg2rad(data_donki.longitude)) != np.sign(wind.lon[wind_ind+j:wind_ind+(j+1)]):
       #     delta_wind = (np.deg2rad(data_donki.longitude) - (wind.lon[wind_ind+j:wind_ind+(j+1)] + 2 * np.pi * np.sign(np.deg2rad(data_donki.longitude))))[0]

        #else:
            
    
    
    delta_solo_list = []
    for j in range(kindays_in_min):
        if np.abs(np.deg2rad(data_donki.longitude)) + np.abs(solo.lon[solo_ind+j:solo_ind+(j+1)]) > np.pi and np.sign(np.deg2rad(data_donki.longitude)) != np.sign(solo.lon[solo_ind+j:solo_ind+(j+1)]):
            delta_solo = (np.deg2rad(data_donki.longitude) - (solo.lon[solo_ind+j:solo_ind+(j+1)] + 2 * np.pi * np.sign(np.deg2rad(data_donki.longitude))))[0]

        else:
            delta_solo = (np.deg2rad(data_donki.longitude) - solo.lon[solo_ind+j:solo_ind+(j+1)])[0]
            
        delta_solo_list.append(delta_solo)
    
    #times for each event kinematic
    time1=[]
    tstart1=copy.deepcopy(t0)
    tend1=tstart1+timedelta(days=kindays)
    #make 30 min datetimes
    while tstart1 < tend1:

        time1.append(tstart1)  
        tstart1 += timedelta(minutes=res_in_min)    

    #make kinematics
    
    timestep=np.zeros([kindays_in_min,n_ensemble])
    cme_r=np.zeros([kindays_in_min, 3])
    cme_v=np.zeros([kindays_in_min, 3])
    cme_lon=np.ones(kindays_in_min)*data_donki.longitude
    cme_lat=np.ones(kindays_in_min)*data_donki.latitude
    cme_id=np.chararray(kindays_in_min, itemsize=27)
    cme_id[:]=data_donki.associatedCMEID
    cme_r_ensemble=np.zeros([kindays_in_min, n_ensemble])
    cme_v_ensemble=np.zeros([kindays_in_min, n_ensemble])
    cme_delta_earth = [[x] * 3 for x in delta_earth_list]
    cme_delta = [[x] * 3 for x in delta_wind_list]
    cme_delta_solo = [[x] * 3 for x in delta_solo_list]
    cme_hit_earth=np.zeros(kindays_in_min)
    cme_hit_earth[np.abs(delta_earth_list[0])<halfwidth] = 1
    cme_hit=np.zeros(kindays_in_min)
    cme_hit[np.abs(delta_wind_list[0])<halfwidth] = 1
    cme_hit_solo=np.zeros(kindays_in_min)
    cme_hit_solo[np.abs(delta_solo_list[0])<halfwidth] = 1
    distance_earth = np.empty([kindays_in_min,3])
    distance_wind = np.empty([kindays_in_min,3])
    distance_solo = np.empty([kindays_in_min,3])
    distance_earth[:] = np.nan
    distance_wind[:] = np.nan
    distance_solo[:] = np.nan
   
    gamma = np.abs(np.random.normal(gamma_update,0.025,n_ensemble))
    ambient_wind = np.random.normal(ambient_wind_update,50,n_ensemble)
    speed = np.random.normal(speed_update,50,n_ensemble)
    
    #print(speed_init[i])
    
    timesteps = np.arange(kindays_in_min)*res_in_min*60
    timesteps = np.vstack([timesteps]*n_ensemble)
    timesteps = np.transpose(timesteps)

    accsign = np.ones(n_ensemble)
    accsign[speed < ambient_wind] = -1.

    distance0_list = np.ones(n_ensemble)*distance0
    
    cme_r_ensemble = (accsign / (gamma * 1e-7)) * np.log(1 + (accsign * (gamma * 1e-7) * ((speed - ambient_wind) * timesteps))) + ambient_wind * timesteps + distance0_list
    cme_v_ensemble = (speed - ambient_wind) / (1 + (accsign * (gamma * 1e-7) * (speed - ambient_wind) * timesteps)) + ambient_wind

    cme_r_mean = cme_r_ensemble.mean(1)
    cme_r_std = cme_r_ensemble.std(1)
    cme_v_mean = cme_v_ensemble.mean(1)
    cme_v_std = cme_v_ensemble.std(1)
    cme_r[:,0]= cme_r_mean*u.km.to(u.au)
    cme_r[:,1]=(cme_r_mean - 2*cme_r_std)*u.km.to(u.au) 
    cme_r[:,2]=(cme_r_mean + 2*cme_r_std)*u.km.to(u.au)
    cme_v[:,0]= cme_v_mean
    cme_v[:,1]=(cme_v_mean - 2*cme_v_std)
    cme_v[:,2]=(cme_v_mean + 2*cme_v_std)
    
    #Ellipse parameters   
    theta = np.arctan(f**2*np.ones([kindays_in_min,3]) * np.tan(halfwidth*np.ones([kindays_in_min,3])))
    omega = np.sqrt(np.cos(theta)**2 * (f**2*np.ones([kindays_in_min,3]) - 1) + 1)   
    cme_b = cme_r * omega * np.sin(halfwidth*np.ones([kindays_in_min,3])) / (np.cos(halfwidth*np.ones([kindays_in_min,3]) - theta) + omega * np.sin(halfwidth*np.ones([kindays_in_min,3])))    
    cme_a = cme_b / f*np.ones([kindays_in_min,3])
    cme_c = cme_r - cme_b

    root_earth = np.sin(cme_delta_earth)**2 * f**2*np.ones([kindays_in_min,3]) * (cme_b**2 - cme_c**2) + np.cos(cme_delta_earth)**2 * cme_b**2
    distance_earth[cme_hit_earth.all() == 1] = (cme_c * np.cos(cme_delta_earth) + np.sqrt(root_earth)) / (np.sin(cme_delta_earth)**2 * f**2*np.ones([kindays_in_min,3]) + np.cos(cme_delta_earth)**2) 
    
    root = np.sin(cme_delta)**2 * f**2*np.ones([kindays_in_min,3]) * (cme_b**2 - cme_c**2) + np.cos(cme_delta)**2 * cme_b**2
    distance_wind[cme_hit.all() == 1] = (cme_c * np.cos(cme_delta) + np.sqrt(root)) / (np.sin(cme_delta)**2 * f**2*np.ones([kindays_in_min,3]) + np.cos(cme_delta)**2) #distance from SUN in AU for given point on ellipse
    
    root_solo = np.sin(cme_delta_solo)**2 * f**2*np.ones([kindays_in_min,3]) * (cme_b**2 - cme_c**2) + np.cos(cme_delta_solo)**2 * cme_b**2
    distance_solo[cme_hit_solo.all() == 1] = (cme_c * np.cos(cme_delta_solo) + np.sqrt(root_solo)) / (np.sin(cme_delta_solo)**2 * f**2*np.ones([kindays_in_min,3]) + np.cos(cme_delta_solo)**2) 
    
    #### linear interpolate to 10 min resolution

    #find next full hour after t0
    format_str = '%Y-%m-%d %H'  
    t0r = datetime.strptime(datetime.strftime(t0, format_str), format_str) +timedelta(hours=1)
    time2=[]
    tstart2=copy.deepcopy(t0r)
    tend2=tstart2+timedelta(days=kindays)
    #make 30 min datetimes 
    while tstart2 < tend2:
        time2.append(tstart2)  
        tstart2 += timedelta(minutes=res_in_min)  

    time2_num=mdates.date2num(time2) #parse_time(time2).plot_date        
    time1_num=mdates.date2num(time1) #parse_time(time1).plot_date
    
    
    arr_time_earth = []
    arrival_earth = []
    arr_time_fin_earth = []
    arr_time_err0_earth = []
    arr_time_err1_earth = []
    arr_speed_fin_earth = []
    arr_speed_err_earth = []
    arr_id_earth = []
    arr_hit_earth = []
    if np.isnan(distance_earth).all() == False:

        for t in range(3):
           
            index_earth = np.argmin(np.abs(np.ma.array(distance_earth[:,t], mask=np.isnan(distance_earth[:,t])) - earth.r[earth_ind:earth_ind2]))            
            arr_time_earth.append(time1[int(index_earth)])
        
        arr_speed_earth = cme_v[:,0][index_earth]
        err_arr_speed_earth = cme_v[:,2][index_earth]-cme_v[:,1][index_earth]
        err_arr_time_earth = (arr_time_earth[1]-arr_time_earth[2]).total_seconds()/3600.   
        #arrival_earth.append([cme_id[0].decode("utf-8"), t0.strftime('%Y-%m-%dT%H:%MZ'), "{:.1f}".format(cme_lon[0]), "{:.1f}".format(cme_lat[0]), "{:.1f}".format(speed_init[i]), arr_time_earth[0].strftime('%Y-%m-%dT%H:%MZ'), "{:.2f}".format(err_arr_time_earth), "{:.2f}".format(arr_speed_earth), "{:.2f}".format(err_arr_speed_earth)])             
        arr_time_fin_earth.append(arr_time_earth[0])
        arr_time_err0_earth.append(arr_time_earth[0]-timedelta(hours=err_arr_time_earth))
        arr_time_err1_earth.append(arr_time_earth[0]+timedelta(hours=err_arr_time_earth))
        arr_id_earth.append(cme_id[0].decode("utf-8"))
        arr_hit_earth.append(1.)
        
        #if i == 0:
         #   print('Arrival Time (earth): ', arr_time_fin_earth[0].strftime('%Y-%m-%dT%H:%MZ'), '  +/-', "{:.2f}".format(err_arr_time_earth), 'h')
          #  print('Arrival Speed (earth): ', "{:.2f}".format(arr_speed_earth), '  +/-', "{:.2f}".format(err_arr_speed), 'km/s')

        #if i == 1:
        print('Updated Arrival Time (Earth): ', arr_time_fin_earth[0].strftime('%Y-%m-%dT%H:%MZ'), '  +/-', "{:.2f}".format(err_arr_time_earth), 'h')
        print('Updated Arrival Speed (Earth): ', "{:.2f}".format(arr_speed_earth), '  +/-', "{:.2f}".format(err_arr_speed_earth), 'km/s')
        
    else:
        arr_time_fin_earth.append(np.nan)
        arr_time_err0_earth.append(np.nan)
        arr_time_err1_earth.append(np.nan)
        arr_id_earth.append(np.nan)
        arr_hit_earth.append(np.nan)
    
    
    arr_time = []
    arrival = []
    arr_time_fin = []
    arr_time_err0 = []
    arr_time_err1 = []
    arr_speed_fin = []
    arr_speed_err = []
    arr_id = []
    arr_hit = []
    
    if np.isnan(distance_wind).all() == False:

        for t in range(3):
            index_wind = np.argmin(np.abs(np.ma.array(distance_wind[:,t], mask=np.isnan(distance_wind[:,t])) - wind.r[wind_ind:wind_ind2])) 
            arr_time.append(time1[int(index_wind)])
        
        arr_speed = cme_v[:,0][index_wind]
        err_arr_speed = cme_v[:,2][index_wind]-cme_v[:,1][index_wind]
        err_arr_time = (arr_time[1]-arr_time[2]).total_seconds()/3600.   
        arrival.append([cme_id[0].decode("utf-8"), t0.strftime('%Y-%m-%dT%H:%MZ'), "{:.1f}".format(cme_lon[0]), "{:.1f}".format(cme_lat[0]), "{:.1f}".format(speed_update), arr_time[0].strftime('%Y-%m-%dT%H:%MZ'), "{:.2f}".format(err_arr_time), "{:.2f}".format(arr_speed), "{:.2f}".format(err_arr_speed)])   
        arr_time_fin.append(arr_time[0])
        arr_time_err0.append(arr_time[0]-timedelta(hours=err_arr_time))
        arr_time_err1.append(arr_time[0]+timedelta(hours=err_arr_time))
        arr_id.append(cme_id[0].decode("utf-8"))
        arr_hit.append(1.)
       
        #if i == 0:
         #   print('Arrival Time (L1): ', arr_time_fin[0].strftime('%Y-%m-%dT%H:%MZ'), '  +/-', "{:.2f}".format(err_arr_time), 'h')
          #  print('Arrival Speed (L1): ', "{:.2f}".format(arr_speed), '  +/-', "{:.2f}".format(err_arr_speed), 'km/s')

        #if i == 1:
        print('Updated Arrival Time (L1): ', arr_time_fin[0].strftime('%Y-%m-%dT%H:%MZ'), '  +/-', "{:.2f}".format(err_arr_time), 'h')
        print('Updated Arrival Speed (L1): ', "{:.2f}".format(arr_speed), '  +/-', "{:.2f}".format(err_arr_speed), 'km/s')

    
    else:
        arr_time_fin.append(np.nan)
        arr_time_err0.append(np.nan)
        arr_time_err1.append(np.nan)
        arr_id.append(np.nan)
        arr_hit.append(np.nan)
        

        
    arr_time_solo = []
    arrival_solo = []
    arr_time_fin_solo = []
    arr_time_err0_solo = []
    arr_time_err1_solo = []
    arr_speed_fin_solo = []
    arr_speed_err_solo = []
    arr_id_solo = []
    arr_hit_solo = []
    if np.isnan(distance_solo).all() == False:

        for t in range(3):
           
            index_solo = np.argmin(np.abs(np.ma.array(distance_solo[:,t], mask=np.isnan(distance_solo[:,t])) - solo.r[solo_ind:solo_ind2]))            
            arr_time_solo.append(time1[int(index_solo)])
        
        arr_speed_solo = cme_v[:,0][index_solo]
        err_arr_speed_solo = cme_v[:,2][index_solo]-cme_v[:,1][index_solo]
        err_arr_time_solo = (arr_time_solo[1]-arr_time_solo[2]).total_seconds()/3600.   
        #arrival_solo.append([cme_id[0].decode("utf-8"), t0.strftime('%Y-%m-%dT%H:%MZ'), "{:.1f}".format(cme_lon[0]), "{:.1f}".format(cme_lat[0]), "{:.1f}".format(speed_init[i]), arr_time_solo[0].strftime('%Y-%m-%dT%H:%MZ'), "{:.2f}".format(err_arr_time_solo), "{:.2f}".format(arr_speed_solo), "{:.2f}".format(err_arr_speed_solo)])             
        arr_time_fin_solo.append(arr_time_solo[0])
        arr_time_err0_solo.append(arr_time_solo[0]-timedelta(hours=err_arr_time_solo))
        arr_time_err1_solo.append(arr_time_solo[0]+timedelta(hours=err_arr_time_solo))
        arr_id_solo.append(cme_id[0].decode("utf-8"))
        arr_hit_solo.append(1.)
        
        #if i == 0:
         #   print('Arrival Time (SolO): ', arr_time_fin_solo[0].strftime('%Y-%m-%dT%H:%MZ'), '  +/-', "{:.2f}".format(err_arr_time_solo), 'h')
          #  print('Arrival Speed (SolO): ', "{:.2f}".format(arr_speed_solo), '  +/-', "{:.2f}".format(err_arr_speed), 'km/s')

        #if i == 1:
        print('Updated Arrival Time (SolO): ', arr_time_fin_solo[0].strftime('%Y-%m-%dT%H:%MZ'), '  +/-', "{:.2f}".format(err_arr_time_solo), 'h')
        print('Updated Arrival Speed (SolO): ', "{:.2f}".format(arr_speed_solo), '  +/-', "{:.2f}".format(err_arr_speed_solo), 'km/s')
        
    else:
        arr_time_fin_solo.append(np.nan)
        arr_time_err0_solo.append(np.nan)
        arr_time_err1_solo.append(np.nan)
        arr_id_solo.append(np.nan)
        arr_hit_solo.append(np.nan)
    
    print(' ')
    #linear interpolation to time_mat times    
    cme_r = [np.interp(time2_num, time1_num,cme_r[:,i]) for i in range(3)]
    cme_v = [np.interp(time2_num, time1_num,cme_v[:,i]) for i in range(3)]
    cme_lat = np.interp(time2_num, time1_num,cme_lat )
    cme_lon = np.interp(time2_num, time1_num,cme_lon )
    cme_a = [np.interp(time2_num, time1_num,cme_a[:,i]) for i in range(3)]
    cme_b = [np.interp(time2_num, time1_num,cme_b[:,i]) for i in range(3)]
    cme_c = [np.interp(time2_num, time1_num,cme_c[:,i]) for i in range(3)]
    #arr_time = np.interp(time2_num, time1_num,arr_time_fin_list)
    #arr_time_err = np.interp(time2_num, time1_num,arr_time_err_list) 
    
    #with open('output/icme_arrival.txt', "ab") as f:
    #    np.savetxt(f, arrival, newline='\n', fmt='%s')
        
    #with open('output/icme_arrival_solo.txt', "ab") as f:
    #    np.savetxt(f, arrival_solo, newline='\n', fmt='%s')
    
    
    return time2_num, cme_r, cme_lat, cme_lon, cme_a, cme_b, cme_c, cme_id, cme_v, arr_time_fin, arr_time_err0, arr_time_err1,  arr_id, arr_hit, arr_time_fin_solo, arr_time_err0_solo, arr_time_err1_solo,  arr_id_solo, arr_hit_solo, time2
