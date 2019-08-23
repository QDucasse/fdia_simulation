# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:44:23 2019

@author: qde
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from fdia_simulation.models            import Radar, PeriodRadar, Track
from fdia_simulation.filters           import  (RadarFilterCA, MultipleRadarsFilterCA, MultiplePeriodRadarsFilterCA,
                                               RadarFilterCV, MultipleRadarsFilterCV, MultiplePeriodRadarsFilterCV,
                                               RadarFilterCT, MultipleRadarsFilterCT, MultiplePeriodRadarsFilterCT,
                                               RadarFilterTA, MultipleRadarsFilterTA, MultiplePeriodRadarsFilterTA,
                                               RadarIMM)
from fdia_simulation.attackers         import (Attacker,DOSAttacker,DriftAttacker, CumulativeDriftAttacker,
                                               PeriodAttacker,DOSPeriodAttacker,DriftPeriodAttacker,CumulativeDriftPeriodAttacker)
from fdia_simulation.benchmarks        import Benchmark
from fdia_simulation.anomaly_detectors import MahalanobisDetector,EuclidianDetector

# ==============================================================================
# ================================ Window Generation ===========================
window = tk.Tk()
window.title("False Data Injection an Air Traffic Control Tower")
window.geometry("650x500")

# ==============================================================================
# ============================= Frames Definitions =============================

traj_frame   = tk.Frame(window,padx = 3, pady = 10)
rad_frame    = tk.Frame(window,padx = 3, pady = 10)
filt_frame   = tk.Frame(window,padx = 3, pady = 10)
att_frame    = tk.Frame(window,padx = 3, pady = 10)
launch_frame = tk.Frame(window,padx = 3, pady = 10)

window.grid_rowconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=1)

traj_frame.grid(row = 0, sticky="n")
rad_frame.grid(row = 1, sticky="n")
filt_frame.grid(row = 2, sticky="n")
att_frame.grid(row = 3, sticky="n")
launch_frame.grid(row = 4, sticky="n")

# ==============================================================================
# ============================== Trajectory Frame ==============================

label_traj = tk.Label(traj_frame,text = "Trajectory")
label_traj.grid(column = 0, row = 0)

trajectories = [ "Take off", "Landing"]
traj_combobox = ttk.Combobox(traj_frame,values = trajectories)
traj_combobox.current(1)
traj_combobox.grid(column = 0, row = 1)

# ==============================================================================
# =============================== Radar(s) Frame ===============================
label_rad = tk.Label(rad_frame,text = "Radar(s)")
label_rad.grid(column = 0, columnspan = 10,row = 0)

# Radar type selection
type_label = tk.Label(rad_frame,text = "Radar type:")
type_label.grid(column = 3, row = 1)
types = [ "Radar","PeriodRadar"]
rad_combobox = ttk.Combobox(rad_frame,values = types)
rad_combobox.current(0)
rad_combobox.grid(column = 4, row = 1)

std_var = tk.IntVar()
prec_var = tk.IntVar()
std_var.set(1)
prec_var.set(1)
# Radar Selection
std_rad_check  = tk.Checkbutton(rad_frame, text="Standard",variable = std_var)
std_rad_check.grid(column = 0, row = 2)
std_rad_x_label = tk.Label(rad_frame,text = "x: ")
std_rad_x_label.grid(column = 1, row = 2)
std_rad_x_entry = tk.Entry(rad_frame)
std_rad_x_entry.insert("end","-6000")
std_rad_x_entry.grid(column = 2, row = 2)
std_rad_y_label = tk.Label(rad_frame,text = "y: ")
std_rad_y_label.grid(column = 3, row = 2)
std_rad_y_entry = tk.Entry(rad_frame)
std_rad_y_entry.insert("end","1000")
std_rad_y_entry.grid(column = 4, row = 2)
std_rad_dt_label = tk.Label(rad_frame,text = "dt: ")
std_rad_dt_label.grid(column = 5, row = 2)
std_rad_dt_entry = tk.Entry(rad_frame)
std_rad_dt_entry.insert("end","0.2")
std_rad_dt_entry.grid(column = 6, row = 2)



prec_rad_check = tk.Checkbutton(rad_frame, text="Precision",variable = prec_var)
prec_rad_check.grid(column = 0, row = 3)
prec_rad_x_label = tk.Label(rad_frame,text = "x: ")
prec_rad_x_label.grid(column = 1, row = 3)
prec_rad_x_entry = tk.Entry(rad_frame)
prec_rad_x_entry.insert("end","1000")
prec_rad_x_entry.grid(column = 2, row = 3)
prec_rad_y_label = tk.Label(rad_frame,text = "y: ")
prec_rad_y_label.grid(column = 3, row = 3)
prec_rad_y_entry = tk.Entry(rad_frame)
prec_rad_y_entry.insert("end","4000")
prec_rad_y_entry.grid(column = 4, row = 3)
prec_rad_dt_label = tk.Label(rad_frame,text = "dt: ")
prec_rad_dt_label.grid(column = 5, row = 3)
prec_rad_dt_entry = tk.Entry(rad_frame)
prec_rad_dt_entry.insert("end","0.05")
prec_rad_dt_entry.grid(column = 6, row = 3)


# ==============================================================================
# ================================ Filter Frame ================================

label_filter = tk.Label(filt_frame,text = "Filters")
label_filter.grid(column = 2, row = 0)

ca_var = tk.IntVar()
cv_var = tk.IntVar()
ct_var = tk.IntVar()
ta_var = tk.IntVar()
ca_var.set(1)
cv_var.set(1)
ct_var.set(1)
ta_var.set(1)

detec_models = [ "None", "Mahalanobis", "Euclidian"]

##### Constant Velocity filter
cv_check  = tk.Checkbutton(filt_frame, text="Constant Velocity", variable = cv_var)
cv_check.grid(column = 0, row = 1)
# q selection: Label + Entry
pn_cv_label = tk.Label(filt_frame,text = "Process noise: ")
pn_cv_label.grid(column = 1, row = 1)
pn_cv_entry = tk.Entry(filt_frame)
pn_cv_entry.insert("end","10")
pn_cv_entry.grid(column = 2, row = 1)
# Detector selection : Label + Combobox
cv_detec_label = tk.Label(filt_frame,text = "Detector model: ")
cv_detec_label.grid(column = 3, row = 1)
cv_detec_combobox = ttk.Combobox(filt_frame,values = detec_models)
cv_detec_combobox.current(0)
cv_detec_combobox.grid(column = 4, row = 1)

##### Constant Acceleration filter
ca_check  = tk.Checkbutton(filt_frame, text="Constant Acceleration", variable = ca_var)
ca_check.grid(column = 0, row = 2)
# q selection: Label + Entry
pn_ca_label = tk.Label(filt_frame,text = "Process noise: ")
pn_ca_label.grid(column = 1, row = 2)
pn_ca_entry = tk.Entry(filt_frame)
pn_ca_entry.insert("end","400")
pn_ca_entry.grid(column = 2, row = 2)
# Detector selection : Label + Combobox
ca_detec_label = tk.Label(filt_frame,text = "Detector model: ")
ca_detec_label.grid(column = 3, row = 2)
ca_detec_combobox = ttk.Combobox(filt_frame,values = detec_models)
ca_detec_combobox.current(0)
ca_detec_combobox.grid(column = 4, row = 2)

##### Constant Turn filter
ct_check  = tk.Checkbutton(filt_frame, text="Constant Turn", variable = ct_var)
ct_check.grid(column = 0, row = 3)
# q selection: Label + Entry
pn_ct_label = tk.Label(filt_frame,text = "Process noise: ")
pn_ct_label.grid(column = 1, row = 3)
pn_ct_entry = tk.Entry(filt_frame)
pn_ct_entry.insert("end","25")
pn_ct_entry.grid(column = 2, row = 3)
# Detector selection : Label + Combobox
ct_detec_label = tk.Label(filt_frame,text = "Detector model: ")
ct_detec_label.grid(column = 3, row = 3)
ct_detec_combobox = ttk.Combobox(filt_frame,values = detec_models)
ct_detec_combobox.current(0)
ct_detec_combobox.grid(column = 4, row = 3)

##### Thrust Acceleration filter
ta_check  = tk.Checkbutton(filt_frame, text="Thrust Acceleration", variable = ta_var)
ta_check.grid(column = 0, row = 4)
# q selection: Label + Entry
pn_ta_label = tk.Label(filt_frame,text = "Process noise: ")
pn_ta_label.grid(column = 1, row = 4)
pn_ta_entry = tk.Entry(filt_frame)
pn_ta_entry.insert("end","350")
pn_ta_entry.grid(column = 2, row = 4)
# Detector selection : Label + Combobox
ta_detec_label = tk.Label(filt_frame,text = "Detector model: ")
ta_detec_label.grid(column = 3, row = 4)
ta_detec_combobox = ttk.Combobox(filt_frame,values = detec_models)
ta_detec_combobox.current(0)
ta_detec_combobox.grid(column = 4, row = 4)





# ==============================================================================
# =============================== Attacker Frame ===============================

label_att = tk.Label(att_frame,text = "Attacker")
label_att.grid(column = 0, columnspan = 10, row = 0)

label_att_type = tk.Label(att_frame,text = "Attacker Type:")
label_att_type.grid(column = 0, row = 1)
att_models = [ "None","DOS", "Constant Drift", "Cumulative Drift"]
att_combobox = ttk.Combobox(att_frame,values = att_models)
att_combobox.current(3)
att_combobox.grid(column = 0, row = 2)

label_type = tk.Label(att_frame,text = "Attacked Radar:")
label_type.grid(column = 1, row = 1)
rad_models = [ "Standard", "Precision"]
att_rad_combobox = ttk.Combobox(att_frame,values = rad_models)
att_rad_combobox.current(0)
att_rad_combobox.grid(column = 1, row = 2)

# Attaker t0 label & entry
t0_label = tk.Label(att_frame,text = "t0:")
t0_label.grid(column = 2, row = 1)
t0_entry = tk.Entry(att_frame, width = 20)
t0_entry.insert("end","300")
t0_entry.grid(column = 3, row = 1)
# New radar y label & entry
time_label = tk.Label(att_frame,text = "time:")
time_label.grid(column = 2, row = 2)
time_entry = tk.Entry(att_frame, width = 20)
time_entry.insert("end","500")
time_entry.grid(column = 3, row = 2)
# New radar dt label & entry
drift_label = tk.Label(att_frame,text = "drift:")
drift_label.grid(column = 2, row = 3)
drift_entry = tk.Entry(att_frame, width = 20)
drift_entry.insert("end","0 0 1")
drift_entry.grid(column = 3, row = 3)


# ==============================================================================
# ============================== Launching Frame ===============================
#Dictionaries initialization

# Current radar button config
def prepare_launch_benchmark():
    refresh_globals()
    generate_dicts()
    trajectory = traj_combobox.get()
    generate_states(trajectory)
    radar_type = rad_combobox.get()
    generate_radars(radar_type)
    generate_filters(radar_type)
    generate_attacker(radar_type)
    with_nees =  nees_var.get()
    launch_benchmark(with_nees)

nees_var = tk.IntVar()
nees_check = tk.Checkbutton(launch_frame, text="Normalized Estimated Error Squared", variable = nees_var)
nees_check.grid(column = 0, row = 0)

launch_button = tk.Button(launch_frame,text = "Launch Benchmark!",command = prepare_launch_benchmark)
launch_button.grid(column = 0, row = 1)


##### Values needed by the benchmark

std_dict  = {}
prec_dict = {}
ca_dict   = {}
ct_dict   = {}
ta_dict   = {}
cv_dict   = {}
att_dict  = {}
states = []
radars = []
filter = None
filters = []
attacker = None
x0,y0,z0 = 0,0,0

def refresh_globals():
    global std_dict,prec_dict,ca_dict,ct_dict,ta_dict,cv_dict,att_dict
    global states,radars,filter,filters,attacker,x0,y0,z0
    std_dict  = {}
    prec_dict = {}
    ca_dict   = {}
    ct_dict   = {}
    ta_dict   = {}
    cv_dict   = {}
    att_dict  = {}
    states = []
    radars = []
    filter = None
    filters = []
    attacker = None
    x0,y0,z0 = 0,0,0


def generate_dicts():
    # Radars dictionaries
    global std_dict
    std_dict = {
        "is_chosen":std_var.get(),
        "x":std_rad_x_entry.get(),
        "y":std_rad_y_entry.get(),
        "dt":std_rad_dt_entry.get()
    }
    global prec_dict
    prec_dict = {
        "is_chosen":prec_var.get(),
        "x":prec_rad_x_entry.get(),
        "y":prec_rad_y_entry.get(),
        "dt":prec_rad_dt_entry.get()
    }
    # Filters dictionaries
    global ca_dict
    ca_dict = {
        "is_chosen": ca_var.get(),
        "q": pn_ca_entry.get(),
        "detector": ca_detec_combobox.get(),
        "name":RadarFilterCA,
        "mname":MultipleRadarsFilterCA,
        "mpname":MultiplePeriodRadarsFilterCA
    }
    global cv_dict
    cv_dict = {
        "is_chosen": cv_var.get(),
        "q": pn_cv_entry.get(),
        "detector": cv_detec_combobox.get(),
        "name":RadarFilterCV,
        "mname":MultipleRadarsFilterCV,
        "mpname":MultiplePeriodRadarsFilterCV
    }
    global ct_dict
    ct_dict = {
        "is_chosen": ct_var.get(),
        "q": pn_ct_entry.get(),
        "detector": ct_detec_combobox.get(),
        "name":RadarFilterCT,
        "mname":MultipleRadarsFilterCT,
        "mpname":MultiplePeriodRadarsFilterCT
    }
    global ta_dict
    ta_dict = {
        "is_chosen": ta_var.get(),
        "q": pn_ta_entry.get(),
        "detector": ta_detec_combobox.get(),
        "name":RadarFilterTA,
        "mname":MultipleRadarsFilterTA,
        "mpname":MultiplePeriodRadarsFilterTA
    }
    # Attacker dictionary
    global att_dict
    att_dict = {
        "type": att_combobox.get(),
        "att_radar": att_rad_combobox.get(),
        "t0": t0_entry.get(),
        "time": time_entry.get(),
        "drift": drift_entry.get()
    }
#### Benchmark useful values

def generate_states(traj_name):
    global states, x0, y0, z0
    trajectory = Track()
    method_name = getattr(trajectory,"gen_"+ traj_name.replace(" ","").lower())
    states = method_name()
    x0, y0, z0 = trajectory.initial_position(states)

def generate_radars(radar_type):
    global std_dict
    global prec_dict
    global radars
    if radar_type == "Radar":
        if std_dict["is_chosen"]:
            x = int(std_dict["x"])
            y = int(std_dict["y"])
            std_radar = Radar(x = x, y = y, r_std = 5., theta_std = 0.005, phi_std = 0.005)
            radars.append(std_radar)
        if prec_dict["is_chosen"]:
            x = int(prec_dict["x"])
            y = int(prec_dict["y"])
            prec_radar = Radar(x = x, y = y)
            radars.append(prec_radar)

    elif radar_type == "PeriodRadar":
        if std_dict["is_chosen"]:
            x = int(std_dict["x"])
            y = int(std_dict["y"])
            dt = float(std_dict["dt"])
            std_radar = PeriodRadar(x = x, y = y, r_std = 5., theta_std = 0.005,
                                    phi_std = 0.005, dt = dt)
            radars.append(std_radar)
        if prec_dict["is_chosen"]:
            x = int(prec_dict["x"])
            y = int(prec_dict["y"])
            dt = float(prec_dict["dt"])
            prec_radar = PeriodRadar(x = x, y = y, dt = dt)
            radars.append(prec_radar)

def generate_filters(radar_type):
    global ca_dict, cv_dict, ct_dict, ta_dict, filter, filters, radars, x0, y0, z0
    filter_dicts = [cv_dict, ca_dict, ct_dict, ta_dict]
    current_filters = []
    for filter_dict in filter_dicts:
        if filter_dict["is_chosen"]:
            current_filters.append(filter_dict)
    # Filter instanciation
    for filter_dict in current_filters:
        detector = None
        if not(filter_dict["detector"] == "None"):
            detector_type = globals()[filter_dict["detector"]+"Detector"]
            detector = detector_type()
        q = float(filter_dict["q"])
        if radar_type == "Radar" and len(radars)==1:
            filter_class = filter_dict["name"]
        elif radar_type == "Radar" and len(radars)==2:
            filter_class = filter_dict["mname"]
        elif radar_type == "PeriodRadar" and len(radars)==2:
            filter_class = filter_dict["mpname"]
        current_filter = filter_class(radars = radars, q = q, detector = detector,
                                      x0 = x0, y0 = y0, z0 = z0)
        filters.append(current_filter)

    # IMM Generation
    if len(filters)==1:
        filter = filters[0]
    if len(filters)==2:
        mu = [0.5, 0.5]
        trans = np.array([[0.998, 0.02],
                          [0.100, 0.900]])
        filter = RadarIMM(filters, mu, trans)
    if len(filters)==3:
        mu = [0.33, 0.33, 0.33]
        trans = np.array([[0.998, 0.001, 0.001],
                          [0.050, 0.900, 0.050],
                          [0.001, 0.001, 0.998]])
        filter = RadarIMM(filters, mu, trans)
    if len(filters)==4:
        mu = [0.25, 0.25, 0.25, 0.25]
        trans = np.array([[0.997, 0.001, 0.001, 0.001],
                          [0.050, 0.850, 0.050, 0.050],
                          [0.001, 0.001, 0.997, 0.001],
                          [0.001, 0.001, 0.001, 0.997]])
        filter = RadarIMM(filters, mu, trans)

def generate_attacker(radar_type):
    global att_dict, filter, attacker, radars
    # T0 argument
    t0 = int(att_dict["t0"])
    # Time argument
    time = int(att_dict["time"])
    # Drift argument
    drift = att_dict["drift"].split(" ")
    attack_drift = np.array([[int(drift[0]),int(drift[1]),int(drift[2])]]).T
    if att_dict["att_radar"] == "Standard":
        attacked_radar = radars[0]
        radar_position = 0
    if att_dict["att_radar"] == "Precision":
        attacked_radar = radars[1]
        radar_position = 1

    # Attacker class
    attacker = None
    if radar_type == "Radar":
        if att_dict["type"] == "DOS":
            attacker = DOSAttacker(filter = filter, t0 = t0, time = time,
                                   radar = attacked_radar, radar_pos = radar_position)
        elif att_dict["type"] == "Constant Drift":
            attacker = DriftAttacker(filter = filter, t0 = t0, time = time,
                                     attack_drift = attack_drift,
                                     radar = attacked_radar, radar_pos = radar_position)
        elif att_dict["type"] == "Cumulative Drift":
            attacker = CumulativeDriftAttacker(filter = filter, t0 = t0, time = time,
                                               delta_drift = attack_drift,
                                               radar = attacked_radar, radar_pos = radar_position)
    if radar_type == "PeriodRadar":
        if att_dict["type"] == "DOS":
            attacker = DOSPeriodAttacker(filter = filter, t0 = t0, time = time,
                                         radar = attacked_radar, radar_pos = radar_position)
        elif att_dict["type"] == "Constant Drift":
            attacker = DriftPeriodAttacker(filter = filter, t0 = t0, time = time,
                                           attack_drift = attack_drift,
                                           radar = attacked_radar, radar_pos = radar_position)
        elif att_dict["type"] == "Cumulative Drift":
            attacker = CumulativeDriftPeriodAttacker(filter = filter, t0 = t0, time = time,
                                                     delta_drift = attack_drift,
                                                     radar = attacked_radar, radar_pos = radar_position)


def launch_benchmark(with_nees):
    global radars, filter, attacker, states
    benchmark = Benchmark(radars = radars,radar_filter = filter,
                          states = states, attacker = attacker)
    benchmark.launch_benchmark(with_nees = with_nees)

## Main loop
window.mainloop()
