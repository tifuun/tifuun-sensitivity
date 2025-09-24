import numpy as np
import os

Tp_amb = 273 # Ambient temperature
Tp_cabin = 290
Tp_co = 4
Tp_chip = 0.12

eta_M1_spill = 0.99
eta_M2_spill = 0.9
eta_WO = 0.99
eta_co = 0.65
eta_lens_antenna_rad = 0.81
eta_circuit = 0.32

d = 8e-3
tandelta = 4.805e-4
neffHDPE = 1.52
window_AR = True
#window_AR = False

F0 = 220 * 1e9
nF = 350
R = 500
F = F0 * (1 + 1/R)**np.arange(nF)

F_QO, eta_QO = np.load(os.path.join("tifuun_sensitivity", "data", "QOfilterstack.npy"))

cascade_list = [
        {"name" : "spillover M1",
             "eta_coup" : eta_M1_spill,
             "T_parasitic" : Tp_amb,
             "group" : "amb"}, # Spillover cascade on M1
            {"name" : "Ohmic M1", 
             "eta_coup" : "Ohmic-Al", 
             "T_parasitic" : Tp_amb,
             "group" : "amb"}, # Ohmic losses on M1
            {"name" : "Ohmic M2",
             "eta_coup" : "Ohmic-Al",
             "T_parasitic" : Tp_amb,
             "group" : "amb"}, # Ohmic losses on M2
            {"eta_coup" : eta_M2_spill, 
             "T_parasitic" : "atmosphere"}, 
            {"eta_coup" : eta_WO,
             "T_parasitic" : Tp_cabin,
             "group" : "cabin"}, # Spillover on WO
            {"eta_coup" : "Ohmic-Al",
             "T_parasitic" : Tp_cabin,
             "group" : "cabin"}, # Ohmic losses WO1
            {"eta_coup" : "Ohmic-Al",
             "T_parasitic" : Tp_cabin,
             "group" : "cabin"}, # Ohmic losses on WO 2
            {"thickness" : d, 
             "tandelta" : tandelta, 
             "neff" : neffHDPE,
             "AR" : window_AR,
             "T_parasitic_refl" : Tp_co,
             "T_parasitic_refr" : Tp_cabin,
             "cryo_window_flag" : True}, # Cryostat window
            {"eta_coup" : (F_QO, eta_QO),
             "T_parasitic" : Tp_co,
             "group" : "cryo"},
            {"eta_coup" : "Ohmic-Al",
             "T_parasitic" : Tp_co,
             "group" : "cryo"}, # Ohmic losses on CO 1
            {"eta_coup" : eta_co,
             "T_parasitic" : Tp_co,
             "group" : "cryo"}, 
            {"eta_coup" : "Ohmic-Al",                        
             "T_parasitic" : Tp_co,                          
             "group" : "cryo"}, # Ohmic losses on CO 2       
            {"eta_coup" : eta_lens_antenna_rad * eta_circuit,
             "T_parasitic" : Tp_chip}
            ]

instrument = {                         
        "f_ch"      : F,               
        "R"         : R,               
        "eta_IBF"   : 0.5,             
        "KID_excess_noise_factor" : 1.1
        }                              

telescope = {               
        "D_tel"     : 10,   
        "s_rms"     : 42e-6,
        "eta_taper" : 0.7,  
        "EL0"       : 60    
        }                   

