__all__ = ["spectrometer_sensitivity"]


# standard library
from typing import List, Union, Dict


# dependent packages
import numpy as np
import pandas as pd
from .atmosphere import eta_atm_func
from .instruments import eta_Al_ohmic_850, photon_NEP2_kid, window_trans, average_over_filterbank, eta_ruze
from .physics import johnson_nyquist_psd, rad_trans, T_from_psd, c, h, k
from .filterbank import generateFilterbankFromR

from .cascade import get_cascade

# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]

# main functions
def calculator(
    cascade_list: List[Dict[any, any]],
    telescope_dict: Dict[any, any],
    instrument_dict: Dict[any, any],
    pwv: float = 0.5,
    Tb_cmb: ArrayLike = 2.725,
    Tp_atm: float = 273,
    snr: float = 1.0,
    obs_hours: float = 2*8.0,
    on_source_fraction: float = 0.4 * 0.9,
    on_off: bool = True
):
    """Calculate the sensitivity of a spectrometer.

    Parameters which are functions of frequency can be a vector (see Parameters).
    Output is a pandas DataFrame which containts results of simulation (see Returns).

    Parameters
    ----------
    cascade_list
        List containing cascade stage dictionaries.
    telescope_dict
        Dictionary containing telescope specifications.
    instrument_dict
        Dictionary containing instrument specifications.
    pwv
        Precipitable water vapour. Units: mm.
    Tb_cmb
        Brightness temperature of CMB. Units: K.
    Tp_atm
        Physical temperature of atmosphere. Units: K.
    snr
        Target signal to noise to be reached (for calculating the MDLF). Units: None.
    obs_hours
        Observing hours, including off-source time and the slew overhead
        between on- and off-source. Units: hours.
    on_source_fraction
        Fraction of the time on source (between 0. and 1.). Units: None.
    on_off
        If the observation involves on_off chopping, then the SNR degrades
        by sqrt(2) because the signal difference includes the noise twice.

    Returns
    ----------
    Dictionary with output. It contains the following fields:
        - "F_KID"         : Filter frequencies in Hz.
        - "F_sky"         : Sky frequencies in Hz.
        - "EL"            : Telescope elevation for calculation.
        - "PWV"           : Precipitable water vapor for calculation.
        - "R"             : Resolving power of filterbank.
        - "filterbank"    : 2D array containing filterbank.
        - "W_F_spec"      : Equivalent width of each filter, coupling to a spectral line.
        - "W_F_cont"      : Equivalent width of each filter, coupling to continuum (so including out-of-band loading).
        - "eta_atm"       : Transmission of atmopsphere, averaged over filterbank response.
        - "eta_ap"        : Aperture efficiency, which is taper x Ruze x atmospherically coupled efficiencies, averaged over filterbank.
        - "eta_fwd"       : Forward efficiency, i.e. fraction of beam coupling to sky, averaged over filterbank response.
        - "eta_sw"        : Total coupling of source to cryostat window, averaged over filterbank response.
        - "eta_window"    : Total transmission efficiency of cryostat window, averaged over filterbank response.
        - "eta_inst"      : Total instrument efficiency, averaged over filterbank response.
        - "Tb_sky"        : Brightness temperature of the sky, averaged over filterbank response.
        - "psd_KID"       : Power spectral density entering each KID, averaged over filterbank response.
        - "P_KID"         : Power entering each KID.
        - "NEP_KID"       : Noise equivalent power for each KID, at the KID itself.
        - "NEP_inst"      : Noise equivalent power for each KID, evaluated at the cryostat window.
        - "NET_sky"       : Noise equivalent temperature, evaluated before atmosphere. Units: K s^0.5
        - "NEFD_line"     : Noise equivalent flux density for each KID, coupling to a spectral line.
        - "NEFD_continuum": Noise equivalent flux density for each KID, coupling to continuum (so including out-of-band loading).
        - "MDLF"          : Minimum detectable line flux, for each KID.
        - "equivalent_Trx": Equivalent receiver noise temperature, for each KID, at the cryostat window.
        - "n_ph"          : Photon occupation number, defined as number of photons arriving per coherence time, for each KID.
        - "W_F_cont"      : Continuum equivalent width.
    """

    # Unpacking telescope dictionary
    D_tel = telescope_dict["D_tel"]
    s_rms = telescope_dict["s_rms"]
    eta_taper = telescope_dict["eta_taper"]
    EL = telescope_dict["EL0"]

    # Unpacking instrument dictionary
    F = instrument_dict["f_ch"]
    R = instrument_dict["R"]
    eta_IBF = instrument_dict["eta_IBF"]
    KID_excess_noise_factor = 1 if instrument_dict.get("KID_excess_noise_factor") is None else instrument_dict.get("KID_excess_noise_factor")
    
    # Equivalent Bandwidth of 1 channel.
    # Used for calculating loading and coupling to a continuum source
    W_F_cont = F / R / eta_IBF
    # Used for calculating coupling to a line source,
    # with a linewidth not wider than the filter channel
    W_F_spec = F / R

    # #############################################################
    # 1. Calculating loading power absorbed by the KID, and the NEP
    # #############################################################
    if not hasattr(F, "__len__"):
        F = np.array([F])
    
    if not hasattr(pwv, "__len__"):
        pwv = np.array([pwv])

    if R is not None:
        filterbank, F_sky, dF_sky = generateFilterbankFromR(R, F)

    else: # Read from file
        pass

    eta_atm = np.squeeze(eta_atm_func(F=F_sky, pwv=pwv, EL=EL))

    eta_cascade, psd_cascade, use_for_eta_inst, use_for_eta_ap = get_cascade(cascade_list, F_sky)
                
    psd_in = johnson_nyquist_psd(F_sky, Tb_cmb)
    psd_atm = johnson_nyquist_psd(F_sky, Tp_atm)
    
    # Start on branch 1, increment each time eta couples to "atmosphere". Set to zero when cryo window is encountered.
    branch_fwd = 1
    index_branches = []

    # Initialise array for eta_inst and eta_ap
    eta_inst = np.ones(F_sky.size)
    eta_ap = np.ones(F_sky.size)
    
    # Do first stage outside loop, is always the same anyways...
    psd_running = eta_atm * psd_in + (1 - eta_atm) *  psd_atm
    psd_sky = average_over_filterbank(psd_running, filterbank)

    psd_in_front_of_cryo_set = False

    # Helper list for storing intermediate transmitted psd
    psd_l = []

    psd_l.append(average_over_filterbank(psd_running, filterbank))

    eta_window_set = False

    for idx, (eta_stage, psd_stage) in enumerate(zip(eta_cascade, psd_cascade)):
        if isinstance(psd_stage, str):
            psd_stage = (1 - eta_atm) * psd_atm
            branch_fwd += 1 
        
        if use_for_eta_ap[idx]:
            eta_ap *= eta_stage     

        if use_for_eta_inst[idx]:
            # If stage is inside cryostat (including window), it counts towards eta_inst
            eta_inst *= eta_stage 
        
            # Also check, if this is first stage inside instrument -> assign to eta_window
            if eta_window_set == False:
                eta_window = average_over_filterbank(eta_stage, filterbank)
                eta_window_set = True

            # Also, stop incorporating eta into eta_fwd
            branch_fwd = 0

            if psd_in_front_of_cryo_set == False:
                # Save psd up till now for Trx calculation
                psd_in_front_of_cryo = psd_running
                psd_in_front_of_cryo_set = True

        if idx == len(eta_cascade) - 1:
            eta_stage = np.tile(eta_stage, F.size).reshape(filterbank.shape) * filterbank

        psd_running = rad_trans(psd_running, psd_stage, eta_stage)

        index_branches.append(branch_fwd)


    psd_KID = np.nansum(psd_running, axis=0) / np.nansum(filterbank, axis=0)
    P_KID = np.nansum(psd_running, axis=0) * dF_sky 
    n_branches = np.unique(index_branches).size - 1

    # Initialise branches
    eta_branches = []
    for n in range(n_branches):
        eta_branches.append(np.ones(F_sky.size))

    branch_counter = 1
    eta_use_flag = False
    for idx_branch, eta_stage in zip(index_branches, eta_cascade):
        if idx_branch == 0:
            break
        
        if idx_branch > branch_counter:
            eta_use = 1 - eta_stage
            eta_use_flag = True
            branch_counter += 1

        for i in range(idx_branch):
            if i == n_branches - 1 and eta_use_flag: # We are at last current branch AND it is start of that branch
                eta_branches[i] *= eta_use
                eta_use_flag = False
            else:
                pass
                eta_branches[i] *= eta_stage

    # Before averaging over filterbank, sum contribution over branches.
    eta_fwd = average_over_filterbank(np.nansum(eta_branches, axis=0), filterbank) 

    # Average eta_inst over filter shapes
    eta_inst = average_over_filterbank(eta_inst, filterbank) 

    # For eta_sw, also smooth eta_atm over filter response
    eta_atm = average_over_filterbank(eta_atm, filterbank) 

    NEP = np.sqrt(np.nansum(photon_NEP2_kid(F_sky[:,None], psd_running), axis=0) * dF_sky) * KID_excess_noise_factor
    NEP_inst = NEP / eta_inst  # Instrument NEP    

    # ##############################################################
    # 2. Calculating source coupling and sensitivtiy (MDLF and NEFD)
    # ##############################################################

    # Efficiencies
    # .........................................................

    Ag = np.pi * (D_tel / 2.0) ** 2.0  # Geometric area of the telescope
    #omega_mb = np.pi * theta_maj * theta_min / np.log(2) / 4  # Main beam solid angle
    #omega_a = omega_mb / eta_mb  # beam solid angle
    #Ae = (c / (F/350)) ** 2 / omega_a  # Effective Aperture (m^2): lambda^2 / omega_a
    #eta_a = Ae / Ag  # Aperture efficiency

    # Here, I define the illumination efficiency as the product of taper and Ruze efficiencies
    eta_illum = eta_taper * eta_ruze(F_sky, s_rms)
    
    # Calculate aperture efficiency using illumination efficiency and accumulated stages that terminate on-sky
    eta_ap *= eta_illum

    eta_ap = average_over_filterbank(eta_ap, filterbank)
    eta_illum = average_over_filterbank(eta_illum, filterbank)

    # Coupling from the "S"ource to outside of "W"indow
    # Note that not the aperture but illumination efficiency are used, because the forward efficiency includes the stages going to atmosphere
    eta_pol = 0.5  # Instrument is single polarization
    eta_sw = eta_pol * eta_atm * eta_illum * eta_fwd  # Source-Window coupling

    # NESP: Noise Equivalent Source Power (an intermediate quantitiy)
    # .........................................................

    NESP = NEP_inst / eta_sw  # Noise equivalnet source power

    # NEF: Noise Equivalent Flux (an intermediate quantitiy)
    # .........................................................

    # From this point, units change from Hz^-0.5 to t^0.5
    # sqrt(2) is because NEP is defined for 0.5 s integration.

    NEF = NESP / Ag / np.sqrt(2)  # Noise equivalent flux

    # If the observation is involves ON-OFF sky subtraction,
    # Subtraction of two noisy sources results in sqrt(2) increase in noise.

    if on_off:
        NEF *= np.sqrt(2)

    NET = NEP / (np.sqrt(2) * eta_inst * eta_fwd * W_F_cont * k)

    # MDLF (Minimum Detectable Line Flux)
    # .........................................................

    # Note that eta_IBF does not matter for MDLF because it is flux.

    MDLF = NEF * snr / np.sqrt(obs_hours * on_source_fraction * 60.0 * 60.0)

    # NEFD (Noise Equivalent Flux Density)
    # .........................................................

    spectral_NEFD = NEF / W_F_spec
    continuum_NEFD = NEF / W_F_cont  # = spectral_NEFD * eta_IBF < spectral_NEFD

    # Equivalent Trx
    # .........................................................

    Trx = NEP_inst / k / np.sqrt(2 * W_F_cont) - average_over_filterbank(T_from_psd(F_sky, psd_in_front_of_cryo), filterbank)  # assumes RJ!

    # Photon occupation number
    # .........................................................
    n_ph = psd_KID / (h * F)

    # ############################################
    # 3. Output results as dictionary
    # ############################################

    return {
            "F_KID"         : F,
            "F_sky"         : F_sky,
            "EL"            : EL,
            "PWV"           : pwv,
            "R"             : R,
            "filterbank"    : filterbank,
            "W_F_spec"      : W_F_spec,
            "W_F_cont"      : W_F_cont,
            "eta_atm"       : eta_atm,
            "eta_ap"        : eta_ap,
            "eta_fwd"       : eta_fwd,
            "eta_sw"        : eta_sw,
            "eta_window"    : eta_window,
            "eta_inst"      : eta_inst,
            "Tb_sky"        : T_from_psd(F, psd_sky),
            "psd_KID"       : psd_KID,
            "P_KID"         : P_KID,
            "NEP_KID"       : NEP,
            "NEP_inst"      : NEP_inst,
            "NET_sky"       : NET,
            "NEFD_line"     : spectral_NEFD,
            "NEFD_continuum": continuum_NEFD,
            "MDLF"          : MDLF,
            "equivalent_Trx": Trx,
            "n_ph"          : n_ph,
            "W_F_cont"      : W_F_cont
            }
