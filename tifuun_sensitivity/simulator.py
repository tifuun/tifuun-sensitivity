__all__ = ["spectrometer_sensitivity"]


# standard library
from typing import List, Union, Dict


# dependent packages
import numpy as np
import pandas as pd
from .atmosphere import eta_atm_func
from .instruments import eta_Al_ohmic_850, photon_NEP_kid, window_trans, average_over_filterbank, eta_mb_ruze
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
    snr: float = 5.0,
    obs_hours: float = 1.0,
    on_source_fraction: float = 0.4 * 0.9,
    on_off: bool = True
):
    """Calculate the sensitivity of a spectrometer.

    Parameters which are functions of frequency can be a vector (see Parameters).
    Output is a pandas DataFrame which containts results of simulation (see Returns).

    Parameters
    ----------
    cascade_list
        Dictionary containing cascades.
    F
        Frequency of the astronomical signal. Units: Hz.
    pwv
        Precipitable water vapour. Units: mm.
    EL
        Telescope elevation angle. Units: degrees.
    R
        Spectral resolving power in F/W_F where W_F is equivalent bandwidth.
        Units: None. See also: http://www.astrosurf.com/buil/us/spe2/hresol7.htm
    eta_IBF
        Fraction of the filter power transmission that is within the filter
        channel bandwidth. Units: None. The rest of the power is cross talk,
        picking up power that is in the bands of neighboring channels.
        This efficiency applies to the coupling to astronomical line signals.
        This efficiency does not apply to the coupling to continuum,
        including the the coupling to the atmosphere for calculating the NEP.
    KID_excess_noise_factor
        Need to be documented. Units: None.
    eta_ap
        Aperture efficiency. Units: None.
        Note that this is the aperture efficiency of the telescope, so not including lens antenna.
    telescope_diameter
        Diameter of the telescope. Units: m.
    Tb_cmb
        Brightness temperature of the CMB. Units: K.
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
    F
        Same as input.
    pwv
        Same as input.
    EL
        Same as input
    eta_atm
        Atmospheric transmission. Units: None.
    R
        Same as input.
    W_F_spec
        Equivalent bandwidth within the bandwidth of F/R. Units: Hz.
    W_F_cont
        Equivalent bandwidth of 1 channel including the power coupled
        outside of the filter channel band. Units: Hz.
    theta_maj
        Same as input.
    theta_min
        Same as input.
    eta_a
        Aperture efficiency. Units: None.
        See also: https://deshima.kibe.la/notes/324
    eta_mb
        Main beam efficiency. Units: None.
    eta_forward
        Forward efficiency. Units: None.
        See also: https://deshima.kibe.la/notes/324
    eta_sw
        Coupling efficiency from a point source to the cryostat window. Units: None.
    eta_window
        Transmission of the cryostat window. Units: None.
    eta_inst
        Instrument optical efficiency. Units: None.
        See also: https://arxiv.org/abs/1901.06934
    eta_circuit
        Same as input.
    Tb_sky
        Planck brightness temperature of the sky. Units: K.
    Pkid
        Power absorbed by the KID. Units: W.
    n_ph
        Photon occupation number. Units: None.
        See also: http://adsabs.harvard.edu/abs/1999ASPC..180..671R
    NEPkid
        Noise equivalent power at the KID with respect to the absorbed power.
        Units: W Hz^0.5.
    NEPinst
        Instrumnet NEP. Units: W Hz^0.5.
        See also: https://arxiv.org/abs/1901.06934
    MDLF
        Minimum Detectable Line Flux. Units: W/m^2.
    equivalent_Trx
        Equivalent receiver noise temperature. Units: K.
        at the moment this assumes Rayleigh-Jeans!
    """

    # Unpacking telescope dictionary
    D_tel = telescope_dict["D_tel"]
    s_rms = telescope_dict["s_rms"]
    eta_ap = telescope_dict["eta_ap"]
    EL = telescope_dict["EL0"]

    # Unpacking instrument dictionary
    F = instrument_dict["f_ch"]
    R = instrument_dict["R"]
    eta_IBF = instrument_dict["eta_IBF"]
    KID_excess_noise_factor = 1 if instrument_dict.get("KID_excess_noise_factor") is None else instrument_dict.get("KID_excess_noise_factor")
    
    # Equivalent Bandwidth of 1 channel.
    # Used for calculating loading and coupling to a continuum source
    W_F_cont = F*1e9 / R / eta_IBF
    # Used for calculating coupling to a line source,
    # with a linewidth not wider than the filter channel
    W_F_spec = F*1e9 / R

    # #############################################################
    # 1. Calculating loading power absorbed by the KID, and the NEP
    # #############################################################

    # .......................................................
    # Efficiencies for calculating sky coupling
    # .......................................................

    # Ohmic loss as a function of frequency, from skin effect scaling

    if not hasattr(F, "__len__"):
        F = np.array([F])
    
    if not hasattr(pwv, "__len__"):
        pwv = np.array([pwv])

    if R is not None:
        filterbank, F_sky, dF_sky = generateFilterbankFromR(R, F)

    else: # Read from file
        pass

    eta_atm = np.squeeze(eta_atm_func(F=F_sky, pwv=pwv, EL=EL))

    eta_cascade, psd_cascade, use_for_eta_inst = get_cascade(cascade_list, F_sky)
                
    psd_in = johnson_nyquist_psd(F_sky*1e9, Tb_cmb)
    psd_atm = johnson_nyquist_psd(F_sky*1e9, Tp_atm)
    
    # Start on branch 1, increment each time eta couples to "atmosphere". Set to zero when cryo window is encountered.
    branch_fwd = 1
    index_branches = []

    # Initialise array for eta_inst
    eta_inst = np.ones(F_sky.size)
    
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
    P_KID = np.nansum(psd_running, axis=0) * dF_sky * 1e9

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
                eta_branches[i] *= eta_stage

    # Before averaging over filterbank, sum contribution over branches.
    eta_fwd = average_over_filterbank(np.nansum(eta_branches, axis=0), filterbank) 

    # Average eta_inst over filter shapes
    eta_inst = average_over_filterbank(eta_inst, filterbank) 

    # For eta_sw, also smooth eta_atm over filter response
    eta_atm = average_over_filterbank(eta_atm, filterbank) 

    NEP = np.sqrt(np.nansum(photon_NEP_kid(F_sky[:,None]*1e9, psd_running), axis=0) * dF_sky * 1e9) * KID_excess_noise_factor
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

    eta_ap = eta_mb_ruze(F*1e9, eta_ap, s_rms)

    # Coupling from the "S"ource to outside of "W"indow
    eta_pol = 0.5  # Instrument is single polarization
    eta_sw = eta_pol * eta_atm * eta_ap * eta_fwd  # Source-Window coupling

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

    Trx = NEP_inst / k / np.sqrt(2 * W_F_cont) - average_over_filterbank(T_from_psd(F_sky*1e9, psd_in_front_of_cryo), filterbank)  # assumes RJ!

    # Photon occupation number
    # .........................................................
    n_ph = psd_KID / (h * F*1e9)

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
            "NEFD_line"     : spectral_NEFD,
            "NEFD_continuum": continuum_NEFD,
            "MDLF"          : MDLF,
            "equivalent_Trx": Trx,
            "n_ph"          : n_ph
            }
