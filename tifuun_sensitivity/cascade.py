import numpy as np
import os
import yaml
import csv

from typing import List, Union, Dict, Tuple

from .instruments import eta_Al_ohmic_850, window_trans
from .physics import johnson_nyquist_psd

def eta_Al_ohmic(F_sky: np.ndarray) -> np.ndarray: 
    """Calculate Ohmic losses for aluminium over array of sky frequencies.
    
    Parameters
    ----------
    F_sky
        Numpy array containing sky frequencies. Units: GHz
    
    Returns
    ----------
    Array with eta values for Ohmic losses.
    """

    return 1.0 - (1.0 - eta_Al_ohmic_850) * np.sqrt(F_sky / 850)

def sizer(eta: Union[np.ndarray, float], 
           F_sky: np.ndarray, 
           F_eta: np.ndarray = None
) -> np.ndarray:
    """Resize efficiency term to new size.

    Used to vectorize or interpolate on efficiency terms.
    If efficiency is a scalar, an array is returned with the same size as F_sky.
    If efficiency is an array with different size then F_sky, an array containing frequencies at which eta is evaluated should also be passed.
    A 1D interpolation on F_sky is then performed to evaluate eta on F_sky.
    If efficiency is array with same size as F_sky, it is returned as-is. 
    Responisibility to verify if the efficiencies are evaluated on the same frequencies as present in F_sky is placed on the user.

    Parameters
    ----------
    eta
        Efficiency term.
    F_sky
        Numpy array containing sky frequencies. Units: GHz
    F_eta
        Numpy array containing frequencies at which eta is evaluated.
        Should only be passed when 1D interpolation is required and defaults to None.
    
    Returns
    ----------
    Array with eta values, depending on input (see above).
    """

    if not hasattr(eta, "__len__"):
        return eta * np.ones(F_sky.size)

    elif F_eta is not None:
        idx_sorted = np.argsort(F_eta)
        return np.interp(F_sky, 
                         F_eta[idx_sorted], 
                         eta[idx_sorted])

    else:
        return eta

def read_from_folder(cascade_folder: str,
                     yaml_name: str = "cascade.yaml"
                     ) -> List[Dict[any, any]]:
    """
    Generate a cascade list from a cascade folder.
    The folder should contain a YAML file containing the cascadelist.
    Any vector-valued efficiency terms should be provided inside the folder as a CSV file, with the first column containing frequencies at which the terms are evaluated and the second column containing the terms themselves.
    Then, the CSV can be referenced inside the YAML by passing the CSV name (including .csv) to the `eta_coup` field inside the YAML.

    Parameters
    ----------
    cascade_folder
        String containing path to folder containing cascade YAML and any related CSV files.

    yaml_name
        String containing the name of the YAML file containing the cascade.
        Defaults to 'cascade.yaml'.

    Returns
    ----------
    List containing the cascade.
    """

    assert(os.path.exists(cascade_folder))
    assert(os.path.exists(yaml_path := os.path.join(cascade_folder, 
                                       yaml_name)))

    with open(yaml_path) as stream:
        try:
            cascade_list = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for stage in cascade_list:
        for key, item in stage.items():
            if isinstance(item, str):
                if item.endswith(".csv"):
                    assert(os.path.exists(csv_path := os.path.join(cascade_folder,
                                                                   item)))
                    freq = []
                    vals = []
                    with open(csv_path, 'r', newline='') as csvfile:                
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                            freq.append(float(row[0]))
                            vals.append(float(row[1]))

                    stage[key] = (np.array(freq), np.array(vals))

    return cascade_list

def save_cascade(cascade_list: List[Dict[any, any]],
                 save_folder: str,
                 yaml_name: str = "cascade") -> None:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    name_index = 0

    cascade_to_write = []

    for stage in cascade_list:
        stage_dict = {}

        for key, item in stage.items():
            if isinstance(item, tuple):
                assert(item[0].size == item[1].size)
                np.savetxt(os.path.join(save_folder, 
                                        f"{name_index}.csv"), 
                           np.column_stack(item),
                           delimiter = ",")

                stage_dict[key] = f"{name_index}.csv"
                
                name_index += 1

            else:
                stage_dict[key] = item

        cascade_to_write.append(stage_dict)

    with open(os.path.join(save_folder, f"{yaml_name}.yaml"), 'w') as outfile:
              yaml.dump(cascade_to_write, outfile)

def get_cascade(cascade_list: List[Dict[any, any]],
                F_sky: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate a cascade list, consisting of efficiency and psd per stage.

    Parameters
    ----------
    cascade_list
        List containing, per element, the efficiency and coupling temperature of each stage in the cascade.
        For reflective stages, the dictionary should contain either:
            - A single eta and temperature
            - A tuple with efficiencies and frequencies at which these are defined, and a temperature
        
        For refractive stages, the dictionary should contain:
            - thickness of dielectric in meters, loss tangent, effective refractive index, whether to use AR coating, temperature seen in reflection coming from the ISS, and temperature seen in refraction.

    F_sky
        Array with sky frequencies. Units: GHz.
    
    Returns
    ----------
    List with list of arrays containing efficiencies as first element, and list containing arrays of psd as second element. 
    """


    group_list = []
    
    for cascade in cascade_list:
        group_list.append(cascade.get("group"))
    
    group_list_red_uniq = list(dict.fromkeys([x for x in group_list if x is not None]))

    idx_group_list = [i if label == None else label for i, label in enumerate(group_list)]
    cascade_type_list = np.array([0 if x.get("eta_coup") is not None else 1 for x in cascade_list])
    idx_group_list_uniq = list(dict.fromkeys(idx_group_list))

    to_delete = []
    for group_label in group_list_red_uniq:
        index_list = [i for i, label in enumerate(idx_group_list) if label == group_label]
        idx_group_list_uniq[idx_group_list_uniq.index(group_label)] = index_list

        to_delete.extend(index_list[:-1])

    cascade_type_list_uniq = np.delete(cascade_type_list, to_delete)
    idx_group_list_uniq = [[x] if hasattr(x, "__len__") == False else x for x in idx_group_list_uniq] 

    # Now get eta for groups

    group_running = None
    all_eta = []
    all_psd = []
    use_for_eta_inst = []
    use_for_eta_ap = []
    eta_inst_flag = 0

    for casc_t, idx_group in zip(cascade_type_list_uniq, idx_group_list_uniq):
        # This cascade group is reflective
        eta_ap_flag = 0
        if casc_t == 0:
            eta_grouped = np.ones(F_sky.size)

            if (T_casc := cascade_list[idx_group[0]].get("T_parasitic")) == "atmosphere":
                all_psd.append(T_casc) # Group couples to atmosphere: calculate psd in rad trans loop.
                eta_ap_flag = 1
            else:
                all_psd.append(johnson_nyquist_psd(F_sky, T_casc)) # Calculate psd for T_parasitic

            for idx_g in idx_group:
                eta_interp_flag = False
                
                if isinstance(eta := cascade_list[idx_g].get("eta_coup"), tuple): # eta_coup is a tuple; array of eta and frequencies
                    assert(len(eta) == 2)
                    
                    eta, F_eta = eta

                    assert(eta.size == F_eta.size)
                    
                    eta_interp_flag = True

                elif eta == "Ohmic-Al": # generate vector with eta of Aluminium
                    eta = eta_Al_ohmic(F_sky) 

                elif eta_interp_flag:
                    eta_grouped *= sizer(eta, F_sky, F_eta)
                else:
                    eta_grouped *= sizer(eta, F_sky)
            
            all_eta.append(eta_grouped)
            
            use_for_eta_inst.append(eta_inst_flag)
            use_for_eta_ap.append(eta_ap_flag)
        
        if casc_t == 1:
            for idx_g in idx_group:
                if (casc := cascade_list[idx_g]).get("cryo_window_flag"):
                    eta_inst_flag = 1
                etas, psds = window_trans(F_sky,
                                          casc.get("thickness"), 
                                          casc.get("tandelta"), 
                                          casc.get("neff"), 
                                          casc.get("AR"), 
                                          casc.get("T_parasitic_refl"), 
                                          casc.get("T_parasitic_refr")) 

                all_eta.extend(etas)
                all_psd.extend(psds)
        
                use_for_eta_inst.extend([eta_inst_flag for _ in range(len(etas))])
                use_for_eta_ap.extend([eta_ap_flag for _ in range(len(etas))])
    
    return all_eta, all_psd, use_for_eta_inst, use_for_eta_ap
