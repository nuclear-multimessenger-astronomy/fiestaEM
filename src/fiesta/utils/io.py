import copy
import numpy as np
import pandas as pd
from astropy.time import Time
import scipy.interpolate as interp
from jaxtyping import Array, Float


# TODO: Check if these can be moved/deleted?

# #######################
# ### BULLA UTILITIES ###
# #######################


# def get_filters_bulla_file(filename: str, drop_times: bool = False) -> list[str]:
#     """
#     Fetch the filters that are in a Bulla model output file (type .dat)

#     Args:
#         filename (str): Filename from which the filters should be fetched
#         drop_times (bool, optional): Whether to drop the time array from the dat file or not. Defaults to False.

#     Returns:
#         list[str]: The filters that are in the file
#     """
    
#     assert filename.endswith(".dat"), "File should be of type .dat"
    
#     # Open up the file and read the first line to get the header
#     with open(filename, "r") as f:
#         names = list(filter(None, f.readline().rstrip().strip("#").split(" ")))
#     # Drop the times column if required, to get only the filters
#     if drop_times:
#         names = [name for name in names if name != "t[days]"]
#     # Replace  colons with underscores
#     names = [name.replace(":", "_") for name in names]
    
#     return names

# def get_times_bulla_file(filename: str) -> list[str]:
#     """
#     Fetch the times array of a Bulla model output file (type .dat)

#     Args:
#         filename (str): The filename from which the times should be fetched

#     Returns:
#         list[str]: The times array
#     """
    
#     assert filename.endswith(".dat"), "File should be of type .dat"
    
#     names = get_filters_bulla_file(filename, drop_times=False)
#     data = pd.read_csv(filename, 
#                        delimiter=" ", 
#                        comment="#", 
#                        header=None, 
#                        names=names, 
#                        index_col=False)
    
#     times = data["t[days]"].to_numpy()

#     return times

# def read_single_bulla_file(filename: str) -> dict:
#     """
#     Load lightcurves from Bulla type .dat files

#     Args:
#         filename (str): Name of the file

#     Returns:
#         dict: Dictionary containing the light curve data
#     """
    
#     # Extract the name of the file, without extensions or directories
#     name = filename.split("/")[-1].replace(".dat", "")
#     with open(filename, "r") as f:
#         names = get_filters_bulla_file(filename)
    
#     df = pd.read_csv(
#         filename,
#         delimiter=" ",
#         comment="#",
#         header=None,
#         names=names,
#         index_col=False,
#     )
#     df.rename(columns={"t[days]": "t"}, inplace=True)

#     lc_data = df.to_dict(orient="series")
#     lc_data = {
#         k.replace(":", "_"): v.to_numpy() for k, v in lc_data.items()
#     }
    
#     return lc_data

def interpolate_nans(data: dict[str, Float[Array, " n_files n_times"]],
                     times: Array, 
                     output_times: Array = None) -> dict[str, Float[Array, " n_files n_times"]]:
    """
    Interpolate NaNs and infs in the raw light curve data. 
    Roughyl inspired by NMMA code.

    Args:
        data (dict[str, Float[Array, 'n_files n_times']]): The raw light curve data
        diagnose (bool): If True, print out the number of NaNs and infs in the data etc to inform about quality of the grid.

    Returns:
        dict[str, Float[Array, 'n_files n_times']]: Raw light curve data but with NaNs and infs interpolated
    """
    
    if output_times is None:
        output_times = times
    
    # TODO: improve this function overall!
    copy_data = copy.deepcopy(data)
    output = {}
    
    for filt, lc_array in copy_data.items():
        
        n_files = np.shape(lc_array)[0]
        
        if filt == "t":
            continue
        
        for i in range(n_files):
            lc = lc_array[i]
            # Get NaN or inf indices
            nan_idx = np.isnan(lc)
            inf_idx = np.isinf(lc)
            bad_idx = nan_idx | inf_idx
            good_idx = ~bad_idx
            
            # Interpolate through good values on given time grid
            if len(good_idx) > 1:
                # Make interpolation routine at the good idx
                good_times = times[good_idx]
                good_mags = lc[good_idx]
                interpolator = interp.interp1d(good_times, good_mags, fill_value="extrapolate")
                # Apply it to all times to interpolate
                mag_interp = interpolator(output_times)
                
            else:
                raise ValueError("No good values to interpolate from")
            
            if filt in output:
                output[filt] = np.vstack((output[filt], mag_interp))
            else:
                output[filt] = np.array(mag_interp)

    return output

def load_event_data(filename: str):
    """
    Takes a file and outputs a magnitude dict with filters as keys.
    
    Args:
        filename (str): path to file to be read in
    
    Returns:
        data (dict[str, Array]): Data dictionary with filters as keys. The array has the structure [[mjd, mag, err]].

    """
    mjd, filters, mags, mag_errors = [], [], [], []
    with open(filename, "r") as input:

        for line in input:
            line = line.rstrip("\n")
            t, filter, mag, mag_err = line.split(" ")

            mjd.append(Time(t, format="isot").mjd) # convert to mjd
            filters.append(filter)
            mags.append(float(mag))
            mag_errors.append(float(mag_err))
    
    mjd = np.array(mjd)
    filters = np.array(filters)
    mags = np.array(mags)
    mag_errors = np.array(mag_errors)
    data = {}

    unique_filters = np.unique(filters)
    for filt in unique_filters:
        filt_inds = np.where(filters==filt)[0]
        data[filt] = np.array([ mjd[filt_inds], mags[filt_inds], mag_errors[filt_inds] ]).T

    return data

def write_event_data(filename: str, data: dict):
    """
    Takes a magnitude dict and writes it to filename. 
    The magnitude dict should have filters as keys, the arrays should have the structure [[mjd, mag, err]].
    
    Args:
        filename (str): path to file to be written
        data (dict[str, Array]): Data dictionary with filters as keys. The array has the structure [[mjd, mag, err]].
    """
    with open(filename, "w") as out:
        for filt in data.keys():
            for data_point in data[filt]:
                time = Time(data_point[0], format = "mjd")
                filt_name = filt.replace("_", ":")
                line = f"{time.isot} {filt_name} {data_point[1]:f} {data_point[2]:f}"
                out.write(line +"\n")
