# Author Information
# Name: Eugene Su
# Email: su.eugene@gmail.com
# GitHub: https://github.com/EugenePig
# Date: 2023/04/15

import argparse
import os
import logging
from os.path import join, basename, splitext, abspath

import h5py
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from multiprocessing import shared_memory
from joblib import Parallel, delayed
import seaborn as sns
from matplotlib import pyplot as plt

# Define constants and configurations
angle = 0  # Incident angle in degrees
polarization = 90  # Polarization angle in degrees
sobol_sampling = f'20230409_sobol_angle_{angle}_SN.csv'  # Sobol sampling CSV file
problem = {
    'num_vars': 6,
    'names': ['r', 'TCD', 'BCD', 'n', 'Depth', 'b'],
    'bounds': [[0.02, 0.1], [0.54, 0.66], [0.4, 0.54], [100, 600], [2.7, 3.3], [0.05, 0.1]]
}
HDF5_EXT_RAW = 'hdf5_EH_abs'  # File extension for HDF5 raw data

def save_results(data, filename, title, folder):
    """
    Save data as a CSV file and visualize it as a heatmap.

    Parameters:
        data (numpy.ndarray): 2D array to save and visualize.
        filename (str): Name of the output file (without extension).
        title (str): Title of the heatmap.
        folder (str): Output folder to save results.
    """
    # Save the data as a CSV file
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(join(folder, f'{filename}.csv'), index=False, header=False)

    # Create and save the heatmap visualization
    plt.figure(figsize=(7, 6.1))
    sns.heatmap(data, cmap="rainbow", annot=False)
    plt.title(title)
    plt.xlabel('Mesh X position')
    plt.ylabel('Mesh Y position')
    plt.savefig(join(folder, f'{filename}.png'))
    plt.close()

def load_hdf5_data(data_folder, sobol_sampling):
    """
    Load HDF5 data and corresponding Sobol sampling information.

    Parameters:
        data_folder (str): Path to the folder containing HDF5 files.
        sobol_sampling (str): Path to the Sobol sampling CSV file.

    Returns:
        tuple: (Y_data, wavelengths)
    """
    # Load the Sobol sampling data
    df_sampling = pd.read_csv(sobol_sampling)
    n_sample = len(df_sampling)
    Y_data = []
    wavelengths = None
    read_wavelengths_done = False

    # Process each sample
    for idx, row in df_sampling.iterrows():
        sn = int(row['SN'])
        file_path = join(data_folder, f'tsv2nd_{sn}_w30.{HDF5_EXT_RAW}')

        # Read HDF5 data for each sample
        with h5py.File(file_path, "r") as f:
            Y_data.append(f['E'][()])
            if not read_wavelengths_done:
                wavelengths = f['lambda_nm'][()]
                read_wavelengths_done = True

    assert len(Y_data) == n_sample, 'Mismatch between the number of samples and data files.'

    # Convert the loaded data to numpy arrays
    wavelengths = [str(round(float(col), 1)) for col in wavelengths]
    Y_data = np.array(Y_data).transpose(1, 2, 3, 0)  # Shape: (n_wavelength, n_mesh_x, n_mesh_y, n_samples)

    return Y_data, wavelengths

def create_shared_memory_array(data):
    """
    Create a shared memory array for the given numpy data.

    Parameters:
        data (numpy.ndarray): Input numpy array.

    Returns:
        tuple: SharedMemory object and numpy view of the shared memory.
    """
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    np.copyto(shared_array, data)
    return shm, shared_array

def compute_sobol_analysis(args, global_shm_name, shape):
    """
    Perform Sobol analysis for a specific (wavelength, mesh_x, mesh_y) combination.

    Parameters:
        args (tuple): (wavelength index, mesh_x index, mesh_y index).
        global_shm_name (str): Name of the shared memory.
        shape (tuple): Shape of the shared memory array.

    Returns:
        tuple: Sobol analysis results for the given combination.
    """
    w, m, n = args
    global_shm = shared_memory.SharedMemory(name=global_shm_name)
    global_Y_data = np.ndarray(shape, dtype=np.float32, buffer=global_shm.buf)

    Si = sobol.analyze(problem, global_Y_data[w, m, n], calc_second_order=False)
    return w, m, n, Si['S1'], Si['S1_conf'], Si['ST'], Si['ST_conf']

def perform_sensitivity_analysis_mp(n_wavelength, n_mesh_x, n_mesh_y, Y_data):
    """
    Perform Sobol sensitivity analysis using parallelization and shared memory.

    Parameters:
        n_wavelength (int): Number of wavelengths.
        n_mesh_x (int): Number of mesh points in x-direction.
        n_mesh_y (int): Number of mesh points in y-direction.
        Y_data (numpy.ndarray): Input data for sensitivity analysis.

    Returns:
        tuple: Sensitivity indices (S1, S1_conf, ST, ST_conf).
    """
    # Create shared memory for the input data
    shm, shared_Y_data = create_shared_memory_array(Y_data)

    # Initialize result arrays
    s1 = np.zeros((n_wavelength, n_mesh_x, n_mesh_y, problem['num_vars']), dtype=float)
    s1_conf = np.zeros_like(s1)
    st = np.zeros_like(s1)
    st_conf = np.zeros_like(s1)

    # Prepare tasks for parallel processing
    tasks = [(w, m, n) for w in range(n_wavelength) for m in range(n_mesh_x) for n in range(n_mesh_y)]
    logging.info('Starting parallel processing...')

    # Perform parallel processing with joblib
    results = Parallel(n_jobs=-1)(delayed(compute_sobol_analysis)(task, shm.name, Y_data.shape) for task in tasks)

    # Collect results
    for w, m, n, Si_S1, Si_S1_conf, Si_ST, Si_ST_conf in results:
        s1[w, m, n] = Si_S1
        s1_conf[w, m, n] = Si_S1_conf
        st[w, m, n] = Si_ST
        st_conf[w, m, n] = Si_ST_conf

    # Clean up shared memory
    shm.close()
    shm.unlink()

    return s1, s1_conf, st, st_conf

def main(data_folder):
    """
    Main function to execute the Sobol sensitivity analysis workflow.

    Parameters:
        data_folder (str): Path to the folder containing input HDF5 files.
    """
    results_folder = join(data_folder, 'Sobol_results')
    os.makedirs(results_folder, exist_ok=True)

    # Create subfolders for each variable
    for var_name in problem['names']:
        os.makedirs(join(results_folder, var_name), exist_ok=True)

    filename, _ = splitext(basename(sobol_sampling))
    Y_data, wavelengths = load_hdf5_data(data_folder, sobol_sampling)
    n_wavelength, n_mesh_x, n_mesh_y, n_Y_data = Y_data.shape

    # Perform Sobol sensitivity analysis
    s1, s1_conf, st, st_conf = perform_sensitivity_analysis_mp(n_wavelength, n_mesh_x, n_mesh_y, Y_data)
    s1[s1 < 0.0] = 0.0  # Ensure no negative values in results

    # Save results for each wavelength and variable
    for w in range(n_wavelength):
        for var_idx, var_name in enumerate(problem['names']):
            folder = join(results_folder, var_name)
            save_results(s1[w, :, :, var_idx], f"{filename}_wavelength_{wavelengths[w]}_variable_{var_name}_S1", f"First-order Sobol index of {var_name}", folder)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Perform Sobol sensitivity analysis on near-field data.')
    parser.add_argument('-f', '--folder', required=True, help='Path to the folder containing HDF5 files with near-field data')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Input folder: {abspath(args.folder)}")

    # Execute the main function
    main(abspath(args.folder))
