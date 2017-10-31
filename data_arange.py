## Library for raw data exporting
# Created by Mario Tobar

import glob
import os
import pandas as pd
import sys

# path to PERSONAL modules
PERSONAL_LIBRARIES_PATH = "/Volumes/Mac_3/Promotion_IMUK/Project/Python_programs/Personal_libraries_python/"

if not PERSONAL_LIBRARIES_PATH in sys.path:
    sys.path.append(PERSONAL_LIBRARIES_PATH)

# Add MUDIS modules
from MUDIS_HDF5 import MUDIS_hdf5 as mudis


# -------------work directories------------

path_all_files = '/Volumes/Mac_3/201705_Radiance_Campaign_data/'
path_note = os.path.dirname(os.path.realpath('20170530_data_analysis_campaign_may_2017')) # Path to directory of notebook
config = {'channel_pixel_adj': 14, # pixels to change in the channel alignment direction
          'dead_fibre': [11, 21, 42, 48],
          'path_all_files': path_all_files, # main path of files
          'path_note': path_note # Used to save the results
         }

config['path_files'] = path_all_files + 'Campaign_201704_relative_radiance_hdf5/relative_radiance/'

# ----------- functions-------------

def ask():
    inp = input('Write the initial index ')
    print(meas_files[int(inp)])
    ans = input("Is it corrected ")
    return inp, ans


# ---- MAIN PROGRAMM---------------

print('Program initializated')

#---------- measured data export -------------

# Import the radiance data from sensor
alignment = pd.read_table(path_all_files + 'Campaign_201704_relative_radiance_hdf5/relative_radiance/calibration_files/Alignment_Lab_UV_20120822.dat', sep='\s+', names=['Channel Number', 'Azimuth', 'Zenith', 'pixel', 'pixel', 'pixel'], skiprows=1)
print("Initial alignment data skymap: \n", alignment.head())

# Import the radiance data from sensor
meas_files = sorted(glob.glob(path_all_files + 'Campaign201705/RELATIVE_RADIANCE/20170511/data_*.txt'))
print('Total files in the measured radiance directory: ' + str(len(meas_files)) + ' files')





# select index
# Ask for initial and final index of measured data for the exporting
inp_in, ans = ask()

while ans != 'y':
    inp_in, ans = ask()

inp_fin, ans = ask()
while ans != 'y':
    inp_fin, ans = ask()

print("initial index", inp_in, "\n final index", inp_fin, "\n Total files to export", int(inp_fin) - int(inp_in))

# Rearranging measured data in channel, wavelength,
#mudis.dat2hdf5_mudis(alignment,
                     # config,
                     # meas_files,
                     # date='20170411',
                     # path_save='',
                     # init_file=int(inp_in),
                     # fin_file=int(inp_fin),
                     # step=1,
                     # expo=200)