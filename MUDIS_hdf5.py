import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Pysolar import solar as ps
import os
import xarray as xr
import scipy.interpolate

# ---------- ARRANGE FUNCTIONS------------

def create_str_dir(config):
    """Create the structured file directory in the path defined in the parameter
    str_dir"""

    # radiance directory
    os.makedirs(os.path.dirname(config['str_dir'] + 'radiance/'), exist_ok=True)
    # allsky images directory
    os.makedirs(os.path.dirname(config['str_dir'] + 'allsky/'), exist_ok=True)
    # simulation directory
    os.makedirs(os.path.dirname(config['str_dir'] + 'simulation/'), exist_ok=True)


def configuration(config):
    """ Apply all configurations for the analysis"""
    create_str_dir(config)
    config['']



def dat2hdf5_mudis(alignment, config, files, date='', init_file=0,
                   fin_file=100, step=1, expo='100'):

    """Function to convert raw data from MUDIS .txt file to hdf5 file with
    attributes.
    Parameters
    ----------
    alignment:
    files:
    init_file:
    fin_file:
    expo:

     """

    # --------SKYMAP--------------
    # Create the directory to save the results
    os.makedirs(
        os.path.dirname(config['path_MUDIS_hdf'] + 'calibration_files/'), exist_ok=True)

    # Extract skymap from alignment file
    skymap = np.zeros([len(alignment), 2])

    for i in np.arange(len(skymap)):
        skymap[i] = alignment['Azimuth'][i], alignment['Zenith'][i]

    # Save Skymap information
    with h5py.File(config['path_MUDIS_hdf'] + 'calibration_files/skymap_radiance.h5', 'w') as sky:

        if not list(sky.items()):
            sky.create_dataset('/skymap', data=skymap)
        else:
            del sky['skymap']

            sky.create_dataset('/skymap', data=skymap)
            sky['skymap'].attrs['Columns'] = 'Azimuth, Zenith'

    # Save MUDIS file information
    for fil in np.arange(init_file, fin_file, step):
        # Import the data from the file
        file = np.genfromtxt(files[fil], delimiter='', skip_header=11)

        # ------------RADIANCE DATA RAW---------------
        # create the radiance matrix
        data = np.zeros([113, 992])

        for i in np.arange(113):
            if str(alignment.iloc[i][3]) == 'nan':
                data[i] = np.nan
            else:
                data[i] = file[:, int(
                    alignment.iloc[i][3] + config['channel_pixel_adj'])]  #
                # read the pixels index
                # in the alignment file and copy the
                # data in the radiance matrix']))

        # Correct time for the file UTC
        name = os.path.split(files[fil])

        # Read name of the file (correct time)
        time = name[1][6:25]
        # convert time to datetime format
        time = datetime.datetime.strptime(time, '%d.%m.%Y_%H_%M_%S')
        # print(time)
        new_name = datetime.datetime.strftime(time, fmt='%Y%m%d_%H%M%S')

        with open(files[fil], 'r') as file:
            dat = file.readlines()

        # Extract information from .dat file
        exposure = int(dat[4][12:-1])
        NumAve = int(dat[7][17:-1])
        CCDTemp = int(dat[8][15:-1])
        NumSingMes = int(dat[10][27:-1])
        ElectrTemp = int(dat[9][23:-1])

        # Create the directory to save the results
        os.makedirs(os.path.dirname(config['path_MUDIS_hdf'] + '{}/data/').format(date),
                    exist_ok=True)

        if exposure == expo:
            # Create a file in the disk
            datos = h5py.File(config['path_MUDIS_hdf'] + date + '/data/' + new_name + '.h5',
                              'w')

            if not list(datos.items()):
                # Create two datasets(use only one time)
                datos.create_dataset('/data', data=data)
                datos.create_dataset('/skymap', data=skymap)
            else:
                del datos['data']
                del datos['skymap']
                print('data deleted and corrected')
                datos.create_dataset('/data', data=data)
                datos.create_dataset('/skymap', data=skymap)

            # Add attributes to datasets
            datos['data'].attrs['time'] = str(time)
            datos['data'].attrs['Exposure'] = exposure
            datos['data'].attrs['NumAver'] = NumAve
            datos['data'].attrs['CCDTemp'] = CCDTemp
            datos['data'].attrs['NumSingMes'] = NumSingMes
            datos['data'].attrs['ElectrTemp'] = ElectrTemp
            datos['data'].attrs['Latitude'] = '52.39N'
            datos['data'].attrs['Longitude'] = '9.7E'
            datos['data'].attrs['Altitude'] = '65 AMSL'

            datos['skymap'].attrs['Columns'] = 'Azimuth, Zenith'

            datos.close()

        else:
            print('Exposure are not same', expo, exposure)
            break
        print('File ' + str(fil - init_file + 1) + ' of ' +
              str((fin_file - init_file)) + ' saved')
    print('Completed')


def loadhdf5file(file_h5, key='data'):
    """Read contains of HDF5 file saved with dat2hdf5_mudis function"""

    with h5py.File(file_h5, 'r') as data:
        # Add datasets to dictionary
        info_value = {}
        info_attrs = {}

        for i in np.arange(len(data.items())):
            info_value.update({str(list(data.items())[i][0]): data[str(list(data.items())[i][0])].value})

        for i in np.arange(len(data[key].attrs)):
            info_attrs.update({list(data[key].attrs.keys())[i]: list(data[key].attrs.values())[i]})

    return info_value, info_attrs


class DataArray:
    """ Create a Dataset with the files selected in the parameters"""

    def __init__(self, files_h5, config, init_file, fin_file, step):
        self.files_h5 = files_h5
        self.config = config
        self.init_file = init_file
        self.fin_file = fin_file
        self.step = step

    def measured(self):
        key = 'data'
        dataray = np.zeros([113, 992, int(round((self.fin_file - self.init_file) / self.step, 0))])
        time_meas = []

        for j in np.arange(self.init_file, self.fin_file, self.step):
            # load a file
            info = loadhdf5file(self.files_h5[j], key=key)

            dataray[:, :, int((j - self.init_file) / self.step)] = (info[0][key] - self.config['dark_current'])
            time_convert = datetime.datetime.strptime(info[1]['time'], '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d_%H%M%S')
            time_meas.append(time_convert)

        channels = np.arange(113)
        wavelengths = self.config['wave']

        # delete time of global attributes
        #del info[1]['time']

        # Create a file in the disk and add attributes to dataArray
        da = xr.DataArray(dataray, coords=[channels, wavelengths, time_meas],
                          dims=['channel', 'wavelength', 'time'],
                          name='radiance',
                          attrs=info[1])
        return da

    def simulated(self):
        key = 'simulated'
        dataray = np.zeros(
            [113, 3, int(round((self.fin_file - self.init_file) / self.step, 0))])
        time_meas = []

        for i in np.arange(self.init_file, self.fin_file, self.step):
            info = loadhdf5file(self.files_h5[i], key=key)

            dataray[:, :, int((i - self.init_file) / self.step)] = info[0][key]
            time_meas.append(info[1]['Time'])

        channels = np.arange(113)
        columns = info[1]['Columns'].split(sep=',')

        # delete time of global attributes
        #del info[1]['Time']

        # Create a file in the disk and add attributes to dataArray
        da = xr.DataArray(dataray, coords=[channels, columns, time_meas],
                          dims=['channel', 'columns', 'time'],
                          name='Clear sky simulation',
                          attrs=info[1])
        return da

    def clouds(self):
        """
        Read hdf5 files and create a Dataset in xarray.
        :return:
        """
        key = '/'
        dataray = np.zeros([113, 2, self.config['temporal_range_size']])

        time_meas = self.config['time_measured']

        j = 0  # index measured data
        k = 0  # index cloud data
        while j < self.config['temporal_range_size']:
            values, attrb = loadhdf5file(self.files_h5[k], key=key)
            time = datetime.datetime.strptime(attrb['Time'],
                                              '%Y-%m-%d %H:%M:%S') #%Y%m%d_%H%M%S')  # string
            time_m = datetime.datetime.strptime(time_meas[j],
                                                '%Y%m%d_%H%M%S')  # string
            delta_time = datetime.timedelta(seconds=10)

            if j == 0:
                if time_m + delta_time < time:
                    print(
                        'Time difference between initial measurement and cloud properties is too big')
                    break
                else:
                    pass

            if time_m <= time:
                dataray[:, 0, j] = values['cloud_cover'][:, 1]
                dataray[:, 1, j] = values['brightness'][:, 1]
                j += 1
            else:
                k += 1

        channels = np.arange(113)
        columns = ['cloud_cover', 'brightness']
        attrb['Columns'] = columns
        print(len(dataray), len(time_meas))
        # delete non global attributes
        del attrb['Brightness_mean']
        del attrb['Cloud_cover_mean']
        del attrb['Time']
        del attrb['UTC']

        # Create a file in the disk and add attributes to dataArray
        da = xr.DataArray(dataray, coords=[channels, columns, time_meas],
                          dims=['channel', 'columns', 'time'],
                          name='Cloud conditions',
                          attrs=attrb)
        return da


def dark_correction(dark_file, alignment, config):
    """
    Calculate the dark current in the measurements
    """

    # -----------DARK CURRENT----------------------
    # Import the data from the file
    dark = np.genfromtxt(dark_file[0], delimiter='', skip_header=11)

    # Create array to save data
    dark = np.zeros(list(dark.shape))

    print('Calculating...')

    # Calculate mean of dark files
    for i in np.arange(len(dark_file)):
        dark += np.genfromtxt(dark_file[i], delimiter='', skip_header=11)

    dark = dark / (i + 1)

    # create the radiance matrix
    dark_current = np.zeros([113, 992])

    for i in np.arange(113):
        if str(alignment.iloc[i][3]) == 'nan':
            dark_current[i] = np.nan
        else:
            dark_current[i] = dark[:, int(alignment.iloc[i][3]) +
                                      config['channel_pixel_adj']]
    print('Complete')

    return dark_current


def wave_correction(wave_files, alignment, config, correction, dir_ind=5):
    """
    Function applies a correction in the wavelegth values of the CCD pixels

    Parameters
    ----------
    wave_file:

    correction:

    dir_ind:(optional) index of channel use to plot the lines and spectrum

    Return
    ------

    wave:

    """
    # -----------WAVE Hg LINES----------------------
    # Import the data from the file
    wave = np.genfromtxt(wave_files[0], delimiter='', skip_header=11)

    # Create array to save data
    waves = np.zeros(list(wave.shape))

    print('Calculating...')

    # Calculate mean of dark files
    for i in np.arange(len(wave_files)):
        waves += np.genfromtxt(wave_files[i], delimiter='', skip_header=11)

    # Mean wave values
    waves_m = waves / (i + 1)

    # create the radiance matrix
    wave_data = np.zeros([113, 992])

    for i in np.arange(113):
        if str(alignment.iloc[i][3]) == 'nan':
            wave_data[i] = np.nan
        else:
            wave_data[i] = waves_m[:, int(alignment.iloc[i][3] +
                                          config['channel_pixel_adj'])]

    # Define Hg-Lamp emission lines
    HgAr_lines = [253.7, 296.728, 302.2, 312.952, 334.148, 365.338, 404.657,
                  435.834, 546.075, 576.96]

    # Create the wavelength array for MUDIS
    wave = np.zeros(992)

    # ----MANUAL DISPLACEMENT------
    for i in np.arange(len(wave)):
        wave[i] = 250 + i * 0.446 + correction

    # Import wave file and plot the data
    # wave_data = np.genfromtxt(wave_files, delimiter='', skip_header=11)
    plt.plot(wave, wave_data[dir_ind, :], 'b-')

    # Plot Hg Lines
    for ind in np.arange(len(HgAr_lines)):
        plt.plot([HgAr_lines[ind], HgAr_lines[ind]], [0, 4000], 'r-')

    plt.title('Hg emission lines on CCD', fontsize=14)
    plt.xlabel('Wavelength[nm]', fontsize=13)
    plt.ylabel('Counts', fontsize=13)
    plt.yscale('log')
    plt.show()

    return wave


def simulate_UVSPEC(file, config):
    """ Calculate and simulate the radiance without clouds for a date and time
    the position of the sun in calculated for the local time, that is why, the
    function use the UTC to correct the simulation to UTC time"""

    wavelength = config['wavelength']

    # Coordenates from position of the station Hannover
    latitude = 52.39  # positive in the northern hemisphere
    longitud = 9.7  # negative reckoning west from prime meridian in Greenwich,

    # Read name of the file (correct time)
    name = os.path.split(file)
    time_n = name[1][0:15]
    print("Time name", time_n)
    # convert time to datetime format
    time = datetime.datetime.strptime(time_n,
                                      '%Y%m%d_%H%M%S')
    # Calculate the azimuth and zenith angles in function of the date
    elev = ps.GetAltitude(latitude, longitud, time)
    azi = ps.GetAzimuth(latitude, longitud, time)
    zenith = 90 - elev

    # Correction between the sign and real azimuth for plot of radiance
    if -180 <= azi < 0:
        azi = 180 - azi
    elif -360 <= azi < -180:
        azi = -azi - 180
    else:
        pass

    print("Azimuth: {:5.1f}".format(azi),
              "\nZenith: {:5.1f}".format(zenith))

    # Change the value of zenith and azimuth angles in function of the time and
    # position in the UVSPEC file

    with open(config['personal_libraries_path'] + 'MUDIS_HDF5/MUDIS_radiance_Input.txt', 'r') as file:
        data = file.readlines()

    data[14] = "day_of_year " + str(time.timetuple().tm_yday) + "  " + "\n"
    data[15] = "wavelength  " + str("{}".format(wavelength)) + "   " + \
               str("{}".format(wavelength)) + \
               "  #  wavelength to calcule [nm] \n"
    data[17] = "sza  " + str("{:2.3f}".format(zenith)) + \
               "       # Solar zenith angle \n"
    data[18] = "phi0  " + str("{:2.3f}".format(azi)) + \
               "      #Azimuth angle with zenith position \n"

    with open(config['personal_libraries_path'] + 'MUDIS_HDF5/MUDIS_radiance_Input.txt', 'w') as file:
        file.writelines(data)

    # Create the directory to save the results
    os.makedirs(os.path.dirname(config['simulations'] + '{}/{}nm/'.format(time_n[0:8],
                                                              wavelength)),
                exist_ok=True)

    # Run the program UVSPEC in the terminal
    os.system(config['UVSPEC_path'] + 'uvspec < ' + config['personal_libraries_path'] +
              'MUDIS_HDF5/MUDIS_radiance_Input.txt>   ' + config['simulations'] + '{}/{}nm/'.format(time_n[0:8], wavelength) + time_n +
              '.txt')


def export_sim_rad(file, config):
    """
    Export the data simulated by UVSPEC from MUDIS configuration to a compa-
    tible format for the analysis. It deletes the information of dead fiber
    direction. Function used by "mudis.sim_radiance()

    IMPORTANT

    Define wavelength value and save in config['wavelength] dictionary

    "

    Last modification(2017.05.24) """

    # os.makedirs(os.path.dirname(path + 'Formatted_data/Simulated/%snm/')
    #             % wavelength, exist_ok=True)
    data = np.genfromtxt(file, delimiter='')
    # source dir is the directory
    source_dir = config['personal_libraries_path'] + "MUDIS_HDF5/MUDIS_config/"

    # Import angle files to arrange simulated data from UVSPEC
    phi = np.genfromtxt(source_dir + "MUDIS_phi_angles.txt")

    # Create a matrix with angles and radiance arranged.
    radiance = np.zeros([584, 3])
    radiance[0] = 0, 0, data[1]

    for i in range(1, 74):
        radiance[i - 1] = 0.0, phi[i - 1], data[i]
        radiance[73 * 1 + i - 1] = 12.0, phi[i - 1], data[73 * 1 + i]
        radiance[73 * 2 + i - 1] = 24.0, phi[i - 1], data[73 * 2 + i]
        radiance[73 * 3 + i - 1] = 36.0, phi[i - 1], data[73 * 3 + i]
        radiance[73 * 4 + i - 1] = 48.0, phi[i - 1], data[73 * 4 + i]
        radiance[73 * 5 + i - 1] = 60.0, phi[i - 1], data[73 * 5 + i]
        radiance[73 * 6 + i - 1] = 72.0, phi[i - 1], data[73 * 6 + i]
        radiance[73 * 7 + i - 1] = 84.0, phi[i - 1], data[73 * 7 + i]

    # Export the same table of data that MUDIS (113 directions) for comparison
    angles_mudis = config['skymap']

    index_ang = np.zeros([113, 2])
    # Look for the index in the simulated table.
    for i in range(1, 584):
        for j in range(113):
            if radiance[i, 1] == angles_mudis[j, 0] and radiance[i, 0] == \
                    angles_mudis[j, 1]:
                index_ang[j] = i, j
            else:
                pass

    sim_data = np.zeros([113, 3])
    # Make the table with the same order that MUDIS skymap
    for i in range(113):
        sim_data[i] = (radiance[int(index_ang[i, 0]), 1], radiance[int(
            index_ang[i, 0]), 0], radiance[int(index_ang[i, 0]), 2])

    # Read name of the file (correct time)
    name = os.path.split(file)
    time_n = name[1][0:15]
    date = name[1][0:8]

    # Create the directory to save the results
    os.makedirs(os.path.dirname(
        config['dir_sim'] + '{}/{}nm/hdf5_files/'.format(date, config['wavelength'])),
                exist_ok=True)

    # Save simulated data information
    simulated = h5py.File(config['dir_sim'] +
                          '{}/{}nm/hdf5_files/{}_simulated_radiance.h5'.format(date, config['wavelength'], time_n),
                          'w')

    if not list(simulated.items()):
        simulated.create_dataset('/simulated', data=sim_data)
    else:
        del simulated['simulated']

        simulated.create_dataset('/simulated', data=sim_data)

    simulated['/simulated'].attrs['Columns'] = 'Azimuth, Zenith, Radiance'
    simulated['/simulated'].attrs['Time'] = time_n

    simulated.close()

    #print('Data of simulation were saved')
    return sim_data



def add2dataset():
    wave
    channel = measured_rad.channel
    time = measured_rad.time.values
    attrib = {}

    for key in measured_rad.attrs:
        attrib['measured_' + key] = measured_rad.attrs[key]
    for key in simul_rad.attrs:
        attrib['simulated_' + key] = simul_rad.attrs[key]
    for key in cloud_info.attrs:
        attrib['cloud_' + key] = cloud_info.attrs[key]

    dS = xr.Dataset({'measured': (['channel', 'wavelength', 'time'], measured_rad),
                     'simulated': (['channel', 'columns', 'time'], simul_rad),
                     'clouds': (['channel', 'columns2', 'time'], cloud_info)},
                    coords={'time': time,
                            'wavelength': wave,
                            'channel': channel
                            },
                    attrs=attrib
                    )
