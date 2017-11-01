import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
from scipy import signal


# hola, probando git banrch
# -----------------------------------------------------------------------------
# Analysis functions
# -----------------------------------------------------------------------------
def spectrum(da, config, index=0, channel=20):
    """
    Function plot a spectrum of the pandas.Panel() data.

     """
    # Look for the data in the matrix
    data = da[channel, :, index]

    plt.plot(config['wave'], data)
    plt.title('Spectrum measured by MUDIS', fontsize=14)
    plt.xlabel('Wavelength[nm]', fontsize=13)
    plt.ylabel('Radiance[counts]', fontsize=13)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.show()


def variation_radiance(da, config, channel=0, wave=400):
    """
    Test change of Radiance Percentage in function of the time
    """
    wave_indx = int((wave - 245 - config['wave_correction']) / 0.446)

    # Create a array to save the data
    diference_p = np.zeros(len(da.time.values))

    # Make a loop over the dataset for a channel and wavelength in function
    # of the time
    for i in range(len(da.time.values)):
        diference_p[i] = (((da[channel, wave_indx, i])
                           # dp.iloc[0].radiance[0][channel][wave_index])
                           # / dp.iloc[0].radiance[0][channel][wave_index])) * 100
                           ))

    time = pd.to_datetime(da.time.values, format='%Y-%m-%d %H:%M:%S')

    # Plot the results
    plt.plot(time, diference_p, '.-')
    plt.title('Radiance', fontsize=14)
    plt.xlabel('time', fontsize=13)
    plt.ylabel('Radiance[counts]', fontsize=13)
    plt.xticks(rotation=60, size=12)
    plt.yticks(size=12)
    plt.show()


def radiance_plot(dataset, config, wave=400, time_indx=0, levels=20, vmax='default', typ=''):
    """
    Plot the radiance distribution in a polar contour plot using data from
    hdf5 functions.

    :parameter

    dataset: Matrix containing radiance dataset. Its form is a 3D matrix
            containing as dimensions [radiance, wavelength, time]

    config: Dictionary containing information about wavecorrection, dead fibre,
            different paths.

    wave: wavelength to plot

    time_indx: index in the matrix time dimension.

    levels: parameter used to plot radiance with pyplot.matplotlib library.

    typ(='meas', 'sim'): parameter to specify if datasets are obtained from
                        measured or simulated data.
    """

    # Select data from DataFrame
    azimuths = config['skymap'][:, 0]  # +180  # azimuths
    zeniths = config['skymap'][:, 1]  # zeniths

    if typ == 'sim':
        # look for wavelength index in array
        waves_sim = dataset.attrs['simulated_Columns'].split('nm')[0].split('[')[1].split(']')[0].split(',')
        waves = np.asarray(list(map(int, waves_sim)))
        wave_indx = np.where(waves == wave)
        try:
            wave_indx = np.int(wave_indx[0][0])
        except:
            print("Wavelength is not in dataset")
        z = dataset.simulated[:, wave_indx, time_indx]

    elif typ == 'meas':
        wave_indx = int((wave - 250 - config['wave_correction']) / 0.446)
        z = dataset.measured[:, wave_indx, time_indx]
    else:
        print('Select a input data type(sim or meas)')


    # Add values in the origin to close the surface interpolation
    azimuths = np.append(azimuths, [270, 0, 0, 0, 0, 0, 0, 0])
    zeniths = np.append(zeniths, [0, 12, 24, 36, 48, 60, 72, 84])
    z = np.append(z, [z[0], z[3], z[9], z[19], z[33], z[51], z[73], z[99]])

    # Convert x to radians
    azimuths = np.radians(azimuths)
    zeniths = np.radians(zeniths)

    # Remove dead channels of the dataset
    azimuths = np.delete(azimuths, config['dead_fibre'])
    zeniths = np.delete(zeniths, config['dead_fibre'])
    z = np.delete(z, config['dead_fibre'])

    # Set up a regular grid of interpolation point
    thetai, ri = np.linspace(azimuths.min(), azimuths.max(),
                             num=len(azimuths)), \
                 np.linspace(zeniths.min(), zeniths.max(), num=len(zeniths))

    ri, thetai = np.meshgrid(ri, thetai, indexing='ij')

    #zi = scipy.interpolate.griddata((azimuths, zeniths), z, (thetai, ri),
    #                                method='linear')

    rbf = scipy.interpolate.Rbf(azimuths, zeniths, z, fucntion='gaussian',
                                epsilon=0.05)

    ZI = rbf(thetai, ri)

    if typ == 'sim':
        name = str(dataset.time[time_indx].values) # ''
    else:
        name = str(dataset.time[time_indx].values)

    # Create the directory to save the results
    os.makedirs(os.path.dirname(config['path_note'] + '/figures/'), exist_ok=True)
    if vmax == 'default':
        vmax = 4200
    else:
        vmax = vmax

    # Plot the dataset
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    cmap = 'Spectral_r'  # 'rainbow'
    a = plt.contourf(thetai, ri, ZI, levels, cmap=cmap, vmin=0, vmax=vmax)  #  , vmax=4932)
    plt.title('{} UTC {}nm'.format(name, wave))
    plt.axis([0, 4200, 0, 1.48])

    plt.scatter(azimuths, zeniths, cmap=cmap, s=1)
    ax.grid(False)
    ax.set_theta_zero_location("N")  # Set the direction of polar plot
    ax.set_theta_direction(1)  # Set the increase direction on azimuth angles
                                # (-1 to clockwise, 1 counterclockwise)
    cbar = plt.colorbar(a)
    cbar.set_label("counts", rotation=90)

    if typ == 'sim':
        plt.savefig('figures/skymap/simulated/skymap{}nm_{}UTC_sim.jpeg'.format(wave, name), dpi=300)
        plt.show();
    else:
        plt.savefig('figures/skymap/measured/skymap{}nm_{}UTC_meas.jpeg'.format(wave, name), dpi=300)


def dcmp_cc_plot(dataset, config, channel=0, show_plot='no'):
    """
    Function plots Directional Cloud Modification Factor (DCMF) in function of
    the Cloud Cover (cc)

    """
    # look for wavelength index in array
    waves_sim = dataset.attrs['simulated_Columns'].split('nm')[0].split('[')[1].split(']')[0].split(',')
    waves = np.asarray(list(map(int, waves_sim)))
    wave_s = waves[1:]

    for wave in wave_s:
        wave_indx = int((wave - 245 - config['wave_correction']) / 0.446)

        # measured radiance
        meas_rad = dataset.measured.values[channel, wave_indx, :]

        # cloud cover.
        cc = dataset.clouds.data[channel, 0, :]

        # Directional cloud modification factor (DCMF)
        # -simulated data arrange
        wave_indx_s = np.where(waves == wave)
        try:
            wave_indx = np.int(wave_indx_s[0][0])
        except:
            print("Wavelength is not in dataset")

        # simulated radiance
        sim_rad = dataset.simulated.values[channel, wave_indx, :]

        # DCMF calculation
        DCMF = meas_rad / (sim_rad)

        # fit a curve in data
        arr = np.column_stack((cc, DCMF))

        # argsort() list of sorted index
        arr_s = arr[arr[:, 0].argsort()]

        x = arr_s[:, 0]
        y = arr_s[:, 1]

        y_smooth = signal.savgol_filter(y, (int(len(x)/2)), 5)

        # plot the smoothed curve
        plt.plot(x, y_smooth, '-.', markersize=1.5)
        plt.title('Smoothed DCMF/cloud cover')
        plt.xlabel('cloud cover')
        plt.ylabel('DCMF')
        plt.legend(wave_s)

        if show_plot=='yes':
            plt.show()
        else:
            pass

# ------------basic quantities ---------------

def cloud_cover(dataset, channel=0):
    return dataset.clouds.data[channel, 0, :]


def meas_radiance(dataset, config, wave=400, channel=0):
    wave_indx = int((wave - 245 - config['wave_correction']) / 0.446)
    return dataset.measured.values[channel, wave_indx, :]


def brightness(dataset, channel=0):
    return dataset.clouds.data[channel, 1, :]


def var_dcmp_cc_plot(dataset, config, channel=0, show_plot='no'):
    """
    Testing,
    Function plots Directional Cloud Modification Factor (DCMF) in function of
    the Cloud Cover (cc)

    """
    # look for wavelength index in array
    waves_sim = dataset.attrs['simulated_Columns'].split('nm')[0].split('[')[1].split(']')[0].split(',')
    waves = np.asarray(list(map(int, waves_sim)))
    wave_s = waves[1:]

    for wave in wave_s:
        wave_indx = int((wave - 245 - config['wave_correction']) / 0.446)

        # measured radiance
        meas_rad = dataset.measured.values[channel, wave_indx, :]

        var_meas = dataset.measured.values[channel, wave_indx, : + 1] - dataset.measured.values[channel, wave_indx, :]

        # cloud cover.
        cc = dataset.clouds.data[channel, 0, : + 1] - dataset.clouds.data[channel, 0, :]

        # Directional cloud modification factor (DCMF)
        # -simulated data arrange
        wave_indx_s = np.where(waves == wave)
        try:
            wave_indx = np.int(wave_indx_s[0][0])
        except:
            print("Wavelength is not in dataset")

        # simulated radiance
        sim_rad = dataset.simulated.values[channel, wave_indx, : + 1] - dataset.simulated.values[channel, wave_indx, :]

        # DCMF calculation
        DCMF = meas_rad # / (sim_rad)

        break
        # fit a curve in data
        arr = np.column_stack((cc, DCMF))

        # argsort() list of sorted index
        arr_s = arr[arr[:, 0].argsort()]

        x = arr_s[:, 0]
        y = arr_s[:, 1]

        y_smooth = signal.savgol_filter(y, (int(len(x)/2)), 5)

        # plot the smoothed curve
        plt.plot(x, y_smooth, '-.', markersize=1.5)
        plt.title('Smoothed DCMF/cloud cover')
        plt.xlabel('cloud cover')
        plt.ylabel('DCMF')
        plt.legend(wave_s)

        if show_plot=='yes':
            plt.show()
        else:
            pass
    return var_meas, cc, DCMF
