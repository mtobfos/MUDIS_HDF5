import datetime
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numba import jit
import numpy as np
import os
from Pysolar import solar as ps
import scipy.misc


###############################################################################

# """ ALL-SKY IMAGE DATA ANALYSIS FUNCTIONS
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""Functions for the analysis of image obtained of a G10 ALL-SKY camera

Modificar !!!!!!!!!!!!!!!!!!!!!!!"""


@jit
def crop_image(image, aoi=1250, cx=1728, cy=2592, radio_image=1400):
    """Crop the image to a effective work area"""

    x_ran, x_ran1 = cx - radio_image, cx + radio_image
    y_ran, y_ran1 = cy - radio_image, cy + radio_image

    image_cropped = np.ones(
        [2 * radio_image, 2 * radio_image, 3], dtype=np.uint8)

    for i in range(x_ran, x_ran1):
        for j in range(y_ran, y_ran1):
            image_cropped[i - x_ran, j - y_ran] = image[i, j, 0], image[
                i, j, 1], image[i, j, 2]

            if aoi ** 2 <= ((i - radio_image - x_ran) ** 2 +
                                    (j - radio_image - y_ran) ** 2):
                image_cropped[i - x_ran, j - y_ran] = 0,0,0 #np.nan, np.nan, np.nan
            else:
                pass

    return image_cropped


@jit
def cloud_sky(image, radio_image=1000):
    """ Filter to obtain only the clouds from an all-sky camera photos and
    calculate the cloud cover.
    image must be the cropped image obtained with the function mudis.crop_image
     """

    clouds = np.zeros([len(image), len(image), 3], dtype=np.uint8)
    np.seterr(divide='ignore')
    # with np.errstate(divide='ignore'):
    ratio_rb = image[:, :, 0] / (image[:, :, 2])
    ratio_gb = image[:, :, 1] / (image[:, :, 2])

    ratio_rb[np.isnan(ratio_rb)] = 0
    ratio_gb[np.isnan(ratio_gb)] = 0

    # diference = abs(ratio_rb - ratio_gb)

    for i in range(len(image)):
        for j in range(len(image)):

            # Sun position
            if image[i, j, 0] >= 255 and image[
                i, j, 1] >= 255 and image[
                    i, j, 2] >= 255:

                clouds[i, j] = [0, 0, 0]

            # Sky
            # red/blue
            elif 0.5 <= ratio_rb[i, j] <= 0.7 or 0.5 <= ratio_gb[i, j] <= 0.80: # 85
                clouds[i, j] = [1, 1, 1]

            # new filter
            #elif diference[i, j] > 0.09:
            #    clouds[i, j] = [1, 1, 1]

            # external area of the image
            elif (i - radio_image) ** 2 + (j - radio_image) ** 2 >= radio_image ** 2:
                clouds[i, j] = [0, 0, 0]
            # Clouds
            else:
                clouds[i, j] = [image[i, j, 0],
                                 image[i, j, 1],
                                 image[i, j, 2]]

    return clouds


##############################################################################


def image_date(file):
    """ Function that return the date and UTC information of the all-sky image
    """
    date = os.path.split(file)
    # Variable to save the file name
    date2 = date[1]

    # Variable for the correct time
    utc = int(date2[20:22])
    date3 = (int('20' + date2[2:4]), int(date2[4:6]), int(date2[6:8]),
             int(date2[9:11]), int(date2[11:13]), int(date2[13:15]))

    return date3, utc


@jit
def circle_sun(image, config,
               date=("year", "month", "day", "hrs", "min", "sec"),
               utc='', zen='', cosk=''):
    """ Place a circle over the sun position.
    sun position is determined by using of the function datetime.datetime() and
    mudis.hemispherical_circle_3D_to_2D()

     IMAGE: Image cropped using the function mudis.crop_image()
     DATE: parameter obtained from the name of the image. We need the local
           time, therefore we use a UTC of +2 for Hannover.
           :type utc: object

     """

    # Coordinates of the position of station Hannover.
    latitude = 52.39  # positive in the northern hemisphere
    longitude = 9.7  # negative reckoning west from prime meridian in
    # Greenwich,

    # Calculate the azimuth and zenith angles in function of the date
    d = datetime.datetime(date[0], date[1], date[2], date[3] + utc, date[4],
                          date[5])
    elev = ps.GetAltitude(latitude, longitude, d)
    azi = ps.GetAzimuth(latitude, longitude, d)
    zenith = 90 - elev

    # Correction between the sign and real azimuth for plot of radiance
    if -180 <= azi < 0:
        azi = 180 - azi
    elif -360 <= azi < -180:
        azi = -azi - 180
    else:
        pass

    print("Azimuth: {:5.1f}".format(azi), "\nZenith: {:5.1f}".format(zenith))

    # calculate position of zenith and azimuth over the image
    image_filtered = hemispherical_circle_3D_to_2D(fov=14,
                                      zen_direction=zenith,
                                      azim_direction=azi + float(config['nord_desv']),
                                      zen=zen,
                                      cosk=cosk,
                                      image=image)
    return image_filtered



##############################################################################
# HEMISPHERICAL FOV
@jit
def zen(image):
    RADIO_IMAGE = int(len(image)/2)
    ZEN = np.zeros([len(image), len(image)])

    azimuth = np.linspace(0, 360, num=10000)
    for zenith in np.arange(0, 90, 0.01):
        for azim in azimuth:
            RADIO_F = RADIO_IMAGE * zenith / 90

            x = int(RADIO_F * np.cos(np.radians(azim))) - RADIO_IMAGE
            y = int(RADIO_F * np.sin(np.radians(azim))) - RADIO_IMAGE
            ZEN[x, y] = zenith
    return ZEN


##############################################################################
@jit
def cosk(image):
    """Return the azimuth angles on the surface of a ALL-SKY image"""

    RADIO_IMAGE = int(len(image)/2)
    COSK = np.zeros([len(image), len(image)])
    RADIO_FOV = np.linspace(0, RADIO_IMAGE, num=6000)
    azimuth = np.linspace(0, 360, num=10000)

    for RADIO_F in RADIO_FOV:
        for azim in azimuth:
            x = int(RADIO_F * np.cos(np.radians(azim - 180))) - RADIO_IMAGE
            y = int(RADIO_F * np.sin(np.radians(azim - 180))) - RADIO_IMAGE
            COSK[x, y] = azim
    return COSK


##############################################################################
# HEMISPHERICAL_CIRCLE_3D_TO_2D from Michael Routines

@jit
def hemispherical_circle_3D_to_2D(fov=9, zen_direction=0, azim_direction=0,
                                  zen=zen, cosk=cosk, image=None,
                                  radio_image=''):
    """ Determine the FOV projection over an all-sky image.
    To speed up the calculus import the zen and cosk variable on the notebook

    last modified 2016.12.27 by Mario Tobar """

    if image is not None:
        RADIO_IMAGE = int(len(image) / 2)
    else:
        RADIO_IMAGE = int(radio_image)

    FOV = fov

    delta_x = int((RADIO_IMAGE * (np.pi / 180) * FOV) * 30)

    geo_b = 90 - zen
    geo_l = cosk

    zenith = zen_direction
    azimu = azim_direction

    if image is None:
        circle_array = np.zeros([2 * int(radio_image), 2 * int(radio_image)])
        print("no image used for calculation")
    else:
        circle_array = image
        print("image used for calculation")

    x_pos = RADIO_IMAGE - RADIO_IMAGE * np.cos(np.radians(azimu)) * zenith / 90

    y_pos = RADIO_IMAGE - RADIO_IMAGE * np.sin(
        np.radians(azimu)) * zenith / 90  # azim-180

    sou_b = 90 - zenith
    sou_l = azimu

    if x_pos - delta_x < 0:
        ax = int(0)
    else:
        ax = int(x_pos - delta_x)

    if x_pos + delta_x > 2 * RADIO_IMAGE:
        ex = int(2 * RADIO_IMAGE)
    else:
        ex = int(x_pos + delta_x)

    if y_pos - delta_x < 0:
        ay = int(0)
    else:
        ay = int(y_pos - delta_x)

    if y_pos + delta_x > 2 * RADIO_IMAGE:
        ey = int(2 * RADIO_IMAGE)
    else:
        ey = int(y_pos + delta_x)

    distance = np.zeros([len(circle_array), len(circle_array)])

    distance[ax:ex, ay:ey] = np.arccos((np.sin(np.radians(
                                        geo_b[ax:ex, ay:ey])) *
                                        np.sin(np.radians(sou_b))) + \
                                       (np.cos(
                                        np.radians(geo_b[ax:ex, ay:ey])) *
                                        np.cos(np.radians(sou_b)) *
                                        np.cos(np.radians(
                                        sou_l - geo_l[ax:ex, ay:ey])))) * \
                             180 / np.pi

    ab_klgl_FOV = np.transpose(np.where(distance <= (FOV / 2)))

    circle_array[ab_klgl_FOV[:, 0], ab_klgl_FOV[:, 1]] = 1.1

    # delete values greater than RADIO_IMAGE
    cx = int(circle_array.shape[0] / 2)
    cy = int(circle_array.shape[1] / 2)
    x_ran, x_ran1 = cx - RADIO_IMAGE, cx + RADIO_IMAGE
    y_ran, y_ran1 = cy - RADIO_IMAGE, cy + RADIO_IMAGE

    for i in range(x_ran, x_ran1):
        for j in range(y_ran, y_ran1):
            circle_array[i - x_ran, j - y_ran] = circle_array[i, j]

            if RADIO_IMAGE ** 2 <= ((i - RADIO_IMAGE - x_ran) ** 2 +
                                            (j - RADIO_IMAGE - y_ran) ** 2):
                circle_array[i - x_ran, j - y_ran] = 0
            else:
                pass

    return circle_array

##############################################################################

@jit
def skyindex(file, config):
    """ Calculate the skyindex of an allsky image"""

    img = plt.imread(file)
    img1 = crop_image(img, aoi=1000, cx=1676, cy=2178,
                      radio_image=1082)  # for g10 camera

    img2 = np.asfarray(img1)

    red = img2[:, :, 0]
    blue = img2[:, :, 2]

    sky_index = np.zeros([len(img2), len(img2)])

    sky = (blue - red) / (blue + red)

    for i in np.arange(len(img2)):
        for j in np.arange(len(img2)):
            if -1 < sky[i, j] < 0:
                sky_index[i, j] = -0.5
            elif 0 <= sky[i, j] < 0.1:
                sky_index[i, j] = 0.05
            elif 0.1 <= sky[i, j] < 0.14:
                sky_index[i, j] = 0.12
            elif 0.14 <= sky[i, j] < 0.18:
                sky_index[i, j] = 0.16
            elif 0.18 <= sky[i, j] < 0.2:
                sky_index[i, j] = 0.19
            elif 0.2 <= sky[i, j] < 0.25:
                sky_index[i, j] = 0.22
            elif 0.25 <= sky[i, j] < 0.4:
                sky_index[i, j] = 0.3
            elif 0.4 <= sky[i, j] < 0.5:
                sky_index[i, j] = 0.45
            elif 0.5 <= sky[i, j] <= 1:
                sky_index[i, j] = 0.9
            else:
                pass

            if 1000 ** 2 <= ((i - 1082) ** 2 + (j - 1082) ** 2):
                sky_index[i, j] = -2
            else:
                pass


    # make a color map of fixed colors
    cmap = colors.ListedColormap(
        ['black', 'red', 'orangered', 'orange', 'yellow', 'green', 'aqua',
         'blue', 'purple', 'navy'])
    bounds = [-2, -1, 0, 0.1, 0.14, 0.18, 0.2, 0.25, 0.4, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(sky_index, cmap=cmap, norm=norm) # , interpolation='bilinear'
    plt.title('Sky Index')
    plt.colorbar()

    return sky_index


@jit
def haze_index(file, zen, cosk, config,):
    """
    Identifies clouds and sky using the SkyIndex and HazeIndex and classifies
    in function of colors. Clouds are red and values between -1 and 0.14. Sky
    is blue with values between 0.2 and 1

    Parameters

    -----------

    file: Directory to image file.

    zen: Matrix with zenith angles projected on a horizontal plane. It is ob-
        tained with the function Allsky_images.zen().

    cosk: Matrix with azimuth angles projected on a horizontal plane. It is ob-
        tained with the function Allsky_images.cosk().


    config: Configuration dictionary

    Return

    ------

    filtered: Numpy array which contains values after classification
    """

    img = plt.imread(file)

    img1 = crop_image(img, aoi=1000, cx=1676, cy=2178,
                      radio_image=1082)  # for g10 camera

    img2 = np.asfarray(img1)  # Important

    red = img2[:, :, 0]
    green = img2[:, :, 1]
    blue = img2[:, :, 2]

    # sky index
    sky = (blue - red) / (blue + red)

    # pixels in middle range
    ind = np.where((0.14 <= sky) & (sky < 0.2))

    # haze index
    haze = ((((red + blue) / 2) - green) / (((red + blue) / 2) + green))

    # Put values of sky index in haze and evaluate middle range
    haze_color = sky
    haze_color[ind] = haze[ind]

    val_corr = np.where((0 <= haze_color[ind]), haze_color[ind] == 0.14,
                        haze_color[ind] == 0.2)

    haze_color[ind] = val_corr

    # make a color map of fixed colors
    cmap = colors.ListedColormap(['black', 'red', 'orangered', 'orange', 'yellow', 'green', 'aqua',
                                  'blue', 'navy', 'purple'])
    bounds = [-2, -1, 0, 0.1, 0.14, 0.18, 0.2, 0.25, 0.4, 0.5, 1]

    # a colormap and a normalization instance
    norm = colors.BoundaryNorm(bounds, cmap.N)

    image_c = cmap(norm(haze_color))  # Image to save with the correct colors RGBA

    image_sa = image_c[:, :, 0:3]  # Load only RGB layers into array

    # Calculate the date and utc information from the image
    date = image_date(file)[0]
    utc = image_date(file)[1]

    # Add circle in the sun position
    filtered = circle_sun(image_sa, config, date, utc=utc, zen=zen, cosk=cosk)

    for j in np.arange(len(img2)):
        for i in np.arange(len(img2)):
            if 1000 ** 2 <= ((i - 1082) ** 2 + (j - 1082) ** 2):
                filtered[i, j] = 0
            else:
                pass

    plt.imshow(filtered)
    plt.close()

    name = '{:4d}{:02d}{:02d}'.format(date[0], date[1], date[2]) # date
    full_name = '{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(date[0], date[1],
                                                              date[2], date[3],
                                                              date[4], date[5])
    # Create the directory to save the results
    os.makedirs(os.path.dirname(config['path_save_img'] + '{}/'.format(name)),
                exist_ok=True)

    scipy.misc.imsave(config['path_save_img'] + '{}/{}_UTC+0{}_filtered.png'.format(
          name, full_name, utc), filtered)

    return filtered

@jit
def image_filter(file, zen, cosk, config=''):
    """ Function used to filter and save a image

    Parameters:
        file:
        zen:
        cosk:
        config:

    Returns:
        """
    # Read a image from the directory index
    image = plt.imread(file)

    image2 = crop_image(image, aoi=1000, cx=1676, cy=2178,
                        radio_image=1082)  # for g10 camera

    image3 = cloud_sky(image2, radio_image=1082)

    # Calculate the date and utc information from the image
    date = image_date(file)[0]
    utc = image_date(file)[1]

    filtered = circle_sun(image3, config, date, utc=utc, zen=zen, cosk=cosk)
    plt.imshow(filtered)

    name = '{:4d}{:02d}{:02d}'.format(date[0], date[1], date[2])
    full_name = '{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(date[0], date[1],
                                                              date[2], date[3],
                                                              date[4], date[5])
    # Create the directory to save the results
    os.makedirs(os.path.dirname(config['path_save_img'] + '{}/'.format(name)),
                exist_ok=True)

    scipy.misc.imsave(config['path_save_img'] + '{}/{}_UTC+0{}_filtered.png'.format(
        name, full_name, utc),
        filtered)

    return filtered

###############################################################################
##############################################################################
# -------------CALCULATES AND EXPORT FOV PIXELS INDEX ----------------------------


@jit
def fov_calculus(fov=9, zen_direction=0, azim_direction=0,
                 zen='', cosk='', image=None, radio_image='', config=''):
    """ Calcule the FOV projection over an all-sky image. Then, it save only
    the array of the fov position.
    To speed up the process, import the zen and cosk variable on the notebook

    routines needed:
    -- mudis.hemishemispherical_circle_3D_to_2D()


    last modified 2016.12.02 by Mario Tobar"""

    fov_array = hemispherical_circle_3D_to_2D(fov=fov,
                                              zen_direction=zen_direction,
                                              azim_direction=azim_direction + float(config['nord_desv']),
                                              zen=zen, cosk=cosk,
                                              image=image, radio_image=radio_image)

    pixels_fov_index = np.transpose(np.where(fov_array >= 1))
    pixels_fov_index.astype(int)

    return pixels_fov_index


##############################################################################
@jit
def fov_save(config, zen=zen, cosk=cosk,
             radio_image=''):
    """ Save the FOV pixels position for each direction on the MUDIS optic input
    This function uses the mudis.fov_calculus() function

    To reduce size of the files, the format of the pixel position is fmt='%.4d'

    routines needed:
    -- mudis.fov_calculus()

     last modified 2016.12.02 by Mario Tobar """

    # Save simulated data information

    for i in np.arange(113):
        fov = fov_calculus(fov=14,
                           zen_direction=config['skymap'][i, 1],
                           azim_direction=config['skymap'][i, 0],
                           zen=zen, cosk=cosk,
                           radio_image=radio_image,
                           config=config)

        with h5py.File(config['path_save_img'] + 'fov_pixels.h5', 'a') as fov_pix:

            if '{:03d}_fov_pixels'.format(i) in list(fov_pix.keys()):
                data = fov_pix['/{:03d}_fov_pixels'.format(i)]
                data[...] = fov
                print('changed')
            else:
                fov_pix.create_dataset('/{:03d}_fov_pixels'.format(i), data=fov)

            fov_pix['/'].attrs['Information'] = 'Pixels over a image captured by G10' \
                                            'Allsky camera of each MUDIS ' \
                                            'channel'

    return print("Completed")

# ############# INFORMATION OF ALL-SKY PHOTOS ##############################

@jit
def cloud_quantity(img, config):
    """ In development:

    Function calculates the quantity of pixels where clouds
    in the image are. It uses different filter values to separate the pixels.
    The value of the pixels are specified in mudis.cloud_sky() function.

    Parameters
    ------------
        -img: Image file name read with mudis.read()
        -index_fov: Variable which loads the file directories of the FOV index
            files. It uses the mudis.read() function.
        -ind: Number of the index of fibre in the optical input (MUDIS:(0-112)

    """
    image = scipy.misc.imread(img)

    # convert image to grayscale
    img_gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    #img_gray = 0.333 * img_r + 0.5 * img_g + 0.1666 * img_b

    # read FOV file from disk
    fov_pix = h5py.File(config['personal_libraries_path'] + 'MUDIS_HDF5/MUDIS_config/fov_pixels.h5', 'r')

    # create array to save data
    cloud_cover = np.zeros([113, 2])
    brightness = np.zeros([113, 2])

    np.seterr(divide='ignore', invalid='ignore')  # To ignore zero division

    for ind in np.arange(113):
        # Import pixels in FOV area like integer
        pixels_fov = fov_pix['{:03d}_fov_pixels'.format(ind)][:]

        # Trim the image for a FOV in the image
        image_fov = np.ones([len(image), len(image)]) * 2  # Create a empty image
        # matrix # dtype=np.uint8)

        image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] = img_gray[pixels_fov[:, 0], pixels_fov[:, 1]]

        # count sky pixels (sky pixel have value 1 in filter function)
        sky = np.count_nonzero(image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] == 1)

        # count external pixels
        external = np.count_nonzero(image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] == 0)

        # total pixels in image
        total = image_fov[pixels_fov[:, 0], pixels_fov[:, 1]].size

        cloud_quant = ((total - external) - sky) / (total - external)

        cloud_cover[ind] = cloud_quant, ind
        brightness[ind] = (img_gray[pixels_fov[:, 0], pixels_fov[:, 1]] - 1).mean(), ind

    fov_pix.close()

    # Mean of values in the whole picture
    cloud_cover_mean = float(cloud_cover[:, 0].mean())
    cloud_brightness_mean = float(brightness[:, 0].mean())

    # save data in HDF5 file
    date = image_date(img)[0]
    utc = image_date(img)[1]
    name = '{:4d}{:02d}{:02d}'.format(date[0], date[1], date[2])
    full_name = '{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(date[0], date[1],
                                                              date[2], date[3],
                                                              date[4], date[5])
    time = datetime.datetime.strptime(full_name,
                                      '%Y%m%d_%H%M%S')

    # Create the directory to save the results
    os.makedirs(os.path.dirname(config['path_save_img'] + '{}/cloud_cover/'.format(name)),
                exist_ok=True)

    with h5py.File(config['path_save_img'] + '{}/cloud_cover/{}_UTC+0{}_cloud_cover.h5'.format(name, full_name, utc), 'w') as cloud:

        if 'cloud_cover' in list(cloud.keys()):
            del cloud['cloud_cover']
            del cloud['brightness']

            cloud.create_dataset('/cloud_cover', data=cloud_cover)
            cloud['/cloud_cover'].attrs['Units'] = 'Fraction'
            cloud['/cloud_cover'].attrs['Columns'] = 'Cloud cover, Channel'

            cloud.create_dataset('/brightness', data=brightness)
            cloud['/brightness'].attrs['Units'] = 'Intensity pixels'
            cloud['brightness'].attrs['Columns'] = 'Brightness, Channel'

        else:
            # data = cloud_c['cloud_cover']
            # data[...] = cloud_cover
            cloud.create_dataset('/cloud_cover', data=cloud_cover)
            cloud['/cloud_cover'].attrs['Units'] = 'Fraction'
            cloud['/cloud_cover'].attrs['Columns'] = 'Cloud cover, Channel'

            cloud.create_dataset('/brightness', data=brightness)
            cloud['/brightness'].attrs['Units'] = 'Intensity pixels'
            cloud['brightness'].attrs['Columns'] = 'Brightness, Channel'

        cloud['/'].attrs[
            'Information'] = 'Cloud cover and brightness for each MUDIS channel'
        cloud['/'].attrs['Time'] = str(time)
        cloud['/'].attrs['UTC'] = utc
        cloud['/'].attrs['Cloud_cover_mean'] = cloud_cover_mean
        cloud['/'].attrs['Brightness_mean'] = cloud_brightness_mean
        cloud['/'].attrs['Latitude'] = '52.39N'
        cloud['/'].attrs['Longitude'] = '9.7E'
        cloud['/'].attrs['Altitude'] = '65 AMSL'
        cloud['/'].attrs['Camera'] = 'Canon g10 with 180º fish eye lens'


# @jit
# def cloud_haze_quantity(img, config):
#     """ In development:
#
#     Function calculates the quantity of pixels where clouds
#     in the image are. It uses different filter values to separate the pixels.
#     The value of the pixels are specified in mudis.cloud_sky() function.
#
#     Parameters
#     ------------
#         -img: Image file name read with mudis.read()
#         -index_fov: Variable which loads the file directories of the FOV index
#             files. It uses the mudis.read() function.
#         -ind: Number of the index of fibre in the optical input (MUDIS:(0-112)
#
#     """
#     image = scipy.misc.imread(img)
#
#     # convert image to grayscale
#     img_gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
#     #img_gray = 0.333 * img_r + 0.5 * img_g + 0.1666 * img_b
#
#     # read FOV file from disk
#     fov_pix = h5py.File(config['personal_libraries_path'] + 'MUDIS_HDF5/MUDIS_config/fov_pixels.h5', 'r')
#
#     # create array to save data
#     cloud_cover = np.zeros([113, 2])
#     brightness = np.zeros([113, 2])
#
#     np.seterr(divide='ignore', invalid='ignore')  # To ignore zero division
#
#     for ind in np.arange(113):
#         # Import pixels in FOV area like integer
#         pixels_fov = fov_pix['{:03d}_fov_pixels'.format(ind)][:]
#
#         # Trim the image for a FOV in the image
#         image_fov = np.zeros([len(image), len(image)])  # Create a empty image
#         # matrix
#         # dtype=np.uint8)
#
#         image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] = img_gray[pixels_fov[:, 0], pixels_fov[:, 1]]
#
#         # count sky pixels (sky pixel have value 1 in filter function. circle_sun has value 2)
#         sky = np.count_nonzero(image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] >= 0.2)
#
#         # count external pixels
#         external = np.count_nonzero(image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] == 0)
#
#         # total pixels in image
#         total = image_fov[pixels_fov[:, 0], pixels_fov[:, 1]].size
#
#         cloud_quant = ((total - external) - sky) / (total - external)
#
#         cloud_cover[ind] = cloud_quant, ind
#         brightness[ind] = (img_gray[pixels_fov[:, 0], pixels_fov[:, 1]] - 1).mean(), ind
#
#     fov_pix.close()
#
#     cloud_cover_mean = float(cloud_cover[:, 0].mean())
#     cloud_brightness_mean = float(brightness[:, 0].mean())
#
#     # save data in HDF5 file
#     date = image_date(img)[0]
#     utc = image_date(img)[1]
#     name = '{:4d}{:02d}{:02d}'.format(date[0], date[1], date[2])
#     full_name = '{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(date[0], date[1],
#                                                               date[2], date[3],
#                                                               date[4], date[5])
#     time = datetime.datetime.strptime(full_name,
#                                       '%Y%m%d_%H%M%S')
#
#     # Create the directory to save the results
#     os.makedirs(os.path.dirname(config['path_save_img'] + '{}/cloud_cover/'.format(name)),
#                 exist_ok=True)
#
#     with h5py.File(config['path_save_img'] + '{}/cloud_cover/{}_UTC+0{}_cloud_cover.h5'.format(name, full_name, utc), 'w') as cloud:
#
#         if 'cloud_cover' in list(cloud.keys()):
#             del cloud['cloud_cover']
#             del cloud['brightness']
#
#             cloud.create_dataset('/cloud_cover', data=cloud_cover)
#             cloud['/cloud_cover'].attrs['Units'] = 'Fraction'
#             cloud['/cloud_cover'].attrs['Columns'] = 'Cloud cover, Channel'
#
#             cloud.create_dataset('/brightness', data=brightness)
#             cloud['/brightness'].attrs['Units'] = 'Intensity pixels'
#             cloud['brightness'].attrs['Columns'] = 'Brightness, Channel'
#
#         else:
#             # data = cloud_c['cloud_cover']
#             # data[...] = cloud_cover
#             cloud.create_dataset('/cloud_cover', data=cloud_cover)
#             cloud['/cloud_cover'].attrs['Units'] = 'Fraction'
#             cloud['/cloud_cover'].attrs['Columns'] = 'Cloud cover, Channel'
#
#             cloud.create_dataset('/brightness', data=brightness)
#             cloud['/brightness'].attrs['Units'] = 'Intensity pixels'
#             cloud['brightness'].attrs['Columns'] = 'Brightness, Channel'
#
#         cloud['/'].attrs[
#             'Information'] = 'Cloud cover and brightness for each MUDIS channel'
#         cloud['/'].attrs['Time'] = str(time)
#         cloud['/'].attrs['UTC'] = utc
#         cloud['/'].attrs['Cloud_cover_mean'] = cloud_cover_mean
#         cloud['/'].attrs['Brightness_mean'] = cloud_brightness_mean
#         cloud['/'].attrs['Latitude'] = '52.39N'
#         cloud['/'].attrs['Longitude'] = '9.7E'
#         cloud['/'].attrs['Altitude'] = '65 AMSL'
#         cloud['/'].attrs['Camera'] = 'Canon g10 with 180º fish eye lens'
#
