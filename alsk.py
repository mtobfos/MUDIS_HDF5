import datetime
import glob
import h5py
from numba import jit
from matplotlib import colors
import numpy as np
import os
import PIL.Image
import PIL.ExifTags
from PIL import Image as pilmg
from pysolar import solar as ps
import scipy.misc
import shutil
import time

# ----------------------------------------------------------------------------
# Functions which can be compiled with numba.jit. It is faster than another
# methods
# ----------------------------------------------------------------------------


def validate_measurements(raw_directory, copied_directory, indexs=[0, 1]):
    """Read ALLSKY JPG metadata, if ExposureTime is different to 1/2000 ms, do
    not copy it into another raw_directory"""

    files = sorted(glob.glob(raw_directory + '*.JPG'))

    for i in np.arange(indexs[0], indexs[1]):
        img = PIL.Image.open(files[i])
        exif = {PIL.ExifTags.TAGS[k]: v for k, v in img._getexif().items() if
                k in PIL.ExifTags.TAGS}

        if exif['ExposureTime'] == 2000:
            shutil.copy2(files[i], copied_directory)
            time.sleep(1)
        else:
            print('do not complain condition')
            
    print('completed')


def circle_aoi(image, aoi, radio_image):
    """
    Crop a image with the area-of-interest- defined in the image parameter. The
    cropped images is imported in the init method
     """
    # create a grid of coordinates pixel position
    xx, yy = np.mgrid[:2 * radio_image, :2 * radio_image]
    # mask a circle using the grid
    mask = (xx - radio_image) ** 2 + (yy - radio_image) ** 2
    # evaluate the circle position using the aoi as radius
    area = np.logical_and(mask < (aoi ** 2), mask >= 0)

    img = np.zeros([len(image), len(image), 3], dtype='uint8')

    img[..., 0] = image[..., 0] * area
    img[..., 1] = image[..., 1] * area
    img[..., 2] = image[..., 2] * area

    return img


##############################################################################
# HEMISPHERICAL FOV


@jit
def zen(image):
    RADIO_IMAGE = int(len(image) / 2)
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
    """
    Return the azimuth angles on the surface of a ALL-SKY image
    """

    RADIO_IMAGE = int(len(image) / 2)
    COSK = np.zeros([len(image), len(image)])
    RADIO_FOV = np.linspace(0, RADIO_IMAGE, num=6000)
    azimuth = np.linspace(0, 360, num=10000)

    for RADIO_F in RADIO_FOV:
        for azim in azimuth:
            x = int(RADIO_F * np.cos(np.radians(azim - 180))) - RADIO_IMAGE
            y = int(RADIO_F * np.sin(np.radians(azim - 180))) - RADIO_IMAGE
            COSK[x, y] = azim
    return COSK


# HEMISPHERICAL_CIRCLE_3D_TO_2D from Michael Routines

@jit(nogil=True)
def hemispherical_circle_3D_to_2D(fov=9, zen_direction=0, azim_direction=0,
                                  zen=0, cosk=0, image=None,
                                  radio_image=''):
    """ Determine the FOV projection over an all-sky image.
    To speed up the calculus import the zen and cosk variable on the notebook

    last modified 2016.12.27 by Mario Tobar """

    if image is not None:
        RADIO_IMAGE = int(len(image) / 2)
        print('Image used for calculation')
    else:
        RADIO_IMAGE = int(radio_image)

    FOV = fov

    delta_x = int((RADIO_IMAGE * (np.pi / 180) * FOV) * 30)

    geo_b = 90 - zen
    geo_l = cosk

    zenith = zen_direction
    azimu = azim_direction

    if image is None:
        circle_array = np.zeros(
            [2 * int(radio_image), 2 * int(radio_image)])

    else:
        circle_array = image

    x_pos = RADIO_IMAGE - RADIO_IMAGE * np.cos(
        np.radians(azimu)) * zenith / 90

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
                                           np.radians(
                                               geo_b[ax:ex, ay:ey])) *
                                        np.cos(np.radians(sou_b)) *
                                        np.cos(np.radians(
                                            sou_l - geo_l[ax:ex,
                                                    ay:ey])))) * \
                             180 / np.pi

    ab_klgl_FOV = np.transpose(np.where(distance <= (FOV / 2)))

    circle_array[ab_klgl_FOV[:, 0], ab_klgl_FOV[:, 1]] = 1.1

    # Delete values out the image area
    circle_aoi(circle_array, ASImage.aoi, ASImage.radio_image)

    return circle_array


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
                                              azim_direction=azim_direction + int(config['nord_desv']),
                                              zen=zen, cosk=cosk,
                                              image=image, radio_image=radio_image)

    pixels_fov_index = np.transpose(np.where(fov_array >= 1))
    pixels_fov_index.astype(int)

    return pixels_fov_index


@jit
def fov_save(config, zen='', cosk='',
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
    print("Completed")

    return

# ----------------------------------------------
# Classes of library


class ASImage(object):
    """
    Class used for work with allsky images
    """

    aoi = 1000
    cx = 1675
    cy = 2270
    radio_image = 1082

    def __init__(self, file, config):
        self.file = file
        self.config = config
        self.image = self.crop_import(np.asarray(pilmg.open(self.file,
                                                        mode='r'), dtype='uint8'))

    @staticmethod
    def crop_import(image):
        """
        Crop the readed image in the _init_ method. Only used for _init_method

        """
        # create a grid of coordinates pixel position
        xx, yy = np.mgrid[:2 * ASImage.radio_image, :2 * ASImage.radio_image]
        # mask a circle using the grid
        mask = (xx - ASImage.radio_image) ** 2 + (
                    yy - ASImage.radio_image) ** 2
        # evaluate the circle position using the aoi as radius
        area = np.logical_and(mask < (ASImage.aoi ** 2), mask >= 0)
        # crop image
        img_crop = image[ASImage.cx - ASImage.radio_image:ASImage.cx + ASImage.radio_image,
                   ASImage.cy - ASImage.radio_image:ASImage.cy + ASImage.radio_image,
                   :]

        filtered = np.zeros([2 * ASImage.radio_image, 2 * ASImage.radio_image, 3],
            dtype='uint8')

        filtered[..., 0] = img_crop[..., 0] * area
        filtered[..., 1] = img_crop[..., 1] * area
        filtered[..., 2] = img_crop[..., 2] * area

        return filtered

    def datetime(self):
        """
         Function returns the datetime and UTC information of the all-sky
         imagen loaded in file
        """
        date = os.path.split(self.file)
        # Variable to save the file name
        date2 = date[1]

        # Variable for the correct time
        utc = int(date2[20:22])
        date3 = (int('20' + date2[2:4]), int(date2[4:6]), int(date2[6:8]),
                 int(date2[9:11]), int(date2[11:13]), int(date2[13:15]))

        return date3, utc

    def circle_sun(self, zen='', cosk=''):
        """ Place a circle over the sun position.
        sun position is determined by using of the function datetime.datetime() and
        mudis.hemispherical_circle_3D_to_2D()

         IMAGE: Image cropped using the function alsk.filtered()
         DATE: parameter obtained from the name of the image. We need the local
               time, therefore we use a UTC of +2 for Hannover.
               :type utc: object

         """
        date, utc = ASImage.datetime(self)

        # Coordinates of the position of station Hannover.
        latitude = 52.39  # positive in the northern hemisphere
        longitude = 9.7  # negative reckoning west from prime meridian in
        # Greenwich,


        # Calculate the azimuth and zenith angles in function of the date
        d = datetime.datetime(date[0], date[1], date[2], date[3] + utc,
                              date[4],
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

        print("Azimuth: {:5.1f}".format(azi),
              "\nZenith: {:5.1f}".format(zenith))

        # calculate position of zenith and azimuth over the image
        image_filtered = hemispherical_circle_3D_to_2D(fov=3,
                                                       zen_direction=zenith,
                                                       azim_direction=azi + int(self.config['nord_desv']),
                                                       zen=zen,
                                                       cosk=cosk,
                                                       image=self.image)
        return image_filtered

    def north_plot(self, zen='', cosk=''):
        """ plot a circle in the allky image. The aligment can be found
        changing the value in the config['nord_desv'] parameter"""
        zenith = 80
        azi = 0
        # calculate position of zenith and azimuth over the image
        image_filtered = hemispherical_circle_3D_to_2D(fov=4,
                                                       zen_direction=zenith,
                                                       azim_direction=azi + int(self.config['nord_desv']),
                                                       zen=zen,
                                                       cosk=cosk,
                                                       image=self.image)
        return image_filtered

    def haze_index(self, zen, cosk):
        """
        Identifies clouds and sky using the SkyIndex and HazeIndex and classifies
        in function of colors. Clouds are red of values between -1 and 0.14. Sky
        is blue with values between 0.2 and 1.
        This functions uses cloud_haze_quantity() method to extract and save
        cloud properties.

        Parameters

        -----------

        file: Directory to image file.

        zen: Matrix with zenith angles projected on a horizontal plane. It is ob-
            tained with the function Allsky_images.zen().

        cosk: Matrix with azimuth angles projected on a horizontal plane. It is ob-
            tained with the function Allsky_images.cosk().


        config: Configuration dictionary

        # Return
        #
        # ------
        #
        # filtered: Numpy array which contains values after classification
        """

        img2 = np.asfarray(self.image, dtype='uint8')  # Important

        red = img2[:, :, 0]
        green = img2[:, :, 1]
        blue = img2[:, :, 2]

        # sky index
        sky = (blue - red) / (blue + red)

        # pixels in middle range
        ind = np.where((0.14 <= sky) & (sky < 0.2))

        # haze index
        haze = (((red + blue) / 2) - green) / (((red + blue) / 2) + green)

        # Put values of sky index in haze and evaluate middle range
        haze_color = sky
        haze_color[ind] = np.asarray(haze[ind], dtype='uint8')

        val_corr = np.where((0 <= haze_color[ind]), 0.14, 0.2)

        haze_color[ind] = val_corr

        # make a color map of fixed colors
        cmap = colors.ListedColormap(
            ['black', 'red', 'orangered', 'orange', 'yellow', 'green', 'aqua',
             'blue', 'navy', 'purple'])
        bounds = [-2, -1, 0, 0.1, 0.14, 0.18, 0.2, 0.25, 0.4, 0.5, 1]

        # a colormap and a normalization instance
        norm = colors.BoundaryNorm(bounds, cmap.N)

        image_c = cmap(norm(haze_color))  # Image to save with the correct colors RGBA

        image_sa = image_c[:, :, 0:3]  # Load only RGB layers into array

        # Calculate the date and utc information from the image
        date = ASImage.datetime(self)[0]
        utc = ASImage.datetime(self)[1]

        # grayscale image for brightness cloud calculations
        gray_img = np.dot(img2[..., :3], [0.299, 0.587, 0.114])

        # Add circle in the sun position
        self.image = image_sa

        filtered = ASImage.circle_sun(self, zen=zen, cosk=cosk)

        filtered = circle_aoi(filtered, ASImage.aoi, ASImage.radio_image)
        # plt.imshow(filtered)

        name = '{:4d}{:02d}{:02d}'.format(date[0], date[1], date[2])  # date

        full_name = '{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(date[0],
                                                                  date[1],
                                                                  date[2],
                                                                  date[3],
                                                                  date[4],
                                                                  date[5])
        # Create the directory to save the results
        os.makedirs(
            os.path.dirname(self.config['path_save_img'] + '{}/'.format(name)),
            exist_ok=True)

        scipy.misc.imsave(
            self.config['path_save_img'] + '{}/{}_UTC+0{}_filtered.png'.format(
                name, full_name, utc), filtered)

        # ---- cloud cover information----------

        ASImage.cloud_haze_quantity(self, haze_color, gray_img)

    def cloud_haze_quantity(self, haze_color, gray_img):
        """ In development:

        Function calculates the quantity of pixels where clouds
        in the image are. It uses different filter values to separate the pixels.
        The value of the pixels are specified in sky index and haze index

        Information about colors:
            -1.0-> 0.0  : hell white clouds(saturation)
            0.0 -> 0.1  : white clouds
            0.1 -> 0.14 : gray clouds
            0.14-> 0.20 : undefined range (thin cloud, nebel or very bright sky)
            0.20-> 0.25 : clear sky
            0.25-> 0.40 : blue sky
            0.40-> 1.00 : dark blue sky

        """
        image = haze_color

        # read FOV file from disk laptop
        fov_pix = h5py.File(self.config[
                                'personal_libraries_path'] + 'MUDIS_HDF5/MUDIS_config/fov_pixels.h5',
                            'r')

        # create array to save data
        cloud_cover = np.zeros([113, 2])
        brightness = np.zeros([113, 2])

        np.seterr(divide='ignore',
                  invalid='ignore')  # To ignore zero division

        for ind in np.arange(113):
            # Import pixels in FOV area like integer
            pixels_fov = fov_pix['{:03d}_fov_pixels'.format(ind)][:]

            # Trim the image for a FOV in the image
            image_fov = np.ones([len(image), len(image)]) * 2  # Create a empty image
            # matrix
            # dtype=np.uint8)

            image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] = image[pixels_fov[:, 0], pixels_fov[:, 1]]

            # ---- cloud cover calculation----

            # count sky pixels (sky pixel have value 1 in filter function. circle_sun has value 2)
            sky = np.count_nonzero((
                image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] >= 0.20) & (image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] <= 1))

            # count external pixels
            external = np.count_nonzero(
                image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] == np.nan)

            # total pixels in image
            total = image_fov[pixels_fov[:, 0], pixels_fov[:, 1]].size

            cloud_quant = ((total - external) - sky) / (total - external)

            cloud_cover[ind] = ind, cloud_quant

            # ---brightness calculation-----

            # find cloud pixels in the FOV area
            cloud_pixels = np.where((image_fov <= 0.14) & (image_fov >= -1))

            # Evaluation of cloudy pixels
            if len(cloud_pixels) >= 1 :
                brightness[ind] = ind, np.nanmean(gray_img[cloud_pixels])
            else:
                brightness[ind] = ind, np.nan

        fov_pix.close()

        cloud_cover_mean = float(cloud_cover[:, 0].mean())
        cloud_brightness_mean = float(brightness[:, 0].mean())

        # save data in HDF5 file
        date, utc = ASImage.datetime(self)

        name = '{:4d}{:02d}{:02d}'.format(date[0], date[1], date[2])
        full_name = '{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(date[0],
                                                                  date[1],
                                                                  date[2],
                                                                  date[3],
                                                                  date[4],
                                                                  date[5])
        time = datetime.datetime.strptime(full_name, '%Y%m%d_%H%M%S')

        # Create the directory to save the results
        os.makedirs(os.path.dirname(
            self.config['path_save_img'] + '{}/cloud_cover/'.format(name)),
                    exist_ok=True)

        with h5py.File(self.config[
                           'path_save_img'] + '{}/cloud_cover/{}_UTC+0{}_cloud_cover.h5'.format(
                name, full_name, utc), 'w') as cloud:

            if 'cloud_cover' in list(cloud.keys()):
                del cloud['cloud_cover']

                cloud.create_dataset('/cloud_cover', data=cloud_cover, dtype='f4')
                cloud['/cloud_cover'].attrs['Units'] = 'Fraction'
                cloud['/cloud_cover'].attrs[
                    'Columns'] = 'Channel, Cloud cover'
            else:
                # data = cloud_c['cloud_cover']
                # data[...] = cloud_cover
                cloud.create_dataset('/cloud_cover', data=cloud_cover, dtype='f4')
                cloud['/cloud_cover'].attrs['Units'] = 'Fraction'
                cloud['/cloud_cover'].attrs[
                    'Columns'] = 'Channel, Cloud cover'

            if 'brightness' in list(cloud.keys()):
                del cloud['brightness']
                cloud.create_dataset('/brightness', data=brightness, dtype='f4')
                cloud['/brightness'].attrs['Units'] = 'Intensity pixels'
                cloud['brightness'].attrs['Columns'] = 'Channel, Brightness'
            else:
                cloud.create_dataset('/brightness', data=brightness, dtype='f4')
                cloud['/brightness'].attrs['Units'] = 'Intensity pixels'
                cloud['brightness'].attrs['Columns'] = 'Channel, Brightness'

            cloud.attrs['Information'] = 'Cloud cover and brightness for each MUDIS channel'
            cloud.attrs['Time'] = str(time)
            cloud.attrs['UTC'] = utc
            cloud.attrs['Cloud_cover_mean'] = cloud_cover_mean
            cloud.attrs['Brightness_mean'] = cloud_brightness_mean
            cloud.attrs['Latitude'] = '52.39N'
            cloud.attrs['Longitude'] = '9.7E'
            cloud.attrs['Altitude'] = '65 AMSL'
            cloud.attrs['Camera'] = 'Canon g10 with 180º fish eye lens'

    #-------------------------------------------------------------------------

    # def cloud_haze_quantity(self, haze_color):
    #     """ In development:
    #
    #     Function calculates the quantity of pixels where clouds
    #     in the image are. It uses different filter values to separate the pixels.
    #     The value of the pixels are specified in sky index and haze index
    #
    #     borders colors:
    #         -1.0-> 0.0  : hell white clouds(saturation)
    #         0.0 -> 0.1  : white clouds
    #         0.1 -> 0.14 : gray clouds
    #         0.14-> 0.20 : undefined range (thin cloud, nebel or very bright sky)
    #         0.20-> 0.25 : clear sky
    #         0.25-> 0.40 : blue sky
    #         0.40-> 1.00 : dark blue sky
    #
    #     """
    #     image = self.image
    #
    #     # convert image to grayscale
    #     img_gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    #
    #     # read FOV file from disk laptop
    #     fov_pix = h5py.File(self.config[
    #                             'personal_libraries_path'] + 'MUDIS_HDF5/MUDIS_config/fov_pixels.h5',
    #                         'r')
    #
    #     # create array to save data
    #     cloud_cover = np.zeros([113, 2])
    #     brightness = np.zeros([113, 2])
    #
    #     np.seterr(divide='ignore',
    #               invalid='ignore')  # To ignore zero division
    #
    #     for ind in np.arange(113):
    #         # Import pixels in FOV area like integer
    #         pixels_fov = fov_pix['{:03d}_fov_pixels'.format(ind)][:]
    #
    #         # Trim the image for a FOV in the image
    #         image_fov = np.zeros(
    #             [len(image), len(image)])  # Create a empty image
    #         # matrix
    #         # dtype=np.uint8)
    #
    #         image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] = img_gray[
    #             pixels_fov[:, 0], pixels_fov[:, 1]]
    #
    #         # ---- cloud cover calculation----
    #         # count sky pixels (sky pixel have value 1 in filter function. circle_sun has value 2)
    #         sky = np.count_nonzero(
    #             image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] >= 0.20)
    #
    #         # count external pixels
    #         external = np.count_nonzero(
    #             image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] == 0)
    #
    #         # total pixels in image
    #         total = image_fov[pixels_fov[:, 0], pixels_fov[:, 1]].size
    #
    #         cloud_quant = ((total - external) - sky) / (total - external)
    #
    #         cloud_cover[ind] = cloud_quant, ind
    #
    #         # ---brightness calculation-----
    #         # find cloud pixels in the FOV area
    #         cloud_pixels = np.where(-1 < image_fov[pixels_fov[:, 0], pixels_fov[:, 1]] <= 0.14)
    #
    #         brightness[ind] = img_gray[cloud_pixels].mean(), ind
    #
    #     fov_pix.close()
    #
    #     cloud_cover_mean = float(cloud_cover[:, 0].mean())
    #     cloud_brightness_mean = float(brightness[:, 0].mean())
    #
    #     # save data in HDF5 file
    #     date, utc = ASImage.datetime(self)
    #
    #     name = '{:4d}{:02d}{:02d}'.format(date[0], date[1], date[2])
    #     full_name = '{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(date[0],
    #                                                               date[1],
    #                                                               date[2],
    #                                                               date[3],
    #                                                               date[4],
    #                                                               date[5])
    #     time = datetime.datetime.strptime(full_name,
    #                                       '%Y%m%d_%H%M%S')
    #
    #     # Create the directory to save the results
    #     os.makedirs(os.path.dirname(
    #         self.config['path_save_img'] + '{}/cloud_cover/'.format(name)),
    #                 exist_ok=True)
    #
    #     with h5py.File(self.config[
    #                        'path_save_img'] + '{}/cloud_cover/{}_UTC+0{}_cloud_cover.h5'.format(
    #             name, full_name, utc), 'w') as cloud:
    #
    #         if 'cloud_cover' in list(cloud.keys()):
    #             del cloud['cloud_cover']
    #
    #             cloud.create_dataset('/cloud_cover', data=cloud_cover)
    #             cloud['/cloud_cover'].attrs['Units'] = 'Fraction'
    #             cloud['/cloud_cover'].attrs[
    #                 'Columns'] = 'Cloud cover, Channel'
    #
    #         else:
    #             # data = cloud_c['cloud_cover']
    #             # data[...] = cloud_cover
    #             cloud.create_dataset('/cloud_cover', data=cloud_cover)
    #             cloud['/cloud_cover'].attrs['Units'] = 'Fraction'
    #             cloud['/cloud_cover'].attrs[
    #                 'Columns'] = 'Cloud cover, Channel'
    #
    #         cloud['/'].attrs[
    #             'Information'] = 'Cloud cover and brightness for each MUDIS channel'
    #         cloud['/'].attrs['Time'] = str(time)
    #         cloud['/'].attrs['UTC'] = utc
    #         cloud['/'].attrs['Cloud_cover_mean'] = cloud_cover_mean
    #         cloud['/'].attrs['Latitude'] = '52.39N'
    #         cloud['/'].attrs['Longitude'] = '9.7E'
    #         cloud['/'].attrs['Altitude'] = '65 AMSL'
    #         cloud['/'].attrs['Camera'] = 'Canon g10 with 180º fish eye lens'
