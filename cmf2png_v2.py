"""
Winston Olson-Duvall 2017-08-30
Converts 4 band CMF image to PNG.  Assumes "bip" type input.
"""
# License: Apache 2.0 (http://www.apache.org/licenses/)


import argparse
import colorsys
from scipy.misc import imsave
import spectral.io.envi as envi
import sys, os
import scipy.ndimage as ndimage
from scipy import zeros, memmap, array, where, uint8
from scipy import int16, uint16, uint32, float32, float64


def infer_data_type(type_flag):
    """
    This serves as a lookup table for IDL types - maps them to numpy types
    :param type_flag: the integer parsed from the image's ENVI .hdr
    :return: a numpy type that matches the IDL type
    """
    type_dict = {2: int16, 12: uint16, 3: uint32, 4: float32, 5: float64}
    return type_dict[type_flag]


def linear_scale(bs, low_clip, high_clip):
    """
    This takes non-negative values of the array, sorts them, and then uses
     thresholds to scale the data to that range. Then it clips values to the
     range it is possible to represent in jpg format
    :param bs: the data to be scaled
    :param low_clip: the fraction of pixels, below which to assign to 0 in jpg
    :param high_clip: the fraction of pixels, above which to assign 255 in jpg
    :return: the scaled data
    """
    mask = bs[:] == -9999
    use = bs[:] > 0
    vals = sorted(bs[use])
    low_thresh = vals[int(len(vals) * low_clip)]
    high_thresh = vals[int(len(vals) * high_clip)]
    bs[where(bs < low_thresh)] = 0
    bs[where(bs >= low_thresh)] -= low_thresh
    bs *= (255 / (high_thresh - low_thresh))  # scale to jpg values
    bs[where(bs < 1)] = 1  # clip values (use 1 to avoid transparency)
    bs[where(bs > 255)] = 255  # clip values
    bs[mask] = 0
    return bs

def enhance_cmf_layer(bs, mask_layer):
    mask = mask_layer[:] == -9999
    bs[where(bs < 1)] = 1  # clip values
    bs[where(bs > 2000)] = 2000  # clip values    
    bs *= (255 / (2000.0 - 0.0))  # scale to jpg values
    bs[where(bs < 1)] = 1  # clip values (use 1 to avoid transparency)
    bs[where(bs > 255)] = 255  # clip values    
    bs[mask] = 0
    return bs

def enhance_layer_old(bs):

    bs[where(bs < 0)] = 0  # clip values
    bs[where(bs > 2000)] = 2000  # clip values    

    ch4 = zeros((bs.shape[0], bs.shape[1]), dtype=uint8)
#    ch4 = img_layer*100000 > thresh_ppmm
    ch4 = bs*100000 > 1000
    ch4 = ndimage.morphology.binary_opening(ch4)
    bs = 0
    bs[where(ch4 == True)] = 255
#    bs *= (255 / (2000.0 - 0.0))  # scale to jpg values
    return bs

def make_ql(file_name, output_directory, output_type):

    # find and set needed paths
    in_hdr = file_name + '.hdr'
    basename = os.path.basename(file_name)

    out_file = basename + '_rgb.png'
    if output_type == 'rgb_image':
        out_file = basename.replace('_cmf_','_rdn_').replace('_ch4mf_','_rdn_') + '.png'
    if output_type == 'gray_image':
        out_file = basename.replace('_cmf_','_rdn_') + '_gray.png'
    if output_type == 'cmf_layer':
        out_file = basename + '_gray.png'
    if output_type == 'rgb_detections':
        out_file = basename.replace('_img','_det') + '_rgb.png'
    if output_type == 'blue_detections':
        out_file = basename.replace('_img','_det') + '_blue.png'
    
    out_path = os.path.join(output_directory, out_file)

    # read and parse header parameters
    hdr = envi.read_envi_header(in_hdr)
    lines = int(hdr['lines'])
    samples = int(hdr['samples'])
    in_bands = int(hdr['bands'])
    type_flag = int(hdr['data type'])

    # set up output array and infer dtype string
    dtype_string = infer_data_type(type_flag=type_flag)
    output = zeros((lines, samples, 3), dtype=uint8)
    out_channels = ['red','green','blue']

    img = memmap(file_name, dtype=dtype_string, mode='r',
                 shape=(lines, samples, in_bands))

    if output_type == 'rgb_image':
        for ind, channel in enumerate(out_channels):
            print('working on channel %s' % channel)
            bs = array(img[:, :, ind], dtype=dtype_string)
            bs = linear_scale(bs=bs, low_clip=0.02, high_clip=0.98)
            output[:,:,ind] = array(bs, dtype=uint8)          

    if output_type == 'gray_image':
        print('working on channel %s' % out_channels[0])
        bs = array(img[:, :, 0], dtype=dtype_string)
        bs = linear_scale(bs=bs, low_clip=0.02, high_clip=0.98)
        output[:,:,0] = array(bs, dtype=uint8)
        output[:,:,1] = array(bs, dtype=uint8)
        output[:,:,2] = array(bs, dtype=uint8)

    if output_type == 'cmf_layer':
        print('converting CMF layer to grayscale')
        bs = array(img[:, :, 3], dtype=dtype_string)
        mask_layer = array(img[:, :, 0], dtype=dtype_string)
        bs = enhance_cmf_layer(bs, mask_layer)
        output[:,:,0] = array(bs, dtype=uint8)
        output[:,:,1] = array(bs, dtype=uint8)
        output[:,:,2] = array(bs, dtype=uint8)


    if output_type == 'rgb_detections':
        print('working on channel %s' % out_channels[0])
        bs = array(img[:, :, 0], dtype=dtype_string)
        bs = linear_scale(bs=bs, low_clip=0.02, high_clip=0.98)
        output[:,:,0] = array(bs, dtype=uint8)
        output[:,:,1] = array(bs, dtype=uint8)
        output[:,:,2] = array(bs, dtype=uint8)        
        # Now create detections overlay
        ch4_layer = img[:,:,3]
        ch4_layer = ndimage.gaussian_filter(ch4_layer,2)
        ch4 = zeros((img.shape[0], img.shape[1]), dtype=uint8)

        thresholds = range(500,2050,50)
        hue_max = 0.6
        step = hue_max / len(thresholds)
        for ind, thresh in enumerate(thresholds):
            print 'Working on ch4 layer with threshold %i' % thresh
            ch4 = ch4_layer > thresh
            ch4 = ndimage.morphology.binary_opening(ch4)

            hue = hue_max - ind * step
            saturation = 1.0
            value = 1.0
            red,green,blue = colorsys.hsv_to_rgb(hue, saturation, value)

            output[ch4,0] = 255 * red
            output[ch4,1] = 255 * green
            output[ch4,2] = 255 * blue

    if output_type == 'blue_detections':
        print('working on channel %s' % out_channels[0])
        bs = array(img[:, :, 0], dtype=dtype_string)
        bs = linear_scale(bs=bs, low_clip=0.02, high_clip=0.98)
        output[:,:,0] = array(bs, dtype=uint8)
        output[:,:,1] = array(bs, dtype=uint8)
        output[:,:,2] = array(bs, dtype=uint8)        
        # Now create detections overlay
        ch4_layer = img[:,:,3]
        ch4_layer = ndimage.gaussian_filter(ch4_layer,2)
        ch4 = zeros((img.shape[0], img.shape[1]), dtype=uint8)

        thresholds = range(500,2050,50)
        step = 255.0 / len(thresholds)
        for ind, thresh in enumerate(thresholds):
            print 'Working on ch4 layer with threshold %i' % thresh
            ch4 = ch4_layer > thresh
            ch4 = ndimage.morphology.binary_opening(ch4)
            output[ch4,0] = 255 - ind * step
            output[ch4,1] = 255 - ind * step
            output[ch4,2] = 255


    del img

    imsave(out_path, output)
    os.system('chmod a+rw ' + out_path)

    magick_convert_exe = '/shared/ImageMagick-7.0.3-4/bin/convert'
    cmd_transparent = ' '.join([magick_convert_exe, out_path, '-transparent black',
                                'PNG32:' + out_path])

    path_arr = out_path.split('_')
    ver_removed = [x for x in path_arr if not x.startswith('v')]
    out_path_resized = '_'.join(ver_removed)
    out_path_resized = out_path_resized.replace('.png','.x8000.png')
    cmd_resize_8000 = ' '.join([magick_convert_exe, out_path, '-resize', 
                    'x8000\>', out_path_resized])

    print(cmd_transparent)
    print(cmd_resize_8000)
    os.system(cmd_transparent)
    os.system(cmd_resize_8000)

def main():
    """
    read in arguments and run script
    :return:
    """

    # args = sys.argv[1:]
    # print 'Read in arguments %s' % args

    # cmf_path = args[0]
    # output_dir = './images/ql/'

    output_choices = ['rgb_image', 'gray_image', 'rgb_detections', 'blue_detections', 'cmf_layer']
    parser = argparse.ArgumentParser()
    parser.add_argument("cmf_path", help="path of 4 band CMF file to process")
    parser.add_argument("output_dir", help="output directory for png")
    parser.add_argument("--output_type", choices=output_choices, default="rgb_image")
    #parser.add_argument("--resize_8000", 
    #                help="creates a version with max 8000 lines (for Google Earth)",
    #--out                action="store_true")
    args = parser.parse_args()
    print ("output type is %s" % args.output_type)
    make_ql(args.cmf_path, args.output_dir, args.output_type)

if __name__ == "__main__":
    main()
