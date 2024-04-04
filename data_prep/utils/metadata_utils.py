"""
Author(s):
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com; abdoukamel@gmail.com)

Working with night mode's metadata text files.
"""
import struct

import numpy as np
import cv2


def get_bayer_by_type(bayer_type):
    """
    Convert bayer type (int) to bayer pattern.
    """
    if bayer_type == 1:
        bayer_pattern = [1, 0, 2, 1]  # GRBG
    elif bayer_type == 2:
        bayer_pattern = [1, 2, 0, 1]  # GBRG
    # TBC...
    else:
        bayer_pattern = [0, 1, 1, 2]  # RGGB
    return bayer_pattern


def reformat_night_mode_input_metadata(night_input_metadata, white_level=1023, bayer_pattern=(0, 1, 1, 2)):
    """
    Reformat metadata from night-mode input format into DNG format.
    """

    metadata_dng = {}

    height = night_input_metadata['image_height']
    width = night_input_metadata['image_width']

    metadata_dng['active_area'] = [0, 0, height, width]
    metadata_dng['linearization_table'] = None
    metadata_dng['black_level'] = night_input_metadata['black_level_array'][0]
    metadata_dng['as_shot_neutral'] = night_input_metadata['as_shot_neutral']

    if 'white_level' in night_input_metadata:
        metadata_dng['white_level'] = night_input_metadata['white_level']
    else:
        metadata_dng['white_level'] = white_level

    if 'bayer_type' in night_input_metadata:
        metadata_dng['cfa_pattern'] = get_bayer_by_type(night_input_metadata['bayer_type'])
    else:
        metadata_dng['cfa_pattern'] = bayer_pattern

    # for now:
    metadata_dng['default_crop_origin'] = None
    metadata_dng['default_crop_size'] = None
    metadata_dng['black_level_delta_h'] = None
    metadata_dng['black_level_delta_v'] = None
    metadata_dng['orientation'] = 0

    metadata_dng['color_matrix_1'] = None
    metadata_dng['color_matrix_2'] = None
    metadata_dng['calib_illum_1'] = None
    metadata_dng['calib_illum_2'] = None
    metadata_dng['hsv_lut'] = None
    metadata_dng['profile_lut'] = None

    if 'lens_gain_resized' in night_input_metadata:
        metadata_dng['opcode_lists'] = {}
        metadata_dng['opcode_lists'][51009] = {}
        lens_gain_resized = night_input_metadata['lens_gain_resized']
        for c in range(lens_gain_resized.shape[-1]):
            lens_gain_resized_channel = lens_gain_resized[..., c]
            opcode_lens_shade = Opcode(id_=9 + c / 10, dng_spec_ver=(1, 3, 0, 0), option_bits=1, size_bytes=960 / 4,
                                       data=None)
            opcode_lens_shade.data = {
                'top': 0,
                'left': 0,
                'bottom': height,
                'right': width,
                'plane': 0,
                'row_pitch': -1,
                'col_pitch': -1,
                'map_gain_2d': lens_gain_resized_channel,  # [H // 2, W // 2]
                # TBC...
            }
            metadata_dng['opcode_lists'][51009][opcode_lens_shade.id] = opcode_lens_shade

    # TBC...

    return metadata_dng


def reformat_night_mode_output_metadata(night_output_metadata, black_level, white_level, bayer_pattern=None):
    """
    Reformat metadata from night-mode output format into DNG format.
    """

    metadata_dng = {
        'input_height': night_output_metadata['input_height'],
        'input_width': night_output_metadata['input_width'],
        'black_level': black_level,
        'white_level': white_level,
    }

    if bayer_pattern is not None:
        metadata_dng['cfa_pattern'] = list(bayer_pattern)
    elif 'bayer_type' in night_output_metadata:
        metadata_dng['cfa_pattern'] = get_bayer_by_type(night_output_metadata['bayer_type'])
    else:
        metadata_dng['cfa_pattern'] = [0, 1, 1, 2]

    metadata_dng['as_shot_neutral'] = night_output_metadata['as_shot_neutral']
    metadata_dng['active_area'] = [0, 0, metadata_dng['input_height'], metadata_dng['input_width']],
    metadata_dng['linearization_table'] = None

    # for now:
    metadata_dng['default_crop_origin'] = None
    metadata_dng['default_crop_size'] = None
    metadata_dng['black_level_delta_h'] = None
    metadata_dng['black_level_delta_v'] = None
    metadata_dng['orientation'] = 0

    metadata_dng['color_matrix_1'] = night_output_metadata['color_matrix_1']
    metadata_dng['color_matrix_2'] = night_output_metadata['color_matrix_2']

    metadata_dng['calib_illum_1'] = None
    metadata_dng['calib_illum_2'] = None
    metadata_dng['hsv_lut'] = None
    metadata_dng['profile_lut'] = None

    # TBC...

    return metadata_dng


def read_mfp_input_raw_image(filename, height=None, width=None):
    """
    Load MFP input image.
    To skip black level subtraction, set black_level to be negative.
    :param filename:
    :param height:
    :param width:
    :param black_level:
    :return:
    """
    hws = {
        3060 * 4080: (3060, 4080),
        3000 * 4000: (3000, 4000),
        2860 * 3880: (2860, 3880),
        3024 * 4032: (3024, 4032),  # [Stavros]
        2736 * 3648: (2736, 3648),  # [Stavros]
        100 * 100: (100, 100)
    }

    with open(filename, mode='rb') as file:
        bytes_ = file.read()
    uint16_vals = struct.unpack("H" * ((len(bytes_)) // 2), bytes_)
    num_pixels = len(uint16_vals)
    if not height or not width:
        height = hws[num_pixels][0]
        width = hws[num_pixels][1]
    raw_image = np.reshape(uint16_vals, (height, width))
    return raw_image


def write_mfp_input_raw_image(filename, data):
    data_lin = np.reshape(data.copy(), (-1,))
    data_lin = np.round(data_lin).astype(np.uint16)
    bytes_ = struct.pack("H" * len(data_lin), *data_lin)
    with open(filename, mode='wb') as f:
        f.write(bytes_)


def read_mfp_metadata(fn):
    with open(fn, 'r') as f:
        line = f.readline()
    if line[0:2] == "//":
        metadata = read_mfp_output_metadata(fn)
    else:
        metadata = read_mfp_input_metadata(fn)
    return metadata


def read_mfp_input_metadata(fn):
    metadata = {}

    with open(fn, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # skip lines that are not key-value pairs separated by a colon
        if ':' not in line:
            continue
        key, val = line.strip().split(': ')
        if key == '' or val == '':
            continue
        val = val[:-1] if val[-1] == ',' else val  # remove comma at the end
        if key == 'i32Height':
            metadata['image_height'] = metadata['i32Height'] = int(val)
        elif key == 'i32Width':
            metadata['image_width'] = metadata['i32Width'] = int(val)
        elif key == 'fIso':
            metadata['iso'] = metadata['fIso'] = float(val)
        elif key == 'fShutter':
            metadata['exposure_time'] = metadata['fShutter'] = float(val)
        elif key == 'i32LensDataWidth':
            metadata['lens_gain_width'] = metadata['i32LensDataWidth'] = int(val)
        elif key == 'i32LensDataHeight':
            metadata['lens_gain_height'] = metadata['i32LensDataHeight'] = int(val)
        elif key == 'fLensDataGain':
            metadata['lens_gain'] = metadata['fLensDataGain'] = np.array([float(z) for z in val.split(',')])
            metadata['lens_gain_resized'] = np.reshape(metadata['lens_gain'],
                                                       (metadata['lens_gain_height'], metadata['lens_gain_width'], 4))
            metadata['lens_gain_resized'] = cv2.resize(metadata['lens_gain_resized'],
                                                       dsize=(
                                                           metadata['image_width'] // 2, metadata['image_height'] // 2))
        elif key == 'fBLOffset':
            metadata['black_offset'] = metadata['fBLOffset'] = np.array([float(z) for z in val.split(',')])
        elif key == 'fWbGain':
            metadata['as_shot_neutral'] = 1.0 / np.array(eval(val))[np.array([0, 1, 3])]
            metadata['fWbGain'] = np.array(eval(val))[np.array([0, 1, 2, 3])]
        elif key == 'i32BlackLevel':
            metadata['black_level_array'] = metadata['i32BlackLevel'] = np.array(eval('[' + val + ']'))
            metadata['black_level'] = metadata['black_level_array'][0]  # assuming all values are equal
        elif key == 'i32BrightLevel':
            metadata['white_level_array'] = metadata['i32BrightLevel'] = np.array(eval('[' + val + ']'))
            metadata['white_level'] = metadata['white_level_array'][0]  # assuming all values are equal
        elif key == 'bayerType':
            metadata['bayer_type'] = metadata['bayerType'] = int(val)
            metadata['cfa_pattern'] = get_bayer_by_type(metadata['bayer_type'])
        elif key == 'fCCM':
            metadata['ccm'] = metadata['fCCM'] = np.array(eval('[' + val.strip() + ']'))
        elif key == 'strCameraID':
            metadata['camera_id'] = metadata['strCameraID'] = val
        elif key == 'strSensorName':
            metadata['sensor'] = metadata['strSensorName'] = val
        elif key == 'strProductName':
            metadata['product'] = metadata['strProductName'] = val
        elif key == 'i32DeviceState':
            metadata['device_state'] = metadata['i32DeviceState'] = int(val)
        elif key == 'flashStatus':
            metadata['flash'] = metadata['flashStatus'] = int(val)
        elif key == 'i64ShootingMode':
            metadata['shooting_mode'] = metadata['i64ShootingMode'] = int(val)
        elif key == 'cameraClient':
            metadata['camera_client'] = metadata['cameraClient'] = int(val)
        elif key == 'fUserZoom':
            metadata['user_zoom'] = metadata['fUserZoom'] = float(val)
        elif key == 'fEV':
            metadata['ev_float'] = metadata['fEV'] = float(val)
        elif key == 'i32EV':
            metadata['ev_int'] = metadata['i32EV'] = int(val)
        elif key == 'fSensorGain':
            metadata['sensor_gain'] = metadata['fSensorGain'] = float(val)
        elif key == 'fISPGain':
            metadata['isp_gain'] = metadata['fISPGain'] = float(val)
        elif key == 'i32LuxIndex':
            metadata['lux_index'] = metadata['i32LuxIndex'] = int(val)
        elif key == 'i32CameraState':
            metadata['camera_state'] = metadata['i32CameraState'] = int(val)
        elif key == 'fBv':
            metadata['bv'] = metadata['fBv'] = float(val)
        elif key == 'i32CaptureEv':
            metadata['capture_ev'] = metadata['i32CaptureEv'] = float(val)
        else:
            pass

    # metadata['black_level'] = 64
    # metadata['white_level'] = 1023

    metadata['linearization_table'] = None
    metadata['orientation'] = 1

    metadata['super_night_crop_region'] = None  # applicable to output images only

    return metadata


def read_mfp_output_metadata(fn, bayer_pattern=(0, 1, 1, 2)):
    metadata = {}

    with open(fn, 'r') as f:
        lines = f.readlines()

    metadata['lens_gain_resized'] = None

    metadata['input_width'] = int(lines[1].strip()[:-1])
    metadata['input_height'] = int(lines[3].strip()[:-1])

    metadata['output_width'] = int(lines[5].strip()[:-1])
    metadata['output_height'] = int(lines[7].strip()[:-1])

    metadata['image_width'] = metadata['output_width']
    metadata['image_height'] = metadata['output_height']

    metadata['lens_facing'] = int(lines[9].strip()[:-1])

    ccm_str = lines[21].replace('}', '').replace('{', '').strip()
    ccm = np.array(eval(ccm_str))
    metadata['color_matrix_1'] = ccm
    metadata['color_matrix_2'] = ccm

    awb_str = lines[27].replace('}', '').replace('{', '').strip()
    awb = np.array(eval(awb_str))[np.array([0, 1, 3])]
    metadata['as_shot_neutral'] = 1.0 / awb

    # is MFP output image already white balanced? YES
    metadata['as_shot_neutral'] = [1, 1, 1]

    metadata['iso'] = int(lines[29].strip()[:-1])

    # i32BrightLevel: 4095, 4095, 4095, 4095,
    # i32BlackLevel: 64, 64, 64, 64,

    metadata['black_level'] = 64
    # metadata['white_level'] = 1023
    metadata['white_level'] = 4096

    metadata['linearization_table'] = None
    metadata['orientation'] = 1

    metadata['cfa_pattern'] = bayer_pattern

    metadata['active_array'] = [int(z) for z in lines[33].strip()[1:-2].split(',')]
    metadata['zoom_region'] = [int(z) for z in lines[35].strip()[1:-2].split(',')]
    metadata['super_night_crop_region'] = [int(z) for z in lines[37].strip()[1:-2].split(',')]

    # TODO: read remaining tags

    return metadata


def update_and_save_input_metadata(metadata_filename, save_to_filename, new_metadata_dict):
    """
    Read MFP input metadata; update ISO and exposure; save into another file.
    :param metadata_filename:
    :param save_to_filename:
    :param new_iso:
    :param new_exposure:
    :return:
    """

    """
    EXAMPLES:
    fEV: -2.000000,
    i32EV: -2,
    i32BrightLevel: 4095,4095,4095,4095,
    i32BlackLevel: 65,65,65,65,
    fWbGain: 1.572266,1.000000,1.000000,2.105469,
    fShutter: 125000000.000000,
    fSensorGain: 3006.000000,
    fIso: 3200.000000,
    fISPGain: 1.000000,
    i32LuxIndex: 491,
    """
    with open(metadata_filename, 'r') as f:
        lines = f.readlines()

    for key, value in new_metadata_dict.items():
        for i, line in enumerate(lines):
            if line[:4] == "fIso" == key:
                lines[i] = "fIso: {:.6f},\n".format(value)
            elif line[:9] == "bayerType" == key:
                lines[i] = "bayerType: {},\n".format(value)
            elif line[:8] == "fShutter" == key:
                lines[i] = "fShutter: {:.6f},\n".format(value)
            elif line[:13] == "strSensorName" == key:
                lines[i] = "strSensorName: {},\n".format(value)
            elif line[:14] == "strProductName" == key:
                lines[i] = "strProductName: {},\n".format(value)
            elif line[:8] == "fISPGain" == key:
                lines[i] = "fISPGain: {:.6f},\n".format(value)
            elif line[:7] == "fWbGain" == key:
                lines[i] = "fWbGain: {:.6f}, {:.6f}, {:.6f}, {:.6f},\n".format(value[0], value[1], value[2], value[3])
            elif line[:11] == "fSensorGain" == key:
                lines[i] = "fSensorGain: {:.6f},\n".format(value)
            elif line[:3] == "fEV" == key:
                lines[i] = "fEV: {:.6f},\n".format(value)
            elif line[:5] == "i32EV" == key:
                lines[i] = "i32EV: {},\n".format(value)
            elif line[:14] == "i32DeviceState" == key:
                lines[i] = "i32DeviceState: {},\n".format(value)
            elif line[:11] == "flashStatus" == key:
                lines[i] = "flashStatus: {},\n".format(value)
            elif line[:9] == "fUserZoom" == key:
                lines[i] = "fUserZoom: {:.6f},\n".format(value)
            elif line[:11] == "i32LuxIndex" == key:
                lines[i] = "i32LuxIndex: {},\n".format(value)
            elif line[:14] == "i32CameraState" == key:
                lines[i] = "i32CameraState: {},\n".format(value)
            # elif line.find('fLensDataGain') >= 0:
            #     """
            #     [Lucy:]
            #     Lens shading correction (LSC) map. Values greater than 1 will make the MFP brighten the inputs.
            #     For simplicity, I replaced them with 1s so that MFP doesn’t do LSC.
            #     """
            #     lsc_str = line.split(':')[1]
            #     lsc_arr = np.array(lsc_str.split(',')[:-1]).astype(np.float32)  # strip the newline character
            #     lsc_strs = ['{:.6f},'.format(1.0) for i in range(lsc_arr.shape[0])]
            #     lsc_str_new = ''.join(lsc_strs)
            #     lines[i] = f"fLensDataGain: {lsc_str_new}\n"

    with open(save_to_filename, 'w', newline='') as f:
        f.writelines(lines)


def update_and_save_input_metadata_DEPRECATED(metadata_filename, save_to_filename, new_iso, new_exposure):
    """
    Read MFP input metadata; update ISO and exposure; save into another file.
    :param metadata_filename:
    :param save_to_filename:
    :param new_iso:
    :param new_exposure:
    :return:
    """
    with open(metadata_filename, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line[:4] == "fIso":
            lines[i] = "fIso: {:.6f},\n".format(new_iso)
        elif line[:8] == "fShutter":
            lines[i] = "fShutter: {:.6f},\n".format(new_exposure)
    with open(save_to_filename, 'w', newline='') as f:
        f.writelines(lines)


def apply_lens_gain(raw_image, lens_gain, bayer_pattern, operation='multiply'):
    """Multiple or divide, according to mode, a raw image by lens gain (shading correction) values."""
    result_image = raw_image.copy()
    upper_left_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    bayer_pattern_idx = np.array(bayer_pattern)
    # blue channel index --> 3
    bayer_pattern_idx[bayer_pattern_idx == 2] = 3
    # second green channel index --> 2
    if bayer_pattern_idx[3] == 1:
        bayer_pattern_idx[3] = 2
    else:
        bayer_pattern_idx[2] = 2
    for c in range(4):
        i0 = upper_left_idx[c][0]
        j0 = upper_left_idx[c][1]
        if operation == 'multiply':
            result_image[i0::2, j0::2] *= lens_gain[:, :, bayer_pattern_idx[c]]
        else:
            result_image[i0::2, j0::2] /= lens_gain[:, :, bayer_pattern_idx[c]]
    return result_image
