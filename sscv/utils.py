import cv2
import numpy as np 
import os
import warnings
import operator


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        imageBGR = cv2.imdecode(n, flags)
        return cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(e)
        return None


def imwrite(filename, imageRGB, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        imageBGR = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
        result, n = cv2.imencode(ext, imageBGR, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
                return True
        else:
                return False

    except Exception as e:
        print(e)
        return False
        
def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    
    """
    Code Reference: 
    https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def comparison_operator(img, thr, ind='>'):
    if ind == '==':
        return operator.eq(img, thr)
    elif ind == '<':
        return operator.lt(img, thr)
    elif ind == '>':
        return operator.gt(img, thr)
    elif ind == '!=':
        return operator.ne(img, thr)

# def apply_ColorToImage(img, mask, color_mask,
#                        img_ratio = 0.3, color_ratio = 0.7
#                        ) :

#     """
#     Apply color to where mask value equals 1

#     Arg:
#         img (numpy.uint8) : original image
#         mask (numpy.bool) : mask to apply color
#         color_mask(numpy.uint8) : color to be applied to mask
#         img_ratio(float) : Brightness ratio to be kept from original image
#         color_ratio(float) : Brightness ratio to be kept from color

#     Returns:
#         colored_img(numpy.uint8) : image

#     """

#     if (mask.dtype != 'bool') :
#         warnings.warn("The input variable 'mask' is not an numpy.bool array")
#         warnings.warn("Variable mask is converted to numpy.bool array")
#         mask = mask.astype(np.bool)

#     img[mask, :] = img[mask, :] * img_ratio + color_mask * color_ratio
#     return

def inference_detector_sliding_window(model, input_img, color_mask,
                                      score_thr = 0.1, window_size = 1024, overlap_ratio = 0.5,):
    from tqdm import tqdm
    from mmdet.apis import inference_detector

    import mmcv

    import pycocotools.mask as maskUtils
    import slidingwindow as sw


    '''
    :param model: is a mmdetection model object
    :param input_img : str or numpy array
                    if str, run imread from input_img
    :param score_thr: is float number between 0 and 1.
                   Bounding boxes with a confidence higher than score_thr will be displayed,
                   in 'img_result' and 'mask_output'.
    :param window_size: is a subset size to be detected at a time.
                        default = 1024, integer number
    :param overlap_ratio: is a overlap size.
                        If you overlap sliding windows by 50%, overlap_ratio is 0.5.

    :return: img_result
    :return: mask_output

    '''

    # color mask has to be updated for multiple-class object detection
    if isinstance(input_img, str) :
        img = imread(input_img)
    else :
        img = input_img

    # Generate the set of windows, with a 256-pixel max window size and 50% overlap
    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, window_size, overlap_ratio)
    mask_output = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)


    for window in tqdm(windows, ascii = True, desc = 'inference by sliding window on ' + os.path.basename(input_img)):
        # Add print option for sliding window detection
        img_subset = img[window.indices()]
        results = inference_detector(model, img_subset)
        bbox_result, segm_result = results
        mask_sum = np.zeros((img_subset.shape[0], img_subset.shape[1]), dtype=np.bool)
        bboxes = np.vstack(bbox_result) # bboxes

        # draw segmentation masks
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]

            for i in inds:
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                mask_sum = mask_sum + mask

        mask_output[window.indices()] = mask_sum

    mask_output = mask_output.astype(np.uint8)
    mask_output[mask_output > 1] = 1

    mask_output_bool = mask_output.astype(np.bool)

    # Add colors to detection result on img
    img_result = img
    img_result[mask_output_bool, :] = img_result[mask_output_bool,:] * 0.3 + color_mask * 0.6

    return img_result, mask_output



def connect_cracks(mask_output, epsilon = 200):
    '''
    :param mask_output: a numpy uint8 variable
    :param epsilon: distance between cracks to be connected
    :return: connect_mask_output : crack-connection result
    '''



    '''
    To-dos : 
    1 . Add iteration option
    2 . Add connection option considering a direction of a crack
    with the direction of ellipse of each crack 
    '''

    # label each crack
    labels, num = label(mask_output, connectivity=2, return_num=True)
    # get information of each crack area
    crack_region_table = regionprops_table(labels, properties=('label', 'bbox', 'coords', 'orientation'))

    width = crack_region_table['bbox-3'] - crack_region_table['bbox-1']
    height = crack_region_table['bbox-2'] - crack_region_table['bbox-0']

    crack_region_table['is_horizontal'] = width > height

    connecting_directions = ['x_axis', 'y_axis']
    connect_line_img = np.zeros_like(mask_output, dtype=np.uint8)

    for connecting_direction in connecting_directions:

        e2_list = []
        e1_list = []

        for crack_num, crack_region in enumerate(crack_region_table['label']):

            min_row = crack_region_table['bbox-0'][crack_num]
            min_col = crack_region_table['bbox-1'][crack_num]
            max_row = crack_region_table['bbox-2'][crack_num] - 1
            max_col = crack_region_table['bbox-3'][crack_num] - 1

            if crack_region_table['is_horizontal'][crack_num]:
                # max col / min col
                col = crack_region_table['coords'][crack_num][:, 1]

                e2 = crack_region_table['coords'][crack_num][np.argwhere(col == max_col), :][-1][0]
                e1 = crack_region_table['coords'][crack_num][np.argwhere(col == min_col), :][0][0]

                if connecting_direction == 'y_axis' and e2[0] < e1[0]:
                    e2, e1 = e1, e2

                e2_list.append(e2)
                e1_list.append(e1)

            else:
                # max row / min row
                row = crack_region_table['coords'][crack_num][:, 0]

                e2 = crack_region_table['coords'][crack_num][np.argwhere(row == max_row), :][-1][0]
                e1 = crack_region_table['coords'][crack_num][np.argwhere(row == min_row), :][0][0]

                if connecting_direction == 'x_axis' and e2[1] < e1[1]:
                    e2, e1 = e1, e2

                e2_list.append(e2)
                e1_list.append(e1)

        crack_region_table['e2'] = e2_list
        crack_region_table['e1'] = e1_list


        n = len(crack_region_table['label'])
        color = (1)  # binary image


        for num_e2, e2 in enumerate(crack_region_table['e2']):

            connect_candidates_e2 = []
            connect_candidates_e1 = []
            distance_list = []

            for num_e1, e1 in enumerate(crack_region_table['e1']):

                if num_e2 != num_e1:
                    d = np.subtract(e1, e2)
                    distance = np.sqrt(d[0] ** 2 + d[1] ** 2)


                    vector_1 = np.asarray(crack_region_table['e1'][num_e2] - e2, dtype=np.float64)
                    vector_2 = np.asarray(e2 - e1, dtype=np.float64)

                    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                    dot_product = np.dot(unit_vector_1, unit_vector_2)
                    angle_1 = np.arccos(dot_product)


                    vector_1 = np.asarray(crack_region_table['e1'][num_e2] - e2, dtype=np.float64)
                    vector_2 = np.asarray(e1 - crack_region_table['e2'][num_e1], dtype=np.float64)

                    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                    dot_product = np.dot(unit_vector_1, unit_vector_2)
                    angle_2 = np.arccos(dot_product)


                    if (distance < epsilon) and (angle_1 < 1.5 / 2) and (angle_2 < 1.5 / 2):
                        distance_list.append(distance)
                        connect_candidates_e2.append(tuple(e2[::-1]))
                        connect_candidates_e1.append(tuple(e1[::-1]))

            if distance_list :
                connect_idx = np.argmin(distance_list)
                connect_e2 = connect_candidates_e2[connect_idx]
                connect_e1 = connect_candidates_e1[connect_idx]
                connect_line_img = cv2.line(connect_line_img, connect_e2, connect_e1, color, 8)

    mask_output = mask_output + connect_line_img
    mask_output[mask_output > 1] = 1

    return mask_output

def remove_cracks(mask_output, threshold = 300):
    '''
    :param mask_output: a numpy uint8 variable
    :param threshold: cracks of which length is under thershold will be removed.
    :return: mask_output : crack mask after thresholding
    '''

    labels, num = label(mask_output, connectivity=2, return_num=True)
    crack_region_table = regionprops_table(labels, properties=('label', 'bbox', 'coords'))

    width = crack_region_table['bbox-3'] - crack_region_table['bbox-1']
    height = crack_region_table['bbox-2'] - crack_region_table['bbox-0']
    crack_region_table['diagonal_length'] = np.sqrt(height**2 + width**2)

    for crack_num in range(len(crack_region_table['label'])):
        if crack_region_table['diagonal_length'][crack_num] < threshold :
            for c in crack_region_table['coords'][crack_num]:
                mask_output[c[0], c[1]] = 0

    return mask_output

