import cv2
import numpy as np 
import os
import warnings


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


def apply_ColorToImage(img, mask, color_mask,
                       img_ratio = 0.3, color_ratio = 0.7
                       ) :

    """
    Apply color to where mask value equals 1

    Arg:
        img (numpy.uint8) : original image
        mask (numpy.bool) : mask to apply color
        color_mask(numpy.uint8) : color to be applied to mask
        img_ratio(float) : Brightness ratio to be kept from original image
        color_ratio(float) : Brightness ratio to be kept from color

    Returns:
        colored_img(numpy.uint8) : image

    """

    if (mask.dtype != 'np.bool') :
        warnings.warn("The input variable 'mask' is not an numpy.bool array")
        warnings.warn("Variable mask is converted to numpy.bool array")
        mask = mask.astype(np.bool)

    img[mask, :] = img[mask, :] * img_ratio + color_mask * color_ratio
    return

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


