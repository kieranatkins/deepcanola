import cv2
import numpy as np
from fil_finder import FilFinder2D
import astropy.units as units
import warnings
from scipy import ndimage
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)


# Takes the output masks from the network and returns a list with position of each element
# corresponding to each instances length, perimeter, area and pseudo width.
# mask :: numpy array of type uint8
# length_contours :: Bool - return the contour object for display
def mask_analysis(mask, scale, id, pixel_vals=False, multiple_masks=False, length_contours=False, calc_perim=True):
    # Find coordinates of mask bounding box
    coords = np.argwhere(mask)

    # if there is no mask, very small mask (< 64 pixels)
    if coords.size < 64:
        # print('Empty/Unsuitable mask found, skipping...')
        length_px, perimeter_px, area_px, width_px = 0.0, 0.0, 0.0, 0.0
        m_masks = False
    else:
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        cropped = mask[x_min:x_max + 1, y_min:y_max + 1]

        mask = cropped * 255

        # Find contours of standard mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Pod perimeter & mask count
        m_masks = True if len(contours) > 1 else False

        if calc_perim:
            perimeter_px = max([cv2.arcLength(c, True) for c in contours], default=0.0)
        else:
            perimeter_px = 0.0

        # Pod area
        area_px = sum([cv2.contourArea(c) for c in contours])

        # Pod length
        fil = FilFinder2D(mask, mask=cropped, beamwidth=units.pix * 0)
        fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(skel_thresh=units.pix * 10, branch_thresh=units.pix * 10,
                              prune_criteria='length')  # 10px ~1mm min length
        length_px = max(fil.lengths().value)

    area_mm2 = float(area_px / (scale ** 2))
    perimeter_mm = float(perimeter_px / scale)
    length_mm = float(length_px / scale)

    ret = [[length_mm, perimeter_mm, area_mm2]]
    if pixel_vals:
        ret.append([length_px, perimeter_px, area_px])
    if multiple_masks:
        ret.append(m_masks)
    if length_contours:
        ret.append(contours)
    # print(ret, flush=True)
    return ret


# Mask analysis helper function to facilitate multiprocessing
def _ma(mask, scale):
    mask = mask.to('cpu').numpy() * 255
    mask = mask.astype(np.uint8)

    measurement, multiple_masks = mask_analysis(mask, scale, multiple_masks=True)
    measurement.append(multiple_masks)

    return measurement
