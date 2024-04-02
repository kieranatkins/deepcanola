# load image and assosiated mask
# clean-up mask (threshold and potentially diffuse and erode)
# place pods in image with assosiated masks
# ...random rotation
# ...random placement (guassian perhaps - limit overlap possibly, see papers)
# ...random scale (guassian)
# ...location jitter (not just central)
# verify dataset

import random
import numpy as np
import cv2
from pathlib import Path
import argparse
import pycocotools.mask as coco
import datetime
import json
from PIL import Image, ImageFilter
from scipy.stats import truncnorm
from pqdm.processes import pqdm
from itertools import starmap
import random
from skimage import exposure, io, color, filters, morphology

MAX_BLUR_ITER = 3

info = {
    'description': 'Synethtic dataset',
    'version': '0.1',
    'year': 2023,
    'contributor': 'Larissa Van Vliet, Kieran Atkins',
    'date_created': datetime.datetime.utcnow().isoformat(' ')
}
licenses = [{
    'url': '',
    'id': 1,
    'name': ''
}]
categories = [{
    'id': 1,
    'name': 'pod',
}]
images = []
annotations = []


def create_image(image_id, background, objects_subset, result_dir, background_recoloring=False):
    # image = Image.open(background)
    image = background.convert("RGB")

    if background_recoloring:
        image = exposure.adjust_gamma(np.array(image), np.random.uniform(0.3, 1.2))
        image = Image.fromarray(image)

    image_name = f'{image_id}.png'
    image_entry = dict(
        license=1,
        file_name=image_name,
        id=int(image_id),
        height=int(image.size[1]),
        width=int(image.size[0])
    )

    depth_mask = np.zeros((image.size))
    object_stack = []

    for obj_image_path, obj_mask_path in objects_subset:
        obj_image = cv2.imread(str(obj_image_path))
        obj_mask = cv2.imread(str(obj_mask_path))

        # Obtain binary mask from image
        _, obj_mask, _ = cv2.split(obj_mask)
        _, obj_mask = cv2.threshold(obj_mask, 0, 255, cv2.THRESH_OTSU)

        # Convert to PIL Image
        obj_mask = Image.fromarray(obj_mask)

        # Get object display mask
        rgb = obj_image[:,:,[2,1,0]]        #bgr 2 rgb
        v = color.rgb2hsv(rgb)[:,:,2]    #value only
        # obj_display_mask = filters.unsharp_mask(obj_display_mask, radius=3, amount=2.0)
        yen = filters.threshold_yen(v)
        obj_display_mask = v < yen
        obj_display_mask = morphology.binary_dilation(obj_display_mask)
        obj_display_mask = morphology.remove_small_holes(obj_display_mask)
        obj_display_mask = morphology.remove_small_objects(obj_display_mask, min_size=128)

        # obj_image[] = [255,255,255]
        obj_image = Image.fromarray(cv2.cvtColor(obj_image, cv2.COLOR_BGR2RGBA))
        obj_image.putalpha(Image.fromarray(~obj_display_mask))

        # obj_image = obj_image.filter(ImageFilter.GaussianBlur(radius=1.0))

        h, w = obj_mask.size

        # Create random scaling size and angle, then resize and rotate
        scale = max(0.1, np.random.normal(1.0, scale=0.3))
        scale_x = max(0.1, np.random.normal(scale, scale=0.1))
        # scale_x = max(0.1, scale_x) #lower bound
        scale_y = max(0.1, np.random.normal(scale, scale=0.1))
        # scale_y = max(0.1, scale_y) #lower bound

        theta = np.random.uniform(0.0, 360.0)

        obj_image = obj_image.resize((round(h * scale_x), round(w * scale_y)))
        obj_mask = obj_mask.resize((round(h * scale_x), round(w * scale_y)))

        obj_image = obj_image.rotate(theta, expand=True, resample=Image.BICUBIC)
        obj_mask = obj_mask.rotate(theta, expand=True)

        # Create random scale with a truncated normal, so as not to have negative
        # value for position and ensure pos always fits within image
        lower, upper = 0, max(1, image.size[0] - obj_mask.size[0])
        mu, sigma = image.size[0] / 2, image.size[0]
        X = truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        x = round(X.rvs())

        lower, upper = 0, max(1, image.size[1] - obj_mask.size[1])
        mu, sigma = image.size[1] / 2, image.size[1]
        Y = truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        y = round(Y.rvs())
        
        # Diffuse and erode mask to cleanup edges
        obj_mask = np.array(obj_mask)

        # inverse gaussian for sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        obj_mask = cv2.filter2D(obj_mask, -1, kernel)
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, np.ones((7, 7)))
        obj_mask = cv2.erode(obj_mask, np.ones((5, 5)))

        obj_mask = Image.fromarray(obj_mask)

        mask = Image.new('1', image.size)
        mask.paste(obj_mask, mask=obj_mask, box=(x, y))
        mask = np.array(mask)

        local_mask = depth_mask.copy()

        # Only take the values that interact with the pod we're putting down
        local_mask[mask.T == False] = 0

        entry = dict(obj_image=obj_image, obj_mask=obj_mask, mask=mask, coords=(x, y), level=int(np.max(local_mask)))
        object_stack.append(entry)

        depth_mask[mask.T] += 1

    object_stack = sorted(object_stack, key=lambda x: x['level'])
    annots = []

    # cv2.imwrite(f'/home/kieran/Outputs/depth_of_field_{image_id}.png', cv2.applyColorMap((depth_mask.astype(np.uint8)*5)**2, cv2.COLORMAP_JET))

    for entry in reversed(object_stack):

        obj_image = entry['obj_image']  # Image of only the pod
        obj_mask = entry['obj_mask']    # 2D array of only the pod's mask - unused
        mask = entry['mask']            # 2D array of only the pod's mask, within the whole image
        coords = entry['coords']        # Coordinates of the pod position
        level = entry['level']          # Level of depth (i.e. how many pods are on top of it) - unused

        # Place object within image using mask
        disc_kernel = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0]
            ]
        ) / (13.0)

        level *= 2 # double effect
        for l in range(level ** 2):
            # for i in range(random.randint(1, MAX_BLUR_ITER)):
            obj_image = obj_image.filter(ImageFilter.Kernel(size=disc_kernel.shape, kernel=disc_kernel.flatten()))

        image.paste(obj_image, coords, obj_image)

        # Encode mask as COCO RLE in order to be saved in COCO format (standard format for associated mask data)
        segmentation = coco.encode(np.asfortranarray(mask, dtype=np.uint8))
        segmentation['counts'] = segmentation['counts'].decode('utf-8')
        annots.append(dict(
            segmentation=segmentation,
            area=float(coco.area(segmentation)),
            bbox=[int(x) for x in coco.toBbox(segmentation).tolist()],
            id=None,
            category_id=1,
            image_id=image_id,
            iscrowd=False
        ))

    image.convert('RGB')
    image.save(result_dir / image_name)

    return image_entry, annots


def main(image_dir, mask_dir, bg_path, num_images=2000, avg_objs_per_image=20):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    bg_path = Path(bg_path)
    all_masks = [f for f in mask_dir.glob('**/*') if f.is_file()]

    # Create object (image and mask) pool
    obj_images = [f for f in image_dir.glob('**/*') if f.is_file()]
    print(len(obj_images))

    def get_mask(path):
        #return mask_dir / (path.name[:20] + 'mask_' + path.name[20:])
        for mask in all_masks:
            if path.name == mask.name.replace('_mask', ''): 
                return mask
        return None

    objects = [(title, get_mask(title)) for title in obj_images]
    objects = [(title, mask) for title, mask in objects if mask is not None]

    # Verify mask & image combo exist
    objects_verified = []
    for img, mask in objects:
        if img.exists() and mask.exists():
            objects_verified.append((img, mask))
    objects = objects_verified

    # Create background pool
    backgrounds = [Image.open(f) for f in bg_path.glob('**/*') if f.is_file()]

    # Create directory to store images
    result_dir = image_dir.parent / 'synthetic_images'
    result_dir.mkdir(exist_ok=True)

    # Prepare parameters for parallelisation
    ids = list(range(num_images))
    backgrounds_set = random.choices(backgrounds, k=num_images)
    num_objects_set = [round(np.random.normal(avg_objs_per_image, scale=avg_objs_per_image / 5)) for _ in
                       range(num_images)]
    objects_subset_set = [random.choices(objects, k=num_objects_set[x]) for x in range(num_images)]
    result_dirs = [result_dir] * num_images
    background_recoloring = [True] * num_images
    args = zip(ids, backgrounds_set, objects_subset_set, result_dirs, background_recoloring)

    # Run image creation function on N cores and collect results
    res = pqdm(args, create_image, n_jobs=14, argument_type='args')
    # res = starmap(create_image, args)
    images, annotations = zip(*res)

    # Flatten annotation list
    annotations = [item for sublist in annotations for item in sublist]

    # Global dataset annotation ID (Required by COCO dataset format)
    for i, annot in enumerate(annotations):
        annot['id'] = i

    ds = dict(info=info, licenses=licenses, categories=categories, images=images, annotations=annotations)
    with open(result_dir / 'ds.json', 'w') as f:
        json.dump(ds, f)


if __name__ == '__main__':
    print('Started.')
    parser = argparse.ArgumentParser(
        description='Process simple images and assosiated masks into semi-synthetic dataset')
    parser.add_argument('image_dir', type=str, nargs=1, metavar='IMAGE_PATH')
    parser.add_argument('mask_dir', type=str, nargs=1, metavar='MASK_PATH')
    parser.add_argument('bg_dir', type=str, nargs=1, metavar='BG_PATH')
    parser.add_argument('--num-images', type=int, nargs=1, default=[1000], metavar='NUM_IMAGES')
    parser.add_argument('--num-per-image', type=int, nargs=1, default=[45], metavar='NUM_PER_IMAGE')

    args = parser.parse_args()
    main(args.image_dir[0],
         args.mask_dir[0],
         args.bg_dir[0],
         num_images=args.num_images[0],
         avg_objs_per_image=args.num_per_image[0])
