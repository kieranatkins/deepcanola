import math
import sys
import time

import skimage.io
import torch
import torchvision.models.detection.mask_rcnn
import torchvision
from . import utils
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
import pycocotools.mask as pct
from collections import defaultdict
import analysis
import logging

from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from statistics import mean

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        del images
        del targets
        
    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def check_errors(name):
    if name[:6] == "BR017-":
        new_name = name
    elif name[:5] == "B017-":
        new_name = "BR017-" + name[5:]
    elif name[5] != "-":
        new_name = "BR017-" + name[5:]
    
    return new_name

@torch.no_grad()
def inference(model, data_loader, device, output_path):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    output_path = Path(output_path)
    scores = []
    thresh = [0, 0.25, 0.5, 0.75]
    score_thresh = 0.5

    data = defaultdict(list)

    for i, (images, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = list(img.to(device) for img in images)
        # logging.info(i)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        images = list(img.to(cpu_device) for img in images)
        
        for out in outputs:
            scores.extend([x for x in out['scores'].tolist()])

        for image, target, out in zip(images, targets, outputs):
            name = data_loader.dataset.coco.loadImgs(target['image_id'])[0]["file_name"] 
            image = torchvision.transforms.ConvertImageDtype(torch.uint8)(image)

            args = torch.argwhere(out['scores'] > 0.5).flatten()
            s = out['scores'][args]
            l = out['labels'][args]
            b = out['boxes'][args]
            m = out['masks'][args]
                
            image = torchvision.utils.draw_bounding_boxes(image, b, [f'{l} - {s:.4f}' for s,l in zip(s.tolist(), l.tolist())])
            image = torchvision.utils.draw_segmentation_masks(image, m.squeeze(1) > 0.5, alpha=0.5)

            image = torchvision.transforms.ToPILImage()(image)
            image.save(output_path / f'{Path(name).stem}.png')

            name = data_loader.dataset.coco.loadImgs(target['image_id'])[0]["file_name"]
            for label, mask, score, box in zip(out['labels'], out['masks'], out['scores'], out['boxes']):
                if score > score_thresh:
                    data['image_id'].append(name)
                    # mask = torch.nn.functional.interpolate(mask.unsqueeze(0), (4200, 2511))[0]
                    (length, perimeter, area), multiple_masks = analysis.mask_analysis(
                        np.array(mask[0] > 0.5, dtype=np.uint8) * 255,
                        1.0, 0, multiple_masks=True)
                    mask = np.array(mask[0] > 0.5, dtype=np.uint8)
                    data['length'].append(length)
                    data['perimeter'].append(perimeter)
                    data['area'].append(area)
                    data['multiple_masks'].append(multiple_masks)
                    data['confidence_score'].append(score.item())
                    data['box'].append(box.tolist())
                    mask_dict = pct.encode(np.asfortranarray(mask))
                    mask_dict['counts'] = mask_dict['counts'].decode('utf-8')
                    data['mask'].append(mask_dict)
    
    for t in thresh:
        score_subset = [x for x in scores if x > t]
        logging.info(f'Thresh: {t} = Confidence:{mean(score_subset) if len(score_subset) > 0 else math.nan}')
    
    data = pd.DataFrame(data)
    data['image_id'] = data['image_id'].apply(check_errors)
    data['image_id'] = data['image_id'].apply(lambda x: Path(x).stem)
    data['treatment'] = data['image_id'].apply(lambda x: int(x.split('-')[1][4]))
    data['genotype'] = data['image_id'].apply(lambda x: x.split('-')[1][:3])
    data.to_csv(output_path / 'data_out.csv')


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    data = defaultdict(list)

    scores = []
    n = 0
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        thresh=0.5

        for out in outputs:
            scores.extend([x for x in out['scores'].tolist() if x > thresh])
            n += 1
        ###### CUSTOM #######

        # #####################         
        res = {torch.tensor(target["image_id"]).item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # pd.DataFrame(data).to_csv('./data_out.csv')
    #exit()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    
    # print(f'Average confidence score = {mean(scores)}')
    print(f'avg conf score = {sum(scores) / len(scores) if len(scores) > 0 else 0} of {n} images')
    import json
    with open('./scores.json', 'w') as f:
        json.dump(scores, f)

    return coco_evaluator
