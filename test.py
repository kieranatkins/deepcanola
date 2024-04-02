from argparse import ArgumentParser
import logging
from pathlib import Path
import torch
from utils.references.coco_utils import CocoDetection
from torch.utils.data.dataloader import DataLoader

import utils.references.utils as utils
import utils.references.transforms as T
from utils.references.engine import inference



from model import get_model


logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("Brassica")



def get_transforms(train=True):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)


def main(image_dir, annot_test, out, weights=None, batch_size=1, num_classes=2):
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dev = torch.device('cuda')

    # Prepare dataset saved in the COCO Format
    dataset_test = CocoDetection(image_dir, annot_test, transforms=get_transforms(train=False))
        
    # Prepare dataloaders
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 collate_fn=utils.collate_fn)
    
    
    model = get_model(num_classes)
    # Either load previous training or start training from fresh
    if weights is not None:
        weights = torch.load(weights, map_location='cpu')
        try:
            model.load_state_dict(weights)
        except TypeError:
            model.load_state_dict(weights.state_dict())
    model.to(dev)
    # evaluate(model, dataloader_test, device=dev)
    out_path = Path(out)
    out_path.mkdir(exist_ok=True)
    inference(model, dataloader_test, device=dev, output_path=out_path)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image_dir', type=str, nargs=1, metavar='IMAGE PATH')
    parser.add_argument('test_annot', type=str, nargs=1, metavar='TEST ANNOTATION FILE')
    parser.add_argument('out_dir', type=str, nargs=1, metavar='OUT PATH')
    parser.add_argument('weights', type=str, default=[''], nargs=1, metavar="SAVED MODEL PATH")
    parser.add_argument('--batch-size', type=int, default=[1], nargs=1, metavar="BATCH SIZE")
    args = parser.parse_args()

    main(args.image_dir[0],
         args.test_annot[0],
         args.out_dir[0],
         args.weights[0],
         batch_size=args.batch_size[0]
     )
