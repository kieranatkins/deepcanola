from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data.dataloader import DataLoader

from utils.references.coco_utils import CocoDetection, ConvertCocoPolysToMask
import utils.references.utils as utils
import utils.references.transforms as T
from utils.references.engine import train_one_epoch, evaluate
from model import get_model


def get_transforms(train=True):
    transforms = []
    transforms.append(ConvertCocoPolysToMask())
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)
    

def main(train_image_dir, annot_train, test_image_dir, annot_test, out, load_dir=None, batch_size=2, schedule=1):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    out = Path(out)
    out.mkdir(exist_ok=True)

    # Prepare dataset saved in the COCO Format
    dataset_train = CocoDetection(train_image_dir, annot_train, transforms=get_transforms(train=True))
    dataset_test = CocoDetection(test_image_dir, annot_test, transforms=get_transforms(train=False))
    num_classes = 2  # 1 class +1 for background

    # Prepare dataloaders
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=utils.collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=utils.collate_fn)

    model = get_model(num_classes)
    # Either load previous training or start training from fresh
    if load_dir != '':
        weights = torch.load(load_dir, map_location=device)
        try:
            model.load_state_dict(weights)
        except TypeError:
            model.load_state_dict(weights.state_dict())
        
    model = model.to(device)    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training schedules defind in: https://arxiv.org/abs/1811.08883
    milestones = [x * schedule for x in [8, 11]]
    num_epochs = 12 * schedule
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Train
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, dataloader_train, device, epoch, print_freq=1)
        lr_scheduler.step()
        evaluate(model, dataloader_test, device=device)
    
    torch.save(model.state_dict(), out / 'model.pth')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('train_image_dir', type=str, nargs=1, metavar='TRAIN IMAGE PATH')
    parser.add_argument('train_annot', type=str, nargs=1, metavar='TRAIN ANNOTATION FILE')
    parser.add_argument('test_image_dir', type=str, nargs=1, metavar='TEST IMAGE PATH')
    parser.add_argument('test_annot', type=str, nargs=1, metavar='TEST ANNOTATION FILE')
    parser.add_argument('out_dir', type=str, nargs=1, metavar='OUT PATH')
    parser.add_argument('--load-file', type=str, default=[''], nargs=1, metavar="SAVED MODEL PATH" )
    parser.add_argument('--batch-size', type=int, default=[1], nargs=1, metavar="BATCH SIZE")
    parser.add_argument('--schedule', type=int, default=[1], nargs=1, metavar="TRAIN SCHEDULE" )
    args = parser.parse_args()
    
    main(args.train_image_dir[0],
         args.train_annot[0],
         args.test_image_dir[0],
         args.test_annot[0],
         args.out_dir[0],
         load_dir=args.load_file[0],
         batch_size=args.batch_size[0],
         schedule=args.schedule[0])
