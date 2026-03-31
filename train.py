"""
 @Time    : 2021/7/6 14:52
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : train.py
 @Function: Training
 
"""
import datetime
import time
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
# from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

# import joint_transforms
from config import cod_training_root
from config import backbone_path
# from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from PFNet import PFNet
from datasets import HillshadeDataset
from datasets import get_transforms
from torch.utils.data import DataLoader

import loss

cudnn.benchmark = True

torch.manual_seed(2021)
device_ids = [1]

ckpt_path = './ckpt'
exp_name = 'PFNet'

args = {
    'epoch_num': 200,
    'train_batch_size': 16,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416,
    'save_point': [],
    'poly_train': True,
    'optimizer': 'SGD',
}

print(torch.__version__)

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
# joint_transform = joint_transforms.Compose([
#     joint_transforms.RandomHorizontallyFlip(),
#     joint_transforms.Resize((args['scale'], args['scale']))
# ])
# img_transform = transforms.Compose([
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# target_transform = transforms.ToTensor()

# Prepare Data Set.
# train_set = ImageFolder(cod_training_root, joint_transform, img_transform, target_transform)

BASE = "/content/drive/MyDrive/Prashant/Forestry_data/data_new/dataset_small_bal"

SEED = 42
TRAIN_IMGS  = f"{BASE}/train/images"
TRAIN_MASKS = f"{BASE}/train/masks"
VAL_IMGS    = f"{BASE}/val/images"
VAL_MASKS   = f"{BASE}/val/masks"

train_set = HillshadeDataset(
    img_dir=TRAIN_IMGS,
    mask_dir=TRAIN_MASKS,
    transform=get_transforms("train", patch_size=256),
    road_biased=True,
    road_ratio=0.7,
    road_min_pixels=30
)
print("Train set: {}".format(train_set.__len__()))
# train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)
train_loader = DataLoader(
    train_set,
    batch_size=args['train_batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_set = HillshadeDataset(
    img_dir=VAL_IMGS,
    mask_dir=VAL_MASKS,
    transform=get_transforms("val", patch_size=256),
    road_biased=False
)

val_loader = DataLoader(
    val_set,
    batch_size=args['train_batch_size'],
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
total_epoch = args['epoch_num'] * len(train_loader)

# loss function
structure_loss = loss.structure_loss().cuda(device_ids[0])
bce_loss = nn.BCEWithLogitsLoss().cuda(device_ids[0])
iou_loss = loss.IOU().cuda(device_ids[0])

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss

def main():
    print(args)
    print(exp_name)

    net = PFNet(backbone_path).cuda(device_ids[0]).train()

    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    net = nn.DataParallel(net, device_ids=device_ids)
    print("Using {} GPU(s) to Train.".format(len(device_ids)))

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()

def compute_metrics(pred, target):
    pred = (pred > 0.5).float()

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "specificity": specificity.item(),
        "accuracy": accuracy.item(),
    }

@torch.no_grad()
def validate(net):
    net.eval()

    total_loss = 0
    all_preds, all_targets = [], []

    for inputs, labels in val_loader:
        inputs = inputs.cuda(device_ids[0])
        labels = labels.cuda(device_ids[0])

        predict_1, predict_2, predict_3, predict_4 = net(inputs)

        loss = bce_iou_loss(predict_1, labels)
        total_loss += loss.item()

        preds = torch.sigmoid(predict_1)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    metrics = compute_metrics(preds, targets)

    return total_loss / len(val_loader), metrics

def train(net, optimizer):
    curr_iter = 1
    start_time = time.time()

    best_dice = 0.0
    no_improve = 0
    PATIENCE = 50

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_1, predict_2, predict_3, predict_4 = net(inputs)

            loss_1 = bce_iou_loss(predict_1, labels)
            loss_2 = structure_loss(predict_2, labels)
            loss_3 = structure_loss(predict_3, labels)
            loss_4 = structure_loss(predict_4, labels)

            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)

            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)

            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_1_record.avg, loss_2_record.avg,
                   loss_3_record.avg, loss_4_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        val_loss, val_metrics = validate(net)
        net.train()

        print(f"\nVAL → Loss: {val_loss:.4f}")
        print(
            f"dice: {val_metrics['dice']:.4f} | "
            f"iou: {val_metrics['iou']:.4f} | "
            f"precision: {val_metrics['precision']:.4f} | "
            f"recall: {val_metrics['recall']:.4f} | "
            f"specificity: {val_metrics['specificity']:.4f} | "
            f"accuracy: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            no_improve = 0

            torch.save(net.module.state_dict(),
                    os.path.join(ckpt_path, exp_name, 'best_model.pth'))

            print("🔥 New best model saved!")

        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{PATIENCE}")

            if no_improve >= PATIENCE:
                print("\n⛔ EARLY STOPPING TRIGGERED")
                print(f"Best Dice: {best_dice:.4f}")
                return
        # if epoch in args['save_point']:
        #     net.cpu()
        #     torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
        #     net.cuda(device_ids[0])

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            return

if __name__ == '__main__':
    main()