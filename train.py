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
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import joint_transforms
from config import cod_training_root
from config import cod_val_root
from config import backbone_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from PFNet import PFNet

import loss

cudnn.benchmark = True

torch.manual_seed(2021)
# device_ids = [1]
device_ids = [0]

ckpt_path = './ckpt'
exp_name = 'PFNet'

args = {
    'epoch_num': 1000,
    'train_batch_size': 8,
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
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
# img_transform = transforms.Compose([
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# img_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# target_transform = transforms.ToTensor()

# Prepare Data Set.
train_set = ImageFolder(cod_training_root)
# train_set = ImageFolder(cod_training_root)
val_set   = ImageFolder(cod_val_root)

print("Train set size:", len(train_set))
print("Val set size:", len(val_set))

img, mask = train_set[0]

print("\n==== FINAL SANITY ====")
print("IMG:", img.shape, img.min().item(), img.max().item())
print("MASK:", mask.shape, torch.unique(mask))
print("=====================\n")

# print("Train set: {}".format(train_set.__len__()))
# print("Val set: {}".format(val_set.__len__()))

train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args['train_batch_size'], shuffle=False, num_workers=4)

# ===== DEBUG SHAPE CHECK =====
sample_img, sample_mask = train_set[0]

print("\n===== DATA SANITY CHECK =====")
print("Image shape :", sample_img.shape)
print("Mask shape  :", sample_mask.shape)
print("Image min/max :", sample_img.min(), sample_img.max())
print("Mask unique :", torch.unique(sample_mask))
print("============================\n")

x, y = next(iter(train_loader))
print("Batch image shape :", x.shape)
print("Batch mask shape  :", y.shape)
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

def dice_metric(pred, mask):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum()

    dice = (2 * inter + 1e-7) / (union + 1e-7)

    return dice.item()

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
    print("\n===== MODEL INFO =====")
    print("Model first layer:", net.module.layer0[0].weight.shape)
    print("======================\n")
    
    print("Using {} GPU(s) to Train.".format(len(device_ids)))

    open(log_path, 'w').write(str(args) + '\n\n')

    # best_dice = 0
    # best_epoch = 0
    # early_stop_patience = 50
    # train(net, optimizer)
    best_dice = 0
    best_epoch = 0
    early_stop_patience = 50

    train(net, optimizer, best_dice, best_epoch, early_stop_patience)
    writer.close()

# def train(net, optimizer):
def train(net, optimizer, best_dice, best_epoch, early_stop_patience):
    curr_iter = 1
    start_time = time.time()
    early_stop_count = 0  

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

            # predict_1, predict_2, predict_3, predict_4 = net(inputs)
            predict_4, predict_3, predict_2, predict_1 = net(inputs)

            if epoch == 1 and curr_iter == 1:
                print("\n===== MODEL OUTPUT CHECK =====")
                print("predict_1:", predict_1.shape)
                print("predict_2:", predict_2.shape)
                print("predict_3:", predict_3.shape)
                print("predict_4:", predict_4.shape)
                print("==============================\n")

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

                # ================= VALIDATION =================
        net.eval()

        val_dice_epoch = 0
        TP = TN = FP = FN = 0

        with torch.no_grad():
            for val_data in val_loader:

                val_inputs, val_labels = val_data
                val_inputs = val_inputs.cuda(device_ids[0])
                val_labels = val_labels.cuda(device_ids[0])

                p4, p3, p2, p1 = net(val_inputs)

                dice = dice_metric(p1, val_labels)
                val_dice_epoch += dice

                prob = torch.sigmoid(p1)
                pred = (prob > 0.5).float()

                TP += ((pred==1) & (val_labels==1)).sum().item()
                TN += ((pred==0) & (val_labels==0)).sum().item()
                FP += ((pred==1) & (val_labels==0)).sum().item()
                FN += ((pred==0) & (val_labels==1)).sum().item()

        val_dice_epoch /= len(val_loader)

        sensitivity = TP/(TP+FN+1e-7)
        specificity = TN/(TN+FP+1e-7)
        accuracy = (TP+TN)/(TP+TN+FP+FN+1e-7)
        precision = TP/(TP+FP+1e-7)
        recall = sensitivity
        f1 = 2*precision*recall/(precision+recall+1e-7)
        iou = TP/(TP+FP+FN+1e-7)

        print(f"\nVAL Dice: {val_dice_epoch:.4f}")
        print(f"Sens:{sensitivity:.4f} Spec:{specificity:.4f} Acc:{accuracy:.4f}")
        print(f"Prec:{precision:.4f} Recall:{recall:.4f} F1:{f1:.4f} IoU:{iou:.4f}")
        
        net.train()

        if val_dice_epoch > best_dice:

            print(" BEST MODEL UPDATED")

            best_dice = val_dice_epoch
            best_epoch = epoch
            early_stop_count = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_epoch': best_epoch,
                'early_stop_counter': early_stop_count
            },
            os.path.join(ckpt_path, exp_name, 'best_model.pth'))

        else:
            early_stop_count += 1


        print("Early stop counter:", early_stop_count)

        if early_stop_count > early_stop_patience:
            print("⛔ EARLY STOPPING TRIGGERED")
            return
        
        if epoch in args['save_point']:
            net.cpu()
            # torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_epoch': best_epoch
            },
            os.path.join(ckpt_path, exp_name, f'last_epoch_{epoch}.pth'))
            net.cuda(device_ids[0])

        if epoch >= args['epoch_num']:
            net.cpu()
            # torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_epoch': best_epoch,
                'early_stop_counter': early_stop_count
            },
            os.path.join(ckpt_path, exp_name, f'final_epoch_{epoch}.pth'))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            return

if __name__ == '__main__':
    main()