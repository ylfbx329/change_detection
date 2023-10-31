import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from operation import train, validate
from path import *
import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from dataset import RsDataset
from utils import get_logger
from networks import USSFCNet

TITLE = 'USSFCNet_CDD'

# 创建日志写入器
writer_train = SummaryWriter('runs/' + TITLE + '/train')
writer_val = SummaryWriter('runs/' + TITLE + '/val')
writer_all = SummaryWriter('runs/' + TITLE + '/all')

# 训练设备选择
print('CUDA: ', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 多设备训练的序号
device_ids = [0, 1]

# 输入数据变换，转为张量，归一化（output[channel] = (input[channel] - mean[channel]) / std[channel]）
src_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 标签数据变换，转为张量，标签数据边缘值为0~255中间的过渡值
label_transform = transforms.Compose([
    transforms.ToTensor()
])


def main(args):
    # 创建模型并发送到设备
    net = USSFCNet(in_ch=3, out_ch=1, ratio=0.5).to(device)
    # 多GPU训练，但未证实，详见https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#dataparallel
    # networks = torch.nn.DataParallel(networks, device_ids=device_ids).to(device)
    # 加载最好的模型继续训练
    # networks.load_state_dict(torch.load('ckps/last.pth', map_location='cuda:0'))

    # 超参记录
    start_epoch = 0
    total_epochs = 200
    best_f1 = 0
    best_epoch = 0

    # 损失函数mean_batch(-w[y * log(x) + (1 - y) * log(1 - x)])
    criterion_ce = nn.BCELoss()
    # criterion_ce = nn.CrossEntropyLoss()
    # 优化器Adam
    optimizer = optim.Adam(net.parameters(), args['lr'], weight_decay=0.0005)

    dataset_train = RsDataset(train_src_t1, train_src_t2, train_label,
                              t1_transform=src_transform,
                              t2_transform=src_transform,
                              label_transform=label_transform)
    dataset_val = RsDataset(test_src_t1, test_src_t2, test_label,
                            t1_transform=src_transform,
                            t2_transform=src_transform,
                            label_transform=label_transform)
    # 数据加载器
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=args['batch_size'],
                                  shuffle=True,
                                  num_workers=4)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4)
    # 训练集大小，源代码为num_dataset = len(dataloader_train.dataset)
    num_dataset = len(dataset_train)
    # 一轮训练迭代次数，数据集长度 / batch_size
    total_step = (num_dataset - 1) // dataloader_train.batch_size + 1

    # 创建日志文件夹
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # 创建日志生成器
    logger = get_logger('logs/' + TITLE + '.log')
    # 写入日志
    logger.info('Net: ' + TITLE)
    logger.info('Batch Size: {}'.format(args['batch_size']))
    logger.info('Learning Rate: {}'.format(args['lr']))
    # 检查点保存路径
    ckp_savepath = 'ckps/' + TITLE
    # 创建检查点保存文件夹
    if not os.path.exists(ckp_savepath):
        os.makedirs(ckp_savepath)

    # 开始循环训练和验证
    for epoch in range(start_epoch, total_epochs):
        # 输出第几轮标识
        print('Epoch {}/{}'.format(epoch + 1, total_epochs))
        print('=' * 10)
        # 轮次更新
        epoch += 1
        # 训练函数
        epoch_loss_train, pre_train, recall_train, f1_train, iou_train, kc_train = train(net, dataloader_train,
                                                                                         total_step, criterion_ce,
                                                                                         optimizer)
        print('epoch %d - train loss:%f, train Pre:%f, train Rec:%f, train F1:%f, train iou:%f, train kc:%f' % (
            epoch, epoch_loss_train / total_step, pre_train, recall_train, f1_train, iou_train, kc_train))
        logger.info(
            'Epoch:[{}/{}]\t train_loss={:.5f}\t train_Pre={:.3f}\t train_Rec={:.3f}\t train_F1={:.3f}\t train_IoU={:.3f}\t train_KC={:.3f}'.format(
                epoch, total_epochs, epoch_loss_train / total_step, pre_train, recall_train, f1_train, iou_train,
                kc_train))
        writer_train.add_scalar('loss_of_train', epoch_loss_train / total_step, epoch)
        writer_train.add_scalar('f1_of_train', f1_train, epoch)
        writer_all.add_scalar('loss_of_train', epoch_loss_train / total_step, epoch)
        writer_all.add_scalar('f1_of_train', f1_train, epoch)

        pre_val, recall_val, f1_val, iou_val, kc_val = validate(net, dataloader_val, epoch)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_epoch = epoch
            ckp_name = TITLE + '_batch={}_lr={}_epoch{}model.pth'.format(args['batch_size'], args['lr'], epoch)
            torch.save(net.state_dict(), os.path.join(ckp_savepath, ckp_name), _use_new_zipfile_serialization=False)

        print('epoch %d - val Pre:%f val Recall:%f val F1Score:%f' % (epoch, pre_val, recall_val, f1_val))
        logger.info(
            'Epoch:[{}/{}]\t val_Pre={:.4f}\t val_Rec:{:.4f}\t val_F1={:.4f}\t IoU={:.4f}\t KC={:.4f}\t best_F1:[{:.4f}/{}]\t'.format(
                epoch, total_epochs, pre_val, recall_val, f1_val, iou_val, kc_val, best_f1, best_epoch))

        writer_val.add_scalar('f1_of_validation', f1_val, epoch)
        writer_all.add_scalar('f1_of_validation', f1_val, epoch)
