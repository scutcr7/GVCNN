import os
import numpy as np
from tqdm import tqdm
import shutil
import sklearn.metrics as metrics
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
import time

from config import args, logger, device
from data import ModelNet40Views
from loss import cal_loss
from GVCNN import GVCNN

cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def train():
   
    save_dir = 'exp_gvcnn_{}_{}'.format(args.mv_backbone.lower(), time.strftime("%Y%m%d-%H%M%S"))
                                                                
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_dir = os.path.join(save_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    summary_dir = os.path.join(save_dir, 'summary')
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    logger.info("Loading dataset...")
    logger.info('ModelNet40')
    train_loader = DataLoader(ModelNet40Views(args.data_dir,  args.mv_backbone, mode='train'), num_workers=8,
                                   batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40Views(args.data_dir,  args.mv_backbone, mode='val'), num_workers=8,
                                  batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    num_classes = 40
    logger.info('classes: {}'.format(num_classes))

    logger.info('Creating model...')
    model = GVCNN(num_classes=40, group_num=args.group_num, model_name=args.mv_backbone).to(device)
    model = nn.DataParallel(model)
    criterion = cal_loss

    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate,
                                 betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.learning_rate * 100,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise RuntimeError('optimizer type not supported.({})'.format(args.optimizer))
    scheduler= CosineAnnealingLR(optimizer, args.num_epochs, eta_min=args.learning_rate)

    summary_writer = SummaryWriter(log_dir=os.path.join(summary_dir, args.mv_backbone))

    logger.info('start training.')
    best_test_acc = 0
    for epoch in range(1, args.num_epochs + 1):
        ####################
        # Train
        ####################
        tqdm_batch = tqdm(train_loader, desc='Epoch-{} training'.format(epoch))
        model.train()
        train_loss = 0.0
        train_pred = []
        count = 0.0
        train_true = []
        for data, label in tqdm_batch:
            data, label = data.to(device), label.to(device)
            batch_size = data.size(0)
            optimizer.zero_grad()

            pred, feature = model(data)
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()

            preds = pred.max(dim=1)[1]
            train_loss += loss.item() * batch_size
            train_pred.append(preds.detach().cpu().numpy())

            count += batch_size
            train_true.append(label.cpu().numpy())

        scheduler.step()

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch, train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(train_true, train_pred))
        logger.info(outstr)

        summary_writer.add_scalar('train/loss', train_loss * 1.0 / count, epoch)
        summary_writer.add_scalar('train/overall_acc', metrics.accuracy_score(train_true, train_pred), epoch)
        summary_writer.add_scalar('train/avg_acc', metrics.balanced_accuracy_score(train_true, train_pred), epoch)


        ####################
        # Test
        ####################
        tqdm_batch = tqdm(test_loader, desc='Epoch-{} testing'.format(epoch))
        model.eval()
        test_loss = 0.0
        test_pred = []
        count = 0.0
        test_true = []
        for data, label in tqdm_batch:
            data, label = data.to(device), label.to(device)
            batch_size = data.size(0)

            pred, feature = model(data)

            loss = criterion(pred, label)

            preds = pred.max(dim=1)[1]
            test_loss += loss.item() * batch_size
            test_pred.append(preds.detach().cpu().numpy())

            count += batch_size
            test_true.append(label.cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)

        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch, test_loss * 1.0 / count,
                                                                              metrics.accuracy_score(test_true,
                                                                                                     test_pred),
                                                                              metrics.balanced_accuracy_score(test_true,
                                                                                                              test_pred))
        logger.info(outstr)

        summary_writer.add_scalar('test/loss', test_loss * 1.0 / count, epoch)
        summary_writer.add_scalar('test/overall_acc', metrics.accuracy_score(test_true, test_pred), epoch)
        summary_writer.add_scalar('test/avg_acc', metrics.balanced_accuracy_score(test_true, test_pred), epoch)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, args.mv_backbone+'best_model.pth'))

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, args.mv_backbone+'model_{}.pth'.format(epoch)))

        logger.info('best_test_acc: {:.6f}'.format(best_test_acc))



if __name__ == '__main__':
    train()
