from utils.data import *
from utils.metric import *
from argparse import ArgumentParser
import torch
import torch.utils.data as Data
from model.MSHNet import *
from model.loss import *
from torch.optim import Adagrad,AdamW
from tqdm import tqdm
import os.path as osp
import os
import time
from timm.scheduler import CosineLRScheduler

os.environ['CUDA_VISIBLE_DEVICES']="0"
torch.backends.cudnn.benchmark = True

def parse_args():

    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--weight-path', type=str, default='weight/MSHNet-2025-01-07-16-43-47/weight.pkl')

    args = parser.parse_args()
    return args

def calculateF1Measure(output_image,gt_image,thre=0.5):
    output_image=output_image.cpu().numpy()
    gt_image=gt_image.cpu().numpy()
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return F1

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        self.start_epoch = 0   
        self.mode = args.mode
        self.batch_size=args.batch_size

        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')

        self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=True)
        self.val_loader = Data.DataLoader(valset, 1, drop_last=False)

        device = torch.device('cuda')
        self.device = device

        model = MSHNet(3)

        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use '+str(torch.cuda.device_count())+' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)
        self.model = model

        # self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=0.05)

        self.scheduler = CosineLRScheduler(self.optimizer,
                                  t_initial=args.epochs,
                                  cycle_mul=1.,
                                  lr_min=5e-4,
                                  cycle_decay=0.1,
                                  warmup_lr_init=5e-4,
                                  warmup_t=5,
                                  cycle_limit=1,
                                  t_in_epochs=True)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch

        if args.mode=='train':
            if args.if_checkpoint:
                check_folder = ''
                checkpoint = torch.load(check_folder+'/checkpoint.pkl')
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch']+1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                self.save_folder = 'weight/MSHNet-%s'%(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
                if not osp.exists(self.save_folder):
                    os.mkdir(self.save_folder)
        if args.mode=='test':
          
            weight = torch.load(args.weight_path)
            # self.model.load_state_dict(weight['state_dict'])
            self.model.load_state_dict(weight)
            '''
                # iou_67.87_weight
                weight = torch.load(args.weight_path)
                self.model.load_state_dict(weight)
            '''
            self.warm_epoch = -1
        

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        tag = False
        for i, (data, mask) in enumerate(tbar):
  
            data = data.to(self.device)
            labels = mask.to(self.device)

            if epoch>self.warm_epoch:
                tag = True

            masks, pred = self.model(data, tag)
            loss = 0

            loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
            for j in range(len(masks)):
                if j>0:
                    labels = self.down(labels)
                loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch)
                
            loss = loss / (len(masks)+1)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
       
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
            # if i==10:
            #     exit(0)
    
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        tbar = tqdm(self.val_loader)
        tag = False
        f1_score=0.
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
    
                data = data.to(self.device)
                mask = mask.to(self.device)

                if epoch>self.warm_epoch:
                    tag = True

                loss = 0
                _, pred = self.model(data, tag)
                # loss += self.loss_fun(pred, mask,self.warm_epoch, epoch)
                # if i==68:
                #     import matplotlib.pyplot as plt
                #     import cv2
                #     # import ipdb; ipdb.set_trace()
                #     tensor=pred[0][0].clone()
                #     tensor = tensor - tensor.min()
                #     tensor = tensor / tensor.max()
                #     tensor = tensor.cpu().numpy()

                #     tensor[tensor<0.5]=0.
                #     tensor[tensor>=0.5]=1.
                #     tensor*=255
                #     tensor=cv2.resize(tensor,(328,325))
                #     print(tensor.shape)
                #     print(tensor)
                #     plt.imsave('output_image.png', tensor, cmap='gray')
                #     exit(0)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()
                f1_score+=calculateF1Measure(pred,mask)

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))
            FA, PD = self.PD_FA.get(len(self.val_loader))
            f1_score/=len(self.val_loader)
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()

            
            if self.mode == 'train':
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                
                    torch.save(self.model.state_dict(), self.save_folder+'/weight.pkl')
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_iou, PD[0], FA[0] * 1000000))
                        
                all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
                torch.save(all_states, self.save_folder+'/checkpoint.pkl')
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')
                print(f'F1 Score: {f1_score}')


         
if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    
    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
            trainer.scheduler.step(epoch)
    else:
        trainer.test(1)
 