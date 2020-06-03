import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import  adabound
import adamW
import random
#from apex import amp,parallel
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        #创建实验文件夹
        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('../experiment', args.save)
        else:
            self.dir = os.path.join('../experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)#重新训练，删除保存文件夹
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save_model(self, model,apath, epoch,psnr ,is_best=False,save_models=True ):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}_{}.pt'.format(epoch,psnr))
            )

        for s in save_dirs:
            torch.save(model.state_dict(), s)

    def save(self, trainer, epoch, psnr,is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        #self.save_model(trainer.model,self.get_path('model'), epoch,psnr=psnr ,is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format('DIV2K')
        fig = plt.figure()
        plt.title(label)

        plt.plot(
            axis,
            self.log[:, 0, 0].numpy(),
            label='Scale {}'.format(4)
        )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(self.get_path('test_{}.pdf'.format('DIV2K')))
        plt.close(fig)


    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()
    #训练时输出过程中图片
    def save_results(self,filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
    #测试时输出图片
    def test_save_result(self,filenamedir,save_list,scale):
        if self.args.save_results:
            filename = '{}_x{}_'.format(filenamedir, scale)

        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].mul(255 / self.args.rgb_range)
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, rgb_range):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    mse = diff.pow(2).mean()

    return -10 * math.log10(mse)

def calc_psnr2(sr, hr, scale, rgb_range):
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

#记录中途训练的print
def writelog(logdir,log):
    # now = int(time.time())
    # timeArray = time.localtime(now)
    # otherStyleTime = time.strftime("%Y-%m-%d-%H:%M:%S", timeArray)

    os.makedirs(os.path.join(logdir), exist_ok=True)
    open_type = 'a' if os.path.exists(os.path.join(logdir,'log.txt')) else 'w'
    log_file = open(os.path.join(logdir,'log.txt'), open_type)

    print(log)
    log_file.write(log + '\n')
    log_file.close()

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    if args.use_mix:
        trainable = target.parameters
    else:
        trainable = filter(lambda x: x.requires_grad, target.model.parameters())

    kwargs_optimizer = {'lr': args.lr,'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'adabound':
        optimizer_class = adabound
    elif args.optimizer =='ADAMW':
        optimizer_class = adamW.AdamW
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    # kwargs_scheduler ={'step_size':1,'gamma':0.8}
    # scheduler_class =lrs.StepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)#输入训练参数list,和参数
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)

    if args.use_mix:
        model, optimizer =amp.initialize(target.model,optimizer, opt_level="O1")
        return optimizer,model
    else:
        return optimizer

