import  os
import  glob
import numpy as np
import pandas as pd
import cv2
import torch
import random
import sys
import time
from datetime import datetime
from shutil import get_terminal_size
#对指定根目录下所有视频文件进行全分帧
def getFrames(root_dir,output_dir):
    allvedio =glob.glob(os.path.join(root_dir,'./*/*.y4m'))
    command = 'ffmpeg -i '
    command2 =' -vsync 0 '
    for _dir in allvedio:
        print(_dir)
        dataClassfication = _dir.split('/')[5]
        source_or_target =_dir.split('/')[6]
        dirName =_dir.split('/')[-1][0:-4]
        print(dirName)
        output_path = os.path.join(output_dir,dataClassfication,source_or_target,dirName)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        outputfile =os.path.join(output_path,dirName+'%3d.bmp')
        #code ='ffmpeg -i xx.y4m -vsync 0 xx%3d.bmp -y'
        finalCommand =command + _dir +command2 +outputfile+' -y'
        os.system(finalCommand)


#对指定根目录下的视频文件进行按频率拆帧
def getFrames_rate(root_dir,rates,output_dir):
    allvedio =glob.glob(os.path.join(root_dir,'./*.y4m'))
    command1 ='ffmpeg -i '
    command2 =' -vsync 0 -y '
    command3 =" -vf select='not(mod(n\,{}))'".format(rates)
    for _dir in allvedio:
        print(_dir)
        dirName = _dir.split('/')[-1][0:-8]
        #code ='ffmpeg -i xxx.y4m -vf select='not(mod(n\,25))' -vsync 0  -y xxx_sub25.y4m'
        outputFile = os.path.join(output_dir,dirName+'_Sub{}_Res.y4m'.format(rates))
        finalCommand =command1+_dir +command3+command2+outputFile
        print(finalCommand)
        os.system(finalCommand)



#对指定文件夹下图片进行合帧,并保存至指定路径、
def composeFrames(dir,output_dir):
       filelist = os.listdir(dir)
       command1 ='ffmpeg -i '
       command2 =' -pix_fmt yuv420p -vsync 0 '
       for _i,_filelist in enumerate(filelist):
         filedir =glob.glob(os.path.join(dir,filelist[_i],'./*.bmp'))
         dirname = filedir[0].split('/')[-1][0:-7]
         dirname2 =filedir[0].split('/')[-3]
         inputDir =os.path.join(dir,_filelist,dirname+'%3d.bmp')
         outputFile =os.path.join(output_dir,dirname2+'.y4m -y')
       #code ='ffmpeg -i xx%3d.bmp  -pix_fmt yuv420p  -vsync 0 xx.y4m -y'
         finalCommand =command1 + inputDir +command2 +outputFile
         os.system(finalCommand)

#求均值方差的函数
def meanAndStd(dataset_dir,sampleNum,img_size_h,img_size_w):
    print(os.path.join(dataset_dir, './*/*.bmp'))
    all_imgs = glob.glob(os.path.join(dataset_dir, './*/*.bmp'))
    train = pd.DataFrame({'image_path': all_imgs})
    a=train.sample(sampleNum,axis=0)
    img_h, img_w = img_size_w,img_size_h
    imgs = np.zeros([img_h, img_w, 3, 1])
    means, stdevs = [], []
    for index, row in a.iterrows():
        img = cv2.imread(row['image_path'])
        img = cv2.resize(img,(img_w,img_h))
        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)
        print(index)

    imgs = imgs.astype(np.float32)/255.
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

#求均值方差的函数
def meanAndStd2(dataset_dir,sampleNum,img_size_h,img_size_w):
    print(os.path.join(dataset_dir, './*/*.bmp'))
    all_imgs = glob.glob(os.path.join(dataset_dir, './*/*.bmp'))
    train = pd.DataFrame({'image_path': all_imgs})
    a=train.sample(sampleNum,axis=0)
    img_h, img_w = img_size_w,img_size_h
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []
    for index, row in a.iterrows():
        img = cv2.imread(row['image_path'])
        img = cv2.resize(img,(img_size_w,img_size_h))
        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)
        print(index)

    imgs = imgs.astype(np.float32)/255.
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

def seed_torch(seed=2048):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def split_bn_params(model, model_params):
    def get_bn_params(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): return module.parameters()
        accum = set()
        for child in module.children(): [accum.add(p) for p in get_bn_params(child)]
        return accum

    mod_bn_params = get_bn_params(model)

    bn_params = [p for p in model_params if p in mod_bn_params]
    rem_params = [p for p in model_params if p not in mod_bn_params]
    return bn_params, rem_params


# Filter out batch norm parameters and remove them from weight decay - gets us higher accuracy 93.2 -> 93.48
# https://arxiv.org/pdf/1807.11205.pdf
# code adopted from https://github.com/diux-dev/imagenet18/blob/master/training/experimental_utils.py
def bnwd_optim_params(model, model_params):
    bn_params, remaining_params = split_bn_params(model, model_params)
    return [{'params': bn_params, 'weight_decay': 0}, {'params': remaining_params}]

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()



# if __name__ == '__main__':
#       #getFrames('/media/lwx/学习/SR/val/target','/home/lwx/SRData')
#       getFrames_rate('/media/lwx/学习/test_output_vedio/45',25,'/media/lwx/学习/test_output_vedio')
#       #composeFrames('/home/lwx/PycharmProjects/SR_competion1/test_output_2019-06-15-15:43:12','/media/lwx/学习/test_output_vedio')
#       #meanAndStd2('/home/lwx/SRData/train/source',500,270,480)