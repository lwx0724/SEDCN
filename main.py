import torch
import utility
import net
from loss import Loss
from option import args
from train import Trainer
from util  import  seed_torch
import data

map_scale ={2:'96',3:'144',4:'192',8:'384'}
train_status =1#0: 正常训练 1:测试

if __name__ == '__main__':
    args.scale ='2' #放大因子
    args.patch_size =96
    args.benchmark_dir = './DataSet/benchmark'
    args.dir_data = './DataSet'
    args.pre_train = './model/X2/model_best.pt'#'../experiment/model/models_ECCV2018RCAN/model_20.pt'
    args.load ='' #'baseline4'

    args.model ='hourglass_sr8' #修改模型
    args.batch_size =20
    args.n_resblocks = 20
    args.res_scale =0.1
    args.n_resgroups = 6#
    args.save ='baseline53'
    args.every_epoch_save_model =50
    args.use_siamese = False
    args.use_several_gpu =False
    args.use_hourglass = True
    args.test_every =1000
    args.n_feats = 64
    args.growth =8
    args.decay ='200-400-600-800'#4个沙漏
    args.epoch = 1000
    args.lr = 1e-3
    args.noise ='.'#测试加入高斯噪声图片['G','15']
    if train_status==1:
        #不改参数model_best_nosie0.15.pt
        args.test_only =True
        args.training =False
        args.ispretrain = 1
        args.n_GPUs =1
        #需要改变的参数
        args.chop =False
        args.self_ensemble = False
        args.test_val =False #False:测试验证集 True:测试测试集
    elif train_status==0:
        #不改参数
        args.test_only =False
        args.training = True
        args.chop = False
        args.self_ensemble = False
        #需改参数
        args.ispretrain = 0

    torch.backends.cudnn.benchmark = True
    seed_torch()
    checkpoint = utility.checkpoint(args)

    #创建dataloader
    dataloader = data.Data(args)

    _model = net.Model(args, checkpoint)
    _loss = Loss(args, checkpoint)

    t = Trainer(args, dataloader,_model, _loss, checkpoint)
    epoch = 1
    while not t.terminate():
        t.train()
        t.test2()
        if epoch %50 ==0 or epoch==1:#or epoch==1
            t.traning_Test()
        epoch +=1
    checkpoint.done()