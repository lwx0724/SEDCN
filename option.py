import argparse

parser = argparse.ArgumentParser(description='wdsr_b')

#Quantitative
parser.add_argument('--scale', default='2',
                    help='super resolution scale')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--data_train',type=str,default='DIV2K',help='train dataset name')
parser.add_argument('--data_val',type=str,default='DIV2K',help='val dataset name')

parser.add_argument('--data_test', type=str, default='benchmark',
                    help='test dataset name')
#设置每个周期迭代次数
parser.add_argument('--test_every',type = int ,default=1000,help='train iter num')
#设置训练集数量
parser.add_argument('--n_train',type=int ,default=800,help='train image num')
#测试集起始位置
parser.add_argument('--offset_val',type=int,default=800,help='val start index')
#设置测试集数量
parser.add_argument('--n_val',type=int,default=5,help='test num ')

#是否只测试，默认false
parser.add_argument('--test_only', default=False ,
                    help='set this option to test the model')
#ture为测试val ,反之test
parser.add_argument('--test_val', default=False,
                    help='set this option to test the model')
#读取格式
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--benchmark_dir',type=str,default='../data/benchmark',help='benchmark dir')
parser.add_argument('--dir_data', type=str, default='../data',
                    help='dataset directory')


#是否训练,与上面设置相反的
parser.add_argument('--training', default=True,help ='training or test ,default is training')

#是否pretrain,若要与加载请置为1
parser.add_argument('--ispretrain', type=int, default=0,
                    help='is or not pretrain')

#导入模型,-1导入前一个周期，0，一定要写pretain地址，导入指定模型；任意>0整数意思为导入model_n模型
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--pre_train', type=str, default='../experiment/model/models_ECCV2018RCAN/RCAN_BIX4.pt',
                    help='pre-trained model directory')

parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
#['G','15']
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')

parser.add_argument('--chop', default=False,
                    help='enable memory-efficient forward')
parser.add_argument('--self_ensemble', default=False,
                    help='use self-ensemble method for test')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
#每隔多少个周期
parser.add_argument('--every_epoch_save_model',type=int,default= 20,help='the number of save model')
#是否使用孪生网络,使用后再net/__init__ 跟换forward
parser.add_argument('--use_siamese', default=False,
                    help='use siamese network')
parser.add_argument('--use_hourglass', default=True,
                    help='use siamese network')
parser.add_argument('--use_several_gpu', default=True,
                    help='use siamese network')

# Model specifications
parser.add_argument('--model', default='rcan',#siamese_rcan
                    help='model name')
parser.add_argument('--batch_size', default=16,
                    help='batch size, e.g. 16, 32, 64...', type=int)
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=10,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=20,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=8,
                    help='number of threads for data loading')
#默认用gpu
parser.add_argument('--cpu', default=False,
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--gpu_id', default=[0], nargs='+',
                    help='gpu ids to use, e.g. 0 1 2 3', type=int)

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')

#衰减策略
parser.add_argument('--decay', type=str, default='200-400-600-800',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')#超分一般无正则项

# Log specifications
#训练新的,创建新的文件夹，重新命名
parser.add_argument('--save', type=str, default='baseline',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
#是否所有模型
parser.add_argument('--save_models', default=True,
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
#是否保存图片
parser.add_argument('--save_results', default=True,
                    help='save output results')
#保存图片时是否保存lr和hr
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

#重新训练,会删除保存模型的文件夹
parser.add_argument('--reset',default=False,
                    help='reset the training')
# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
#是否使用混合精度
parser.add_argument('--use_mix',default=False,
                    help='use mix precision')

parser.add_argument('--block_feats',default=128,
                    help='use mix precision')
parser.add_argument('--growth',default=4,
                    help='desneblock growth')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

args = parser.parse_args()