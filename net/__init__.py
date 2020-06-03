import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo
from apex import amp,parallel
from util import bnwd_optim_params
from collections import  OrderedDict

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda:0')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        self.training =args.training
        module = import_module('net.' + args.model.lower())
        self.model = module.make_model(args)#调试下
        self.use_siamese = args.use_siamese
        self.self_ensemble =args.self_ensemble
        self.chop =args.chop
        self.precision = args.precision
        self.every_save_model = args.every_epoch_save_model
        self.use_several_gpu = args.use_several_gpu
        #是否使用混合精度
        if args.use_mix:
            print('using Mix precision')
            self.model = parallel.convert_syncbn_model(self.model)
            self.model = self.model.to(self.device)
            self.parameters = [p for p in self.model.parameters() if p.requires_grad]
            self.parameters = bnwd_optim_params(self.model, self.parameters)
        else:
            self.model = self.model.to(self.device)

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        if args.ispretrain:
            self.load(
                ckp.get_path('model'),
                pre_train=args.pre_train,
                resume=args.resume,
                cpu=args.cpu
            )

        #print(self.model, file=ckp.log_file)

    # def forward(self, x):#, idx_scale,改
    #     if self.training:
    #         if self.n_GPUs > 1:
    #             return P.data_parallel(self.model,x, range(self.n_GPUs))
    #         else:
    #             return self.model(x)
    #     else:
    #         forward_function = self.model.forward
    #         return forward_function(x)

    #输入不定参数,idx_scale 和 bigx
    def forward(self, x, *kw):
        self.idx_scale = kw[0]
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(kw[0])

        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward
            return self.forward_x8(x, forward_function)

        elif self.chop and not self.training:
            return self.forward_chop(x)
        else:
            if self.use_siamese ==True:
                return self.model(x,kw[1])
            else:
                return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    # def forward(self, x, bigx):  # , idx_scale,改
    #     if self.training:
    #         if self.n_GPUs > 1:
    #             return P.data_parallel(self.model, (x, bigx), range(self.n_GPUs))
    #         else:
    #             return self.model(x, bigx)
    #     else:
    #         forward_function = self.model.forward
    #         return forward_function(x, bigx)



    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            if epoch % self.every_save_model==0:
                save_dirs.append(
                    os.path.join(apath, 'model_{}.pt'.format(epoch))
                )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        new_state_dict = OrderedDict()
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)

        else:
            print('Load the model_{}'.format(resume))
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if self.training == False and self.use_several_gpu==True:
            for k, v in load_from.items():
                name = k[7:]
                new_state_dict[name] = v

        if load_from:
            if self.training:
                self.model.load_state_dict(load_from,strict=False)#, strict=False
            else:
                if self.use_several_gpu:
                    self.model.load_state_dict(new_state_dict,strict=False)
                else:
                    self.model.load_state_dict(load_from, strict=False)

    def forward_chop(self, x, shave=8, min_size=160000):
        scale = int(self.scale[self.idx_scale])
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

