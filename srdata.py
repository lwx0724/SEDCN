import os

from data import common

import numpy as np
import scipy.misc as misc
import torch
import torch.utils.data as data
from PIL import  Image
import time

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale
            ]

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr = self._scan()
            print(self.images_hr)
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        lr = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_hr, list_lr = self._scan()
                hr = [misc.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [misc.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                _load_bin()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr,hr,addpixel_h,addpixel_w = self._get_patch(lr,hr)
        if self.args.use_siamese:
            img = Image.fromarray(lr.astype('uint8')).convert('RGB')
            h = img.size[0]
            w = img.size[1]
            lr_up = img.resize((h * int(self.scale), w * int(self.scale)), Image.BICUBIC)
            lr_up = np.asarray(lr_up)
            lr, hr,lr_up = common.set_channel([lr, hr,lr_up], self.args.n_colors)
            lr_tensor, hr_tensor,lr_up_tensor = common.np2Tensor([lr, hr,lr_up], self.args.rgb_range)
            return lr_tensor, hr_tensor,filename,lr_up_tensor,addpixel_h,addpixel_w
        else:
            lr, hr = common.set_channel([lr, hr], self.args.n_colors)
            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
            return lr_tensor, hr_tensor, filename,0,addpixel_h,addpixel_w

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = misc.imread(lr)
            hr = misc.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, filename

    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = int(self.scale[self.idx_scale])
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, self.args.noise)
        else:
            if self.args.use_siamese ==True:
                #xx = time.time()
                ih, iw = lr.shape[0:2]
                addPixel_h = int(ih %8)
                addPixel_w = int(iw %8)
                if addPixel_h!=0:
                    addPixel_h = 8-ih %8
                if addPixel_w!=0:
                    addPixel_w = 8-iw %8
                hr = hr[0:ih * scale, 0:iw * scale]

                if addPixel_h!=0 or addPixel_w!=0:
                    lr = np.pad(lr, ((0, addPixel_h), (0, addPixel_w),(0,0)), 'constant', constant_values=(128, 128))
                    hr = np.pad(hr, ((0, addPixel_h*scale), (0, addPixel_w*scale),(0,0)), 'constant', constant_values=(128, 128))
                #bb  = time.time()-xx
                #print('pading time:',bb)
                return lr,hr,addPixel_h,addPixel_w
            else:
                if self.args.use_hourglass:
                    zc = 8
                    ih, iw = lr.shape[0:2]
                    addPixel_h = int(ih % zc)
                    addPixel_w = int(iw % zc)
                    if addPixel_h != 0:
                        addPixel_h = zc - ih % zc
                    if addPixel_w != 0:
                        addPixel_w = zc - iw % zc
                    hr = hr[0:ih * scale, 0:iw * scale]

                    if addPixel_h != 0 or addPixel_w != 0:
                        lr = np.pad(lr, ((0, addPixel_h), (0, addPixel_w), (0, 0)), 'constant',
                                    constant_values=(128, 128))
                        hr = np.pad(hr, ((0, addPixel_h * scale), (0, addPixel_w * scale), (0, 0)), 'constant',
                                    constant_values=(128, 128))
                    lr = common.add_noise(lr, self.args.noise)
                    return lr, hr, addPixel_h, addPixel_w
                else:
                    ih, iw = lr.shape[0:2]
                    lr = common.add_noise(lr, self.args.noise)
                    hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr,0,0

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

