from importlib import import_module

from dataloader2 import MSDataLoader
#from torch.utils.data.dataloader import default_collate
from torch.utils.data import _utils
import os
class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] =_utils.collate.default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = _utils.collate.default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        self.loader_val = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            module_val = import_module('data.' + args.data_val.lower())
            valset = getattr(module_val, args.data_val)(args,train=False)

            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )
            self.loader_val = MSDataLoader(
                args,
                valset,
                batch_size=1,
                shuffle=False,
                **kwargs
            )

        if args.data_test =='benchmark':
            dir_list = os.listdir(args.benchmark_dir)
            self.dataloaders ={}
            for item in dir_list:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test,'Benchmark')(args,item ,train=False)

                self.dataloaders[item] = MSDataLoader(
                    args,
                    testset,
                    batch_size=1,
                    shuffle=False,
                    **kwargs
                )
