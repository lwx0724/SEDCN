import os
import math
from decimal import Decimal
import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from ssim import  ssim,rgb2ycbcr

from torch.autograd import Variable
class Trainer():
    def __init__(self, args,dataloader,my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = dataloader.loader_train
        self.loader_val = dataloader.loader_val#测试dataloader list
        self.loader_test = dataloader.dataloaders
        self.use_siamese =args.use_siamese
        self.loss = my_loss
        if args.use_mix:
            self.optimizer,self.model= utility.make_optimizer(args,my_model)
            if len(args.gpu_id) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_id, output_device=args.gpu_id[0])
        else:
            self.model = my_model
            self.optimizer = utility.make_optimizer(args,self.model)

        self.firstLoad = True
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, batchData in tqdm(enumerate(self.loader_train)):
            lr = batchData[0]
            hr = batchData[1]
            scale_index = batchData[-1]
            if self.use_siamese:
                lr_up =batchData[3]
                lr, hr,lr_up = self.prepare(lr, hr,lr_up)
            else:
                lr, hr = self.prepare(lr, hr)

            timer_data.hold()#导入数据累计用时
            timer_model.tic()#当前时间

            self.optimizer.zero_grad()
            if self.use_siamese:
                sr = self.model(lr,scale_index,lr_up)
            else:
                sr =self.model(lr,scale_index)
            # loss1 = self.loss(s1, hr)
            # loss2 = self.loss(s2, hr)
            # loss3 = self.loss(s3, hr)
            # loss4 = self.loss(s4, hr)
            # loss5 = self.loss(s5, hr)
            # loss6 = self.loss(s6, hr)
            # loss7 = self.loss(s7, hr)
            # loss8 = self.loss(sr, hr)
            # loss = (0.1*loss1+0.2*loss2+0.3*loss3+0.4*loss4+0.5*loss5+0.6*loss6+0.7*loss7+loss8).div(3.3)
            loss = self.loss(sr, hr)

            # 半精度
            if self.args.use_mix:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            #loss.backward()
            self.optimizer.step()
            timer_model.hold()#模型一个batch用时

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()#刷新数据导入时间点

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        print('loss is :',self.error_last.item())
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, 2, 1)#单个测试集 torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()

        save_num =20
        saved_num =0
        for batchData in tqdm(self.loader_val):#调试下
            lrs = batchData[0]
            hr = batchData[1]
            filename =batchData[2]
            scale_index =batchData[-1]
            if self.use_siamese:
                lr_up = batchData[3]
                lrs, hr,lr_up = self.prepare(lrs, hr,lr_up)
                sr = self.model(lrs, scale_index,lr_up)
            else:
                lrs, hr = self.prepare(lrs, hr)
                sr = self.model(lrs,scale_index)

            #将补过齐的测试图片还原
            addpixel_h = batchData[4].numpy()[0]
            addpixel_w = batchData[5].numpy()[0]
            if addpixel_h != 0 or addpixel_w != 0:
                #不能对tensor 做截断
                scale = int(self.scale)
                lrh = lrs.size()[2]
                lrw = lrs.size()[3]
                tmp_lrs = lrs.cpu().numpy()
                tmp_lrs = tmp_lrs[:,:,0:lrh-addpixel_h,0:lrw-addpixel_w]
                tmp_lrs = torch.from_numpy(tmp_lrs)
                srh = (lrh - addpixel_h) * scale
                srw = (lrw - addpixel_w) * scale
                tmp_sr  = sr.cpu().numpy()
                tmp_sr  = tmp_sr[:,:,0:srh,0:srw]
                tmp_sr  = torch.from_numpy(tmp_sr)
                tmp_hr  = hr.cpu().numpy()
                tmp_hr  = tmp_hr[:, :, 0:srh, 0:srw]
                tmp_hr  = torch.from_numpy(tmp_hr)
                lrs, hr,sr = self.prepare(tmp_lrs, tmp_hr,tmp_sr)

            sr = utility.quantize(sr, self.args.rgb_range)#返回整数像素
            lrs = utility.quantize(lrs, self.args.rgb_range)
            hr = utility.quantize(hr, self.args.rgb_range)

            tmp = ssim(sr,hr,data_range=255,size_average=True)
            self.ckp.log[-1,1,0] +=tmp

            save_list = [sr]
            self.ckp.log[-1,0,0] += utility.calc_psnr2(
                sr, hr, self.args.rgb_range
            )

            # if self.args.save_gt:
            #     save_list.extend([lr, hr])
            #
            # if self.args.save_results:
            #     self.ckp.save_results(filename, save_list, 4)
            if saved_num<save_num:
                save_list.extend([lrs, hr])
                self.ckp.save_results(filename[0], save_list, self.scale)
                saved_num +=1


        self.ckp.log[-1,0,0] /= len(self.loader_val)
        self.ckp.log[-1,1,0] /= len(self.loader_val)
        best = self.ckp.log.max(0)
        self.ckp.write_log(
            '[x{}]\tPSNR: {:.3f}\tSSIM: {:.3f}'.format(
                self.scale,
                self.ckp.log[-1, 0, 0],
                self.ckp.log[-1,1,0]
            )
        )
        tmp = self.ckp.log[-1, 0, 0]
        tmp = float('%.2f'%tmp)
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, psnr=tmp,is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)


    #以set5为验证集
    def test2(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, 2, 1)#单个测试集 torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()

        _item ='Set5'
        currentloader = self.loader_test[_item]

        for batchData in tqdm(currentloader):  # 调试下
            lr = batchData[0]
            hr  = batchData[1]
            filename = batchData[2]
            scale_index =batchData[-1]
            if self.use_siamese:
                lr_up = batchData[3]
                lr, hr, lr_up = self.prepare(lr, hr, lr_up)
                sr = self.model(lr, scale_index,lr_up)
            else:
                lr, hr = self.prepare(lr, hr)
                sr = self.model(lr,scale_index)

            # 将补过齐的测试图片还原
            addpixel_h = batchData[4].numpy()[0]
            addpixel_w = batchData[5].numpy()[0]
            scale = int(self.scale)
            if addpixel_h != 0 or addpixel_w != 0:
                # 不能对tensor 做截断
                lrh = lr.size()[2]
                lrw = lr.size()[3]
                tmp_lrs = lr.cpu().numpy()
                tmp_lrs = tmp_lrs[:, :, 0:lrh - addpixel_h, 0:lrw - addpixel_w]
                tmp_lrs = torch.from_numpy(tmp_lrs)
                srh = (lrh - addpixel_h) * scale
                srw = (lrw - addpixel_w) * scale
                tmp_sr = sr.cpu().numpy()
                tmp_sr = tmp_sr[:, :, 0:srh, 0:srw]
                tmp_sr = torch.from_numpy(tmp_sr)
                tmp_hr = hr.cpu().numpy()
                tmp_hr = tmp_hr[:, :, 0:srh, 0:srw]
                tmp_hr = torch.from_numpy(tmp_hr)
                lr, hr, sr = self.prepare(tmp_lrs, tmp_hr, tmp_sr)

            lr = utility.quantize(lr, self.args.rgb_range)
            sr = utility.quantize(sr, self.args.rgb_range)  # 返回整数像素
            hr = utility.quantize(hr, self.args.rgb_range)


            tmp = ssim(sr, hr, data_range=255, size_average=True)
            self.ckp.log[-1,1,0]  += tmp

            self.ckp.log[-1,0,0] += utility.calc_psnr2(
                sr, hr,int(self.scale), self.args.rgb_range
            )

            save_list = [sr]
            save_list.extend([lr, hr])

            self.ckp.save_results(filename[0], save_list, self.scale)

        self.ckp.log[-1, 0, 0] /= 5
        self.ckp.log[-1, 1, 0] /= 5
        best = self.ckp.log.max(0)

        self.ckp.write_log(
            '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                scale,
                self.ckp.log[-1,0,0],
                best[0][0,0],
                best[1][0,0]
            )
        )

        tmp = self.ckp.log[-1, 0, 0]
        tmp = float('%.2f' % tmp)
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, psnr=tmp, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
        #生成test原图,按文件夹分类保存
    def outputTest(self):
        torch.set_grad_enabled(False)
        self.ckp.write_log('\nTest:')
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()

        save_dir ='./test_output'
        for _item in ['Set5','Set14','B100','Urban100']:# ['B100','Set5','Set14','Urban100']:
            currentloader = self.loader_test[_item]
            all_psnr =0
            all_ssim =0
            for batchData in tqdm(currentloader):  # 调试下
                lr = batchData[0]
                hr  = batchData[1]
                filename = batchData[2]
                scale_index =batchData[-1]
                if self.use_siamese:
                    lr_up = batchData[3]
                    lr, hr, lr_up = self.prepare(lr, hr, lr_up)
                    sr = self.model(lr, scale_index,lr_up)
                else:
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr,scale_index)
                # 将补过齐的测试图片还原
                addpixel_h = batchData[4].numpy()[0]
                addpixel_w = batchData[5].numpy()[0]
                if addpixel_h != 0 or addpixel_w != 0:
                    # 不能对tensor 做截断
                    scale = int(self.scale)
                    lrh = lr.size()[2]
                    lrw = lr.size()[3]
                    tmp_lrs = lr.cpu().numpy()
                    tmp_lrs = tmp_lrs[:, :, 0:lrh - addpixel_h, 0:lrw - addpixel_w]
                    tmp_lrs = torch.from_numpy(tmp_lrs)
                    srh = (lrh - addpixel_h) * scale
                    srw = (lrw - addpixel_w) * scale
                    tmp_sr = sr.cpu().numpy()
                    tmp_sr = tmp_sr[:, :, 0:srh, 0:srw]
                    tmp_sr = torch.from_numpy(tmp_sr)
                    tmp_hr = hr.cpu().numpy()
                    tmp_hr = tmp_hr[:, :, 0:srh, 0:srw]
                    tmp_hr = torch.from_numpy(tmp_hr)
                    lr, hr, sr = self.prepare(tmp_lrs, tmp_hr, tmp_sr)

                lr = utility.quantize(lr, self.args.rgb_range)
                sr = utility.quantize(sr, self.args.rgb_range)  # 返回整数像素
                hr = utility.quantize(hr, self.args.rgb_range)

                sr_np = sr.squeeze(dim=0).transpose(0,2).transpose(0,1).cpu().numpy()
                hr_np = hr.squeeze(dim=0).transpose(0,2).transpose(0,1).cpu().numpy()
                sr_np = rgb2ycbcr(sr_np)
                hr_np = rgb2ycbcr(hr_np)
                sr_np = sr_np[:, :, 0].astype(float)
                sr_tensor = Variable(torch.from_numpy(sr_np).float()).view(1, -1, sr_np.shape[0], sr_np.shape[1])
                hr_np = hr_np[:, :, 0].astype(float)
                hr_tensor = Variable(torch.from_numpy(hr_np).float()).view(1, -1, hr_np.shape[0], hr_np.shape[1])

                tmp = ssim(sr_tensor, hr_tensor, data_range=255, size_average=True)

                all_ssim += tmp

                all_psnr += utility.calc_psnr2(
                    sr, hr, int(self.scale),self.args.rgb_range
                )

                save_list = [sr]
                save_list.extend([lr, hr])
                #构造输出文件夹
                outputdir = os.path.join(save_dir,_item)
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                #输出图片路径
                outputfilename =os.path.join(outputdir,filename[0])
                self.ckp.test_save_result(outputfilename, save_list, self.scale)
            all_psnr /=len(currentloader)
            all_ssim /=len(currentloader)
            utility.writelog(save_dir ,'{} PSNR:{:.3f} SSIM:{:.3f}\n'.format(_item,all_psnr,all_ssim))

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)

    def traning_Test(self):
        torch.set_grad_enabled(False)
        self.ckp.write_log('\nTest:')
        self.model.eval()
        timer_test = utility.timer()

        epoch = self.optimizer.get_last_epoch()

        if self.args.save_results: self.ckp.begin_background()

        root_dir = self.ckp.dir
        save_dir = os.path.join(root_dir,'test_output')
        log_dir =os.path.join(save_dir,'epoch{}'.format(str(epoch)))
        for _item in ['B100','Set5', 'Set14', 'Urban100']:#, 'Set5', 'Set14', 'Urban100'
            currentloader = self.loader_test[_item]
            all_psnr = 0
            all_ssim = 0
            for batchData in tqdm(currentloader):  # 调试下
                lr = batchData[0]
                hr = batchData[1]
                filename = batchData[2]
                scale_index = batchData[-1]

                # k = torch.squeeze(lr_up)
                # imgRGB = transforms.ToPILImage()(k).convert('RGB')
                # t_save_path = os.path.join('./test_output/LR_UP2.png')
                # imgRGB.save(t_save_path,quality=100)

                # k2 = torch.squeeze(hr)
                # imgRGB2 = transforms.ToPILImage()(k2).convert('RGB')
                # t_save_path = os.path.join('./test_output/HR2.png')
                # imgRGB2.save(t_save_path,quality=100)
                if self.use_siamese:
                    lr_up = batchData[3]
                    lr, hr, lr_up = self.prepare(lr, hr, lr_up)
                    sr = self.model(lr, scale_index,lr_up)
                else:
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr,scale_index)

                # 将补过齐的测试图片还原
                addpixel_h = batchData[4].numpy()[0]
                addpixel_w = batchData[5].numpy()[0]
                if addpixel_h != 0 or addpixel_w != 0:
                    # 不能对tensor 做截断
                    scale = int(self.scale)
                    lrh = lr.size()[2]
                    lrw = lr.size()[3]
                    tmp_lrs = lr.cpu().numpy()
                    tmp_lrs = tmp_lrs[:, :, 0:lrh - addpixel_h, 0:lrw - addpixel_w]
                    tmp_lrs = torch.from_numpy(tmp_lrs)
                    srh = (lrh - addpixel_h) * scale
                    srw = (lrw - addpixel_w) * scale
                    tmp_sr = sr.cpu().numpy()
                    tmp_sr = tmp_sr[:, :, 0:srh, 0:srw]
                    tmp_sr = torch.from_numpy(tmp_sr)
                    tmp_hr = hr.cpu().numpy()
                    tmp_hr = tmp_hr[:, :, 0:srh, 0:srw]
                    tmp_hr = torch.from_numpy(tmp_hr)
                    lr, hr, sr = self.prepare(tmp_lrs, tmp_hr, tmp_sr)

                lr = utility.quantize(lr, self.args.rgb_range)
                sr = utility.quantize(sr, self.args.rgb_range)  # 返回整数像素
                hr = utility.quantize(hr, self.args.rgb_range)
                tmp = ssim(sr, hr, data_range=255, size_average=True)
                all_ssim += tmp

                all_psnr += utility.calc_psnr2(
                    sr, hr, int(self.scale),self.args.rgb_range
                )

                save_list = [sr]
                save_list.extend([lr, hr])
                # 构造输出文件夹
                outputdir = os.path.join(save_dir,'epoch{}'.format(str(epoch)), _item)
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                # 输出图片路径
                outputfilename = os.path.join(outputdir, filename[0])
                self.ckp.test_save_result(outputfilename, save_list, self.scale)
            all_psnr /= len(currentloader)
            all_ssim /= len(currentloader)
            utility.writelog(log_dir, '{} PSNR:{:.3f} SSIM:{:.3f}\n'.format(_item, all_psnr, all_ssim))

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda:0')
        return [a.to(device) for a in args]

    def terminate(self):
        if self.args.test_only:
            if self.args.test_val:
                #测试指定数据的psnr,不记录
                self.test2()
            else:
                self.outputTest()

            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
