"""
MIT License

Copyright (c) 2022 SLAMcore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Data import get_loader
from Models import get_model
from Losses import get_losses
from Metrics.metrics import epe_metric
from Metrics.metrics import tripe_metric
from Metrics.metrics import epe_metric_non_zero
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import gc
from matplotlib import pyplot as plt
from torchsummary import summary

class TrainSolver(object):

    def __init__(self, config):

        self.config = config

        from datetime import datetime
        self.sessionTimeStamp = datetime.now().strftime("%d_%m___%H_%M_%S")
        print("start session time = ", self.sessionTimeStamp)
        self.cfg_solver = config['solver']
        self.cfg_dataset = config['data']
        self.cfg_model = config['model']
        self.reloaded = True if self.cfg_solver['resume_iter'] > 0 else False
        self.writer = None
        self.max_disp = self.cfg_model['max_disp']
        self.loss_name = self.cfg_model['loss']
        self.train_loader, self.val_loader = get_loader(self.config)
        self.model = get_model(self.config)
        self.crit = get_losses(self.loss_name, max_disp=self.max_disp, lcn_weight=self.cfg_solver['lcn_weight'], occluded_weight=self.cfg_solver['occluded_weight'])

        if self.cfg_solver['optimizer_type'].lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.cfg_solver['lr_init'])
        elif self.cfg_solver['optimizer_type'].lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg_solver['lr_init'])
        else:
            raise NotImplementedError('Optimizer type [{:s}] is not supported'.format(self.cfg_solver['optimizer_type']))
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg_solver['milestones'], gamma=self.cfg_solver['gamma'])
        self.global_step = 1
        self.save_val = self.cfg_solver['save_eval']

    def save_checkpoint(self):

        ckpt_root = os.path.join(self.cfg_solver['exp_prefix'], self.cfg_solver['model_name'], 'models')
        
        if not os.path.exists(ckpt_root):
            os.makedirs(ckpt_root)
        
        ckpt_name = 'iter_{:d}.pth'.format(self.global_step)
        states = {
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        ckpt_full = os.path.join(ckpt_root, ckpt_name)
        
        torch.save(states, ckpt_full)
    
    def load_checkpoint(self):

        ckpt_root = os.path.join(self.cfg_solver['exp_prefix'], self.cfg_solver['model_name'], 'models')

        ckpt_name = 'iter_{:d}.pth'.format(self.cfg_solver['resume_iter'])
        
        ckpt_full = os.path.join(ckpt_root, ckpt_name)

        states = torch.load(ckpt_full, map_location=lambda storage, loc: storage)

        self.global_step = states['global_step']
        self.model.load_state_dict(states['model_state'])
        self.optimizer.load_state_dict(states['optimizer_state'])
        self.scheduler.load_state_dict(states['scheduler_state'])

    def run(self):
        # self.model.CoarseNet.conv3d_5.register_forward_hook(get_activation('coarse conv3d_5 before upsample'))
        self.model.CoarseNet.disp_reg.register_forward_hook(get_activation('coarse activation maps distribution (full disparity values)'))

        #  edge disparity detection refine ( disparity added to coarse disparity map )
        self.model.RefineNet.conv2.register_forward_hook(get_activation('refine activation map distribution (full disparity values)'))
        print(self.model)
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))

        if self.cfg_solver['resume_iter'] > 0:
            self.load_checkpoint()
            print('[{:d}] Model loaded.'.format(self.global_step))
        data_iter = iter(self.train_loader)
        print_info = 1000
        tot_loss = 0.0
        tot_EPE_ref = 0.0
        tot_EPE_coarse = 0.0
        sample_weight = 1.0
        refine_weight = self.cfg_solver['refine_head_weight']
        self.optimizer.zero_grad() # optimizing performance of optimizer (memory wise)
        while True:
            try:
                data_batch = data_iter.next()
            except StopIteration:
                data_iter = iter(self.train_loader)
                data_batch = data_iter.next()

            if self.global_step > self.cfg_solver['max_steps']:
                break

            sample_wei = sample_weight
            start_time = time.time()
            
            self.model.train()
            imgL, imgR, disp_L, _ = data_batch
            imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()
            disp_pred_ref_left, disp_pred_coarse_left, disp_pred_ref_right, disp_pred_coarse_right, res_disp_left, res_disp_right = self.model(imgL, imgR, disp_L, True)

            lossRefine, recon_img_left_ref, recon_img_right_ref = self.crit(imgL, imgR, disp_pred_ref_left, disp_pred_ref_right, disp_L, sample_wei)
            lossCoarse, recon_img_left_coarse, recon_img_right_coarse = self.crit(imgL, imgR, disp_pred_coarse_left, disp_pred_coarse_right, disp_L, sample_wei)
            loss = refine_weight * lossRefine + (1 - refine_weight) * lossCoarse
            loss_hist = loss.item()
            tot_loss += loss_hist
            loss /= self.cfg_solver['accumulate']
            loss.backward()

            if self.global_step % self.cfg_solver['SaveImagesInterval'] == 0:
                dir = os.path.join(self.cfg_solver['exp_prefix'], self.cfg_solver['model_name'], 'saved figs', self.sessionTimeStamp)
                if not os.path.exists(dir):
                    os.makedirs(dir)

                fig, ax = plt.subplots(3, 3)
                fig.set_size_inches(20, 20)

                ax[0, 0].imshow(imgL[0].permute(1, 2, 0).cpu().detach().numpy());
                ax[0, 0].set_title('left')

                ax[0, 1].imshow(recon_img_left_ref[0].permute(1, 2, 0).cpu().detach().numpy());
                ax[0, 1].set_title('reconstructed left')

                ax[0, 2].imshow(np.abs(imgL[0].permute(1, 2, 0).cpu().detach().numpy() - recon_img_left_ref[0].permute(1, 2, 0).cpu().detach().numpy()))
                ax[0, 2].set_title('left - reconstructed left')

                ax[1, 0].imshow(imgR[0].permute(1, 2, 0).cpu().detach().numpy());
                ax[1, 0].set_title('right')

                ax[1, 1].imshow(recon_img_right_ref[0].permute(1, 2, 0).cpu().detach().numpy());
                ax[1, 1].set_title('reconstructed right')

                ax[1, 2].imshow(disp_L[0].permute(1, 2, 0).cpu().detach().numpy(), vmin=0, vmax=self.max_disp);
                ax[1, 2].set_title('GT disparity')

                ax[2, 0].imshow(disp_pred_coarse_left[0].permute(1, 2, 0).cpu().detach().numpy(), vmin=0, vmax=self.max_disp);
                ax[2, 0].set_title('Coarse only left')

                ax[2, 1].imshow(res_disp_left[0].permute(1, 2, 0).cpu().detach().numpy(), vmin=0, vmax=self.max_disp);
                ax[2, 1].set_title('refined only left')

                ax[2, 2].imshow(disp_pred_ref_left[0].permute(1, 2, 0).cpu().detach().numpy(), vmin=0, vmax=self.max_disp)
                ax[2, 2].set_title('Coarse + Refine')

                file = os.path.join(dir, str(self.global_step) + '_.png')
                # plt.show()
                fig.savefig(file)

            coarse = disp_pred_coarse_left[0].detach().cpu()
            mask = (coarse > self.max_disp) & (coarse < 0)
            coarse[mask] = 0

            all = disp_pred_ref_left[0].detach().cpu()
            mask = (all > self.max_disp) & (all < 0)
            all[mask] = 0

            refined = res_disp_left[0].detach().cpu()
            mask = (refined > self.max_disp) & (refined < 0)
            refined[mask] = 0

            maxVal = disp_L[0].cpu().max()

            refinedOnlyOut = np.stack((refined * (1.0 / maxVal),) *3, axis=0).squeeze()
            overallOut = np.stack((all * (1.0 / maxVal),) *3, axis=0).squeeze()
            coarseOut = np.stack((coarse * (1.0 / maxVal),) *3, axis=0).squeeze()
            gt = np.stack((disp_L[0].cpu() * (1.0 / maxVal.max()),) * 3, axis=0).squeeze()

            images =  torchvision.utils.make_grid([imgL[0], recon_img_left_ref[0], imgR[0]])
            dispMaps = torchvision.utils.make_grid([torch.Tensor(gt), torch.tensor(overallOut), torch.Tensor(coarseOut), torch.tensor(refinedOnlyOut)])

            if self.writer is None:
                self.writer = SummaryWriter()

            self.writer.add_scalar('Loss/Train', loss, self.global_step)
            self.writer.add_image("gt | coarse+refine | coarse | refine", dispMaps, self.global_step)
            self.writer.add_image("left, reconstructed left, right", images, self.global_step)

            self.writer.add_histogram("coarse weights before last", self.model.module.CoarseNet.conv3d_4[0].weight, self.global_step)
            self.writer.add_histogram("coarse weights last", self.model.module.CoarseNet.conv3d_5[0].weight, self.global_step)

            self.writer.add_histogram("refine weights before last", self.model.module.RefineNet.resblock6.conv[0].weight, self.global_step)
            self.writer.add_histogram("refine weights", self.model.module.RefineNet.conv2[0].weight, self.global_step)

            self.writer.add_histogram("coarse out", activation['coarse activation maps distribution (full disparity values)'], self.global_step)
            self.writer.add_histogram("refine out", activation['refine activation map distribution (full disparity values)'], self.global_step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # if self.global_step % self.cfg_solver['accumulate'] == 0:
                # log_gradients_in_model(self.model.module, self.writer, self.global_step)


            elapsed = time.time() - start_time
            train_EPE_left_ref = epe_metric(disp_L.detach(), disp_pred_ref_left.detach(), self.max_disp)
            train_3PE_left = tripe_metric(disp_L.detach(), disp_pred_ref_left.detach(), self.max_disp)
            train_EPE_left_coarse = epe_metric(disp_L.detach(), disp_pred_coarse_left.detach(), self.max_disp)

            self.writer.add_scalar('EPE overall prediction/Train', train_EPE_left_ref, self.global_step)

            tot_EPE_ref += train_EPE_left_ref
            tot_EPE_coarse += train_EPE_left_coarse
            print('[{:d}/{:d}] Train Loss = {:.6f}, Avg. Train Loss = {:.3f}, EPE_ref = {:.3f} px, EPE_coarse = {:.3f}, Avg. EPE_ref = {:.3f} px, Avg. EPE_coarse = {:.3f} px, 3PE = {:.3f}%, time = {:.3f}s.'.format
                (
                    self.global_step, self.cfg_solver['max_steps'],
                    loss_hist,
                    tot_loss / ((self.global_step % print_info) + 1),
                    train_EPE_left_ref, 
                    train_EPE_left_coarse,
                    tot_EPE_ref / ((self.global_step % print_info) + 1),
                    tot_EPE_coarse / ((self.global_step % print_info) + 1),
                    train_3PE_left * 100,
                    elapsed
                )
            )

            self.scheduler.step()
            if self.global_step % self.cfg_solver['save_steps'] == 0 and not self.reloaded:
                self.save_checkpoint()
                print('')
                print('[{:d}] Model saved.'.format(self.global_step))
            
            
            if self.global_step % self.cfg_solver['eval_steps'] == 0 and not self.reloaded:
                elapsed = 0.0
                
                self.model.eval()
                with torch.no_grad():
                    EPE_metric_left = 0.0
                    val_EPE_metric_left = 0.0
                    val_TriPE_metric_left = 0.0
                    N_total = 0
                    valid = 1e-6
                    fig_root = os.path.join(self.cfg_solver['exp_prefix'], self.cfg_solver['model_name'], 'results', 'val', str(self.global_step).zfill(6))
                    if self.save_val and (not os.path.exists(fig_root)):
                        os.makedirs(fig_root)

                    for val_batch in self.val_loader:
                        imgL, imgR, disp_L, _= val_batch
                        imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()

                        N_curr = imgL.shape[0]

                        start_time = time.time()
                        ref_pred_left, coarse_pred_left, _, _, _, _= self.model(imgL, imgR, disp_L, False)

                        elapsed += (time.time() - start_time)
                        N_total += N_curr
                        is_valid = (disp_L > 0).float().mean() > 0.5

                        if is_valid:
                            if self.cfg_solver['refine_head_weight'] > -0.5:
                                EPE_metric_left = epe_metric_non_zero(disp_L, ref_pred_left, self.max_disp) * N_curr 
                                val_TriPE_metric_left += tripe_metric(disp_L, ref_pred_left, self.max_disp) * N_curr
                            else:
                                EPE_metric_left = epe_metric_non_zero(disp_L, coarse_pred_left, self.max_disp) * N_curr
                                val_TriPE_metric_left += tripe_metric(disp_L, coarse_pred_left, self.max_disp) * N_curr

                            val_EPE_metric_left += EPE_metric_left
                            valid += N_curr

                        if self.save_val:
                            fig, ax = plt.subplots(3,3)
                            fig.set_size_inches(10,10)
                            if is_valid:
                                fig.suptitle('EPE error: ' + str(EPE_metric_left))
                            else:
                                fig.suptitle('Invalid')
                            ax[0,0].imshow(imgL[0].permute(1,2,0).cpu().detach().numpy()); ax[0,0].set_title('IR0')
                            ax[0,1].imshow(imgR[0].permute(1,2,0).cpu().detach().numpy()); ax[0,1].set_title('IR1')
                            ax[0,2].imshow(disp_L[0,0].cpu().detach().numpy(), vmin=0, vmax=self.max_disp); ax[0,2].set_title('GT disparity')
                            ax[1,0].imshow(ref_pred_left[0,0].cpu().detach().numpy(), vmin=0, vmax=self.max_disp); ax[1,0].set_title('Refine head')
                            ax[1,1].imshow(coarse_pred_left[0,0].cpu().detach().numpy(), vmin=0, vmax=self.max_disp); ax[1,1].set_title('Coarse head')
                            ax[1,2].imshow(np.abs(ref_pred_left[0,0].cpu().detach().numpy() - coarse_pred_left[0,0].cpu().detach().numpy()), vmin=0, vmax=5); ax[1,2].set_title('Diff: Ref - Coarse')
                            ax[2,0].imshow(np.abs(ref_pred_left[0,0].cpu().detach().numpy() - disp_L[0,0].cpu().detach().numpy()), vmin=0, vmax=10); ax[2,0].set_title('Diff: Ref - GT')
                            ax[2,1].imshow(np.abs(coarse_pred_left[0,0].cpu().detach().numpy() - disp_L[0,0].cpu().detach().numpy()), vmin=0, vmax=10); ax[2,1].set_title('Diff: Coarse - GT')
                            ax[2,2].axis('off')
                            fig_nm = fig_root + '/' + str(N_total).zfill(5) + '.png'
                            fig.savefig(fig_nm)                           

                        if N_total % 1 == 0:
                            plt.close('all')
                        print(
                            '[{:d}/{:d}] Validation : valid = {:d}, EPE = {:.6f} px, Avg. EPE = {:.3f} px, Avg. 3PE = {:.3f} %, time = {:.3f} s.'.format(
                            N_total, len(self.val_loader),
                            int(valid),
                            EPE_metric_left,
                            val_EPE_metric_left / valid, 
                            val_TriPE_metric_left * 100 / valid, 
                            elapsed / N_total
                            ), end='\r'
                        )

                    plt.close('all')
                    val_EPE_metric_left /= valid
                    val_TriPE_metric_left /= valid

                    print(
                        '[{:d}/{:d}] Validation : valid = {:d}, EPE = {:.6f} px, 3PE = {:.3f} %, time = {:.3f} s.'.format(
                            N_total, len(self.val_loader),
                            int(valid),
                            val_EPE_metric_left, 
                            val_TriPE_metric_left * 100, 
                            elapsed / N_total
                        )
                    )

            if self.global_step % print_info == 0:
                print('Total updates: {:d}, Avg loss = {:.3f}, Avg. EPE_ref = {:.3f} px, Avg. EPE_coarse = {:.3f} px\n'.format(self.global_step, 
                    tot_loss / print_info,
                    tot_EPE_ref / print_info,
                    tot_EPE_coarse / print_info))
                tot_loss = 0.0
                tot_EPE_ref = 0.0
                tot_EPE_coarse = 0.0
            self.global_step += 1

            self.reloaded = False


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None and tag.find('weight') > 0:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)

def showImages(data, rows, columns):
    fig = plt.figure(figsize=(rows, columns), dpi=100)
    axes = []
    for i in range(1, columns * rows + 1):
        element = data[i - 1]
        axes.append(fig.add_subplot(rows, columns, i))
        axes[i - 1].imshow(data[i - 1])
    plt.show()