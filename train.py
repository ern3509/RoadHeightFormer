import argparse
import os
import shutil
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from utils.dataset import RSRD
from torch.cuda.amp import GradScaler
from models.loss import MyLoss, LossReg, LossReg2
from torch.utils.data import DataLoader
from models.model import Elevation
import pickle
from torch.hub import load_state_dict_from_url
import os
from utils.metric import Metric
from utils.experiment import *
import numpy as np
from datetime import datetime
from CARDSet.dataset import CARDSetDataset, CARDSetDatasetV2Smalldataset
import wandb
import time
import matplotlib.pyplot as plt


os.environ['WANDB_MODE'] = 'online'

now = datetime.now()
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Early stopping to terminate training when validation loss does not improve.

        Parameters:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change in monitored value to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_loss):
        """
        Check if training should stop based on current validation loss.

        Parameters:
            current_loss (float): Current epoch's validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            return False

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

def unnormalize(ele_pred, h_min, h_max):
    #height = ele_pred[:, 0:1]  # keep channel dim
    #height = height * ((h_max - h_min) / 2) + ((h_max + h_min) / 2)
    #ele_pred = torch.cat([height, ele_pred[:, 1:2]], dim=1)

    ele_pred = ele_pred * ((h_max - h_min) / 2) + ((h_max + h_min) / 2)
    return ele_pred

def train_regression():
    print("Training with regression loss")
    run = wandb.init(
        entity = "erwan-adonie-njike-ndjongang-cariad",
        project = "RoadHeightFormer",
        name = args.name_run +  str(now.month) + '/' + str(now.day),
        notes = args.notes,
        config ={
            "learning_rate" : args.lr,
            "epochs": args.epochs,
            "dataset": args.dataset,
            "trainloader length": len(train_loader),
            "testloader length": len(test_loader),
            "scheduler" : args.scheduler,
            "backbone" : args.backbone,
            "loss_function" : args.loss,
            "Batch_size" : args.batch_size,
    })
    global_step = 0
    logged_train_static = False
    gt_vmax = [0, 0, 0]
    gt_vmin = [14, 14, 14]
    logged_eval_static = False

    for epoch_idx in tqdm(range(args.epochs)):
        time_epoch = time.time()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch_idx+1}", unit="batch") as pbar:
            for i, sample in enumerate(train_loader):
                global_step += 1
                start_time = time.time()
                if args.stereo:
                    (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
                    imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
                else:
                    (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
                imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if args.stereo:
                        ele_pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
                    else:
                        ele_pred = model(imgs_left, proj_index_left)
                        #print("train ele pred shape:", ele_pred.shape)
                        ele_pred_fixed = model(train_imgs_fixed, train_proj_fixed)
                    
                    loss_all = loss_func(ele_pred, ele_gt, ele_mask)
                    if args.normalize:
                        h_min = - ele_range * 100
                        h_max = ele_range * 100                        
                        ele_pred_fixed = unnormalize(ele_pred_fixed, h_min, h_max)
                        print("max and min after normalization:", ele_pred_fixed.max().item(), ele_pred_fixed.min().item())

            
            #/****logging ***********************
                print("logging step:", global_step, args.summary_freq)
                if global_step % args.summary_freq == 0: 
                    log_dict = {}
                    for s in range(len(fixed_train_indices)):
                        if not logged_train_static:
                            gt_np = np.ma.masked_where(
                            train_mask_fixed[s].cpu().numpy() == 0,
                            train_gt_fixed[s].cpu().numpy(),
                            )
                            gt_vmin[s] = gt_np.min()
                            gt_vmax[s] = gt_np.max()

                        height_prediction = ele_pred_fixed[s]#, 0]	
                        combined_img = wandb_combined_image(
                        height_prediction.squeeze(),
                        train_gt_fixed[s],
                        train_mask_fixed[s],
                        train_imgs_fixed[s],
                        caption=f"Combined Visualization of sample {s} at step {global_step}",
                        vmin=gt_vmin[s],
                        vmax=gt_vmax[s],
                        )
                        wandb.log({"train/combined_sample_" + str(s): combined_img}, step=global_step)

                    logged_train_static = True
                    wandb.log(log_dict, step=global_step)
                scaler.scale(loss_all).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()


                if args.regression and args.scheduler == 'reduceonplaeau':
                    scheduler.step(loss_all.data.item())
                else:
                    scheduler.step()


                epoch_active_time = time.time() - start_time
                loss_wandb = loss_all.data.item()

                if np.isnan(loss_wandb):
                    print('nan loss!')
                    torch.save(model.state_dict(), "modelbeforebreak.ckpt")
                    exit()
                print("loss has been logged")
                wandb.log({"loss": loss_wandb}, step = global_step)

                info = 'train--> epoch%2d, lr:%.6f, loss:%.4f' % (epoch_idx+1, optimizer.param_groups[0]['lr'], loss_wandb)
                print(info)


                if global_step % (10*args.summary_freq) == 0:
                    """    loss_data = loss_all.data.item()
                    if np.isnan(loss_data):
                        print('nan loss!')
                        exit() 
                    """
                    log_file.write(info + '\n')
                    log_file.flush()
                    [metric_all, _], eval_loss = test_sample_regression(test_loader, global_step, run, logged_eval_static)
                    wandb.log({"metrics/eval_loss": eval_loss}, step = global_step)
                    wandb.log({"metrics/metric_abs": metric_all[0]}, step = global_step)
                    wandb.log({"metrics/metric_rmse": metric_all[1]}, step = global_step)
                    wandb.log({"metrics/metric_gt05cm": metric_all[2]}, step = global_step)
                    #wandb.log({"log_rmse": metric_all[3]}, step = global_step)
                    wandb.log({"metrics/abs_err_0.1": metric_all[3]}, step = global_step)
                    wandb.log({"metrics/abs_err_1": metric_all[4]}, step = global_step)
                    wandb.log({"metrics/le90": metric_all[5]}, step = global_step)
                    wandb.log({"metrics/grad_err": metric_all[6]}, step = global_step)

                if global_step % (20*args.summary_freq) == 0:
                    #torch.save(model.state_dict(), "{}/checkpoint_epoch{:0>2}_{:0>6}.ckpt".format(args.logdir, epoch_idx+1, global_step))

                    torch.cuda.empty_cache() 

                    early_stopping(eval_loss)
                    

                    info = 'test:  abs_err:%.3f, rmse:%.3f, >0.5cm:%.2f, grad_error:%.3f eval_loss:%.3f' % (metric_all[0], metric_all[1],metric_all[2]*100, metric_all[-1], eval_loss)
                    log_file.write(info + '\n')
                    log_file.flush()
                    print(info)
            pbar.update(1)
            # if early_stopping.should_stop:
            #     print("Early stopping triggered!")
            #     break
        time_epoch_end = time.time() - time_epoch
        wandb.log({"epoch/epoch_duration": time_epoch_end}, step=global_step)
        wandb.log({"epoch/epoch": epoch_idx+1}, step=global_step)
    run.finish()
@make_nograd_func
def test_sample_regression(test_loader, global_step, run, logged_eval_static):
    model.eval()
    eval_loss = 0.0
    gt_vmin = [0, 0, 0]
    gt_vmax = [14, 14, 14]
    h_min = - ele_range*100
    h_max = ele_range*100
    #save file for visualization pytorch
    ele_pred_fixed = model(eval_imgs_fixed, eval_proj_fixed)
    
    for s in range(len(fixed_eval_indices)):
        if args.normalize:
            print("undo normalization in visualization of some testing sample")
            ele_pred_fixed = ele_pred_fixed * ((h_max - h_min) / 2) + ((h_max + h_min) / 2)
        
        if not logged_eval_static:
            gt_np = np.ma.masked_where(
                        eval_mask_fixed[s].cpu().numpy() == 0,
                        eval_gt_fixed[s].cpu().numpy(),
                    )
            gt_vmin[s] = gt_np.min()
            gt_vmax[s] = gt_np.max()

        height_prediction = ele_pred_fixed[s]
        combined_img = wandb_combined_image(
                        height_prediction.squeeze(),
                        eval_gt_fixed[s],
                        eval_mask_fixed[s],
                        eval_imgs_fixed[s],
                        caption=f"Combined Evaluation Visualization at step {global_step}",
                        vmin=gt_vmin[s],
                        vmax=gt_vmax[s],
                        test=True
                        )
        wandb.log({"test/combined_sample_" + str(s): combined_img}, step=global_step)
    logged_eval_static = True
    for i, sample in enumerate(test_loader):
        if args.stereo:
            (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
            imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
        else:
            (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
        imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()
        
        with torch.cuda.amp.autocast(dtype=torch.float16):

            if args.stereo:
                ele_pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
            else:
                ele_pred = model(imgs_left, proj_index_left)
                ele_pred_fixed = model(eval_imgs_fixed, eval_proj_fixed)
                #ele_pred = ele_pred[:, 0, :, :] #from B, 2, H, W to B, H, W
                #ele_pred_fixed = ele_pred_fixed[:, 0, :, :]
            
            if args.normalize:
                print("undo normalization in testing")
                h_min = - ele_range*100
                h_max = ele_range*100
                ele_pred = ele_pred * ((h_max - h_min) / 2) + ((h_max + h_min) / 2)
                ele_pred_fixed = ele_pred_fixed * ((h_max - h_min) / 2) + ((h_max + h_min) / 2)

            metric.compute(ele_pred, ele_gt, ele_mask)
            
            loss_all = sum_absolute_error(ele_pred[ele_mask > 0], ele_gt[ele_mask > 0]) / ele_mask.sum().item()
            log_dict = {}
            """"
            if i == 0:
                # Log static data ONCE
                if not logged_eval_static:
                    for s in range(len(fixed_eval_indices)):
                        gt_np = np.ma.masked_where(
                                eval_mask_fixed[s].cpu().numpy() == 0,
                                eval_gt_fixed[s].cpu().numpy(),
                            )
                        gt_vmin[s] = gt_np.min()
                        gt_vmax[s] = gt_np.max()
                        print("gt min and max:", gt_vmin[s], gt_vmax[s])
                        log_dict[f"eval/static/sample_{s}/image"] = \
                            wandb_rgb_image(eval_imgs_fixed[s], caption="Eval image")

                        log_dict[f"eval/static/sample_{s}/gt"] = \
                            wandb_heightmap_image(eval_gt_fixed[s], eval_mask_fixed[s], caption="GT (cm)")

                    logged_eval_static = True

                # Log predictions WITH SLIDER
                for s in range(len(fixed_eval_indices)):
                    height_prediction = ele_pred_fixed[s]	
                    print("height prediction before softmax shape:", height_prediction.shape)
                    #height_prediction = torch.sum(height_prediction * model.ele_values[0],dim=0)
                    img = wandb_heightmap_image(
                            height_prediction.squeeze(),
                            torch.ones_like(height_prediction),
                            caption=f"step {global_step}",
                            vmin= gt_vmin[s],
                            vmax= gt_vmax[s]
                        )
                    error_img = wandb_error_map(height_prediction.squeeze(), eval_gt_fixed[s], eval_mask_fixed[s], caption=f"Error map at step {global_step} of sample {s}")
                    wandb.log({"eval/pred_sample_" + str(s): img}, step=global_step)
                    wandb.log({"eval/error_map_sample_" + str(s): error_img}, step=global_step)

                wandb.log(log_dict, step=global_step)
"""""
                #CARDSetDatasetV2Smalldataset.visualize_height_map_and_mask(img_prob.squeeze(), ele_mask.squeeze(), colormap='plasma', save_path='Heightmap/' + 'test_' + '_pred' + global_step.__str__() + '.png')
                #CARDSetDatasetV2Smalldataset.visualize_height_map_and_mask(ele_gt.squeeze(), ele_mask.squeeze(), colormap='plasma', save_path='Heightmap/' + 'test_' + '_gt' + global_step.__str__() + '.png')
            
            
            eval_loss += loss_all

    model.train()
    metric_values = metric.get_metric()
    metric.clear()
    eval_loss /= len(test_loader)
    return metric_values, eval_loss
def train():
    print("Train classificationmodel")
    run = wandb.init(
        entity = "erwan-adonie-njike-ndjongang-cariad",
        project = "RoadHeightFormer",
        name = args.name_run +  str(now.month) + '/' + str(now.day),
        notes = args.notes,
        config ={
            "learning_rate" : args.lr,
            "epochs": args.epochs,
            "dataset": args.dataset,
            "trainloader length": len(train_loader),
            "testloader length": len(test_loader),
            "scheduler" : args.scheduler,
            "backbone" : args.backbone,
            "loss_function" : args.loss,
            "Batch_size" : args.batch_size,
    })
    global_step = 0
    logged_train_static = False
    gt_vmax = [0, 0, 0]
    gt_vmin = [14, 14, 14]
    logged_eval_static = False
    for epoch_idx in tqdm(range(args.epochs)):
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch_idx+1}", unit="batch") as pbar:
            for i, sample in enumerate(train_loader):
                global_step += 1
                start_time = time.time()
                if args.stereo:
                    (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
                    imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
                else:
                    (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
                imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if args.stereo:
                        ele_pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
                    else:
                        ele_pred = model(imgs_left, proj_index_left)
                        print("train ele pred shape:", ele_pred.shape)
                        ele_pred_fixed = model(train_imgs_fixed, train_proj_fixed)
                    loss_all = loss_func(ele_pred, ele_gt, ele_mask)

                
            #/****logging ***********************
                if global_step % 5 == 0:
                    log_dict = {}
                    
                    for s in range(len(fixed_train_indices)):
                        
                        # Log static data ONCE
                        if not logged_train_static:
                            log_dict[f"train/static/sample_{s}/image"] = \
                            wandb_rgb_image(train_imgs_fixed[s], caption=f"Train image sample {s}")
                            gt_np = np.ma.masked_where(
                                train_mask_fixed[s].cpu().numpy() == 0,
                                train_gt_fixed[s].cpu().numpy(),
                            )

                            gt_vmin[s] = gt_np.min()
                            gt_vmax[s] = gt_np.max()

                            log_dict[f"train/static/sample_{s}/gt"] = \
                                wandb_heightmap_image(train_gt_fixed[s], train_mask_fixed[s], caption="GT (cm)", vmin = gt_vmin[s], vmax = gt_vmax[s])

                            

                # # Log predictions WITH SLIDER
                # for s in range(3):
                        print("ele pred fixed shape:", ele_pred_fixed.shape)	
                        height_prediction = F.softmax(ele_pred_fixed[s], dim=0)  #for classification ele pred shape: B(number of samples), num_classes, H, W,  #height prediction shape: num_classes, H, W
                        print("height prediction shape:", height_prediction.shape)
                        height_prediction = torch.sum(height_prediction * model.ele_values[0],dim=0) #shape after sum: H, W
                        print("height prediction after sum shape:", height_prediction.shape)
                        img = wandb_heightmap_image(
                                height_prediction.squeeze(),
                                train_mask_fixed[s],  
                                caption=f"step {global_step}", vmin = gt_vmin[s], vmax = gt_vmax[s]
                            )
                        error_img = wandb_error_map(height_prediction.squeeze(), train_gt_fixed[s], train_mask_fixed[s], caption=f"Error map at step {global_step} of sample {s}")
                        wandb.log({"train/error_map_sample_" + str(s): error_img}, step=global_step)
                        wandb.log({"train/pred_sample_" + str(s): img}, step=global_step)
                    
                    logged_train_static = True
                    wandb.log(log_dict, step=global_step)
                scaler.scale(loss_all).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_active_time = time.time() - start_time
                loss_wandb = loss_all.data.item()
                if np.isnan(loss_wandb):
                    print('nan loss!')
                    exit()
                print("loss has been logged")
                wandb.log({"loss": loss_wandb}, step = global_step)

                if global_step % args.summary_freq == 0:
                    loss_data = loss_all.data.item()
                    if np.isnan(loss_data):
                        print('nan loss!')
                        exit()
                    info = 'train--> epoch%2d, lr:%.6f, loss:%.4f' % (epoch_idx+1, optimizer.param_groups[0]['lr'], loss_data)
                    log_file.write(info + '\n')
                    log_file.flush()
                    print(info)

                if global_step % (3*args.summary_freq) == 0:
                    #torch.save(model.state_dict(), "{}/checkpoint_epoch{:0>2}_{:0>6}.ckpt".format(args.logdir, epoch_idx+1, global_step))

                    torch.cuda.empty_cache()
                    [metric_all, _], eval_loss = test_sample(test_loader, global_step, run, logged_eval_static)
                    wandb.log({"eval_loss": eval_loss}, step = global_step)
                    wandb.log({"metric_abs": metric_all[0]}, step = global_step)
                    wandb.log({"metric_rmse": metric_all[1]}, step = global_step)
                    wandb.log({"metric_gt05cm": metric_all[2]}, step = global_step)

                    early_stopping(eval_loss)
                    

                    info = 'test:    abs_err:%.3f, rmse:%.3f, >0.5cm:%.2f, eval_loss:%.3f' % (metric_all[0], metric_all[1], metric_all[2]*100, eval_loss)
                    log_file.write(info + '\n')
                    log_file.flush()
                    print(info)
                epoch_passiv_time = time.time() - epoch_active_time
                #run.log({"epoch_log_time": epoch_passiv_time, "epoch_active_time": epoch_active_time})
            pbar.update(1)
            if early_stopping.should_stop:
                print("Early stopping triggered!")
                break
    run.finish()
@make_nograd_func
def test_sample(test_loader, global_step, run, logged_eval_static=False):
    model.eval()
    eval_loss = 0.0
    gt_vmin = [0, 0, 0]
    gt_vmax = [14, 14, 14]
    

    for i, sample in enumerate(test_loader):
        if args.stereo:
            (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
            imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
        else:
            (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
        imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()
        
        with torch.cuda.amp.autocast(dtype=torch.float16):

            if args.stereo:
                ele_pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
            else:
                ele_pred = model(imgs_left, proj_index_left)
                ele_pred_fixed = model(eval_imgs_fixed, eval_proj_fixed)
            metric.compute(ele_pred, ele_gt, ele_mask)
            #print("youuuu", ele_pred.shape, ele_gt.shape)
            #ele_pred = torch.tensor(ele_pred.unsqueeze(dim=0))
            loss_all = sum_absolute_error(ele_pred[ele_mask > 0], ele_gt[ele_mask > 0]) / ele_mask.sum().item()
            log_dict = {}
            
            if i == 0:
                # Log static data ONCE
                if not logged_eval_static:
                    for s in range(len(fixed_eval_indices)):
                        gt_np = np.ma.masked_where(
                                eval_mask_fixed[s].cpu().numpy() == 0,
                                eval_gt_fixed[s].cpu().numpy(),
                            )
                        gt_vmin[s] = gt_np.min()
                        gt_vmax[s] = gt_np.max()

                        log_dict[f"eval/static/sample_{s}/image"] = \
                            wandb_rgb_image(eval_imgs_fixed[s], caption="Eval image")

                        log_dict[f"eval/static/sample_{s}/gt"] = \
                            wandb_heightmap_image(eval_gt_fixed[s], eval_mask_fixed[s], caption="GT (cm)")

                    logged_eval_static = True
            
                # Log predictions WITH SLIDER
                pred_images = []
                for s in range(len(fixed_eval_indices)):
                    height_prediction = ele_pred_fixed[s]	
                    print("height prediction before softmax shape:", height_prediction.shape)
                    #height_prediction = torch.sum(height_prediction * model.ele_values[0],dim=0)
                    img = wandb_heightmap_image(
                            height_prediction.squeeze(),
                            torch.ones_like(height_prediction),
                            caption=f"step {global_step}",
                            vmin=gt_vmin[s],
                            vmax=gt_vmax[s]
                        )
                    error_img = wandb_error_map(height_prediction.squeeze(), eval_gt_fixed[s], eval_mask_fixed[s], caption=f"Error map at step {global_step} of sample {s}")
                    wandb.log({"eval/pred_sample_" + str(s): img}, step=global_step)
                    wandb.log({"eval/error_map_sample_" + str(s): error_img}, step=global_step)

                wandb.log(log_dict, step=global_step) 

                #CARDSetDatasetV2Smalldataset.visualize_height_map_and_mask(img_prob.squeeze(), ele_mask.squeeze(), colormap='plasma', save_path='Heightmap/' + 'test_' + '_pred' + global_step.__str__() + '.png')
                #CARDSetDatasetV2Smalldataset.visualize_height_map_and_mask(ele_gt.squeeze(), ele_mask.squeeze(), colormap='plasma', save_path='Heightmap/' + 'test_' + '_gt' + global_step.__str__() + '.png')
            
            
            eval_loss += loss_all

    model.train()
    metric_values = metric.get_metric()
    metric.clear()
    eval_loss /= len(test_loader)
    return metric_values, eval_loss

def wandb_error_map(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    caption: str = "",
    cmap: str = "RdBu_r",
    vmin: float = None,
    vmax: float = None,
) -> wandb.Image:
    """
    Compute per-cell error (pred - gt) and return as wandb.Image.

    Args:
        pred: torch.Tensor (H, W) prediction
        gt: torch.Tensor (H, W) ground truth
        mask: torch.Tensor (H, W), 0 = invalid
        caption: image caption
        cmap: diverging colormap 
        vmin/vmax: optional fixed range for color normalization (recommended)

    Returns:
        wandb.Image
    """
    # --- to numpy ---
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    # --- compute error ---
    error = pred - gt
    error = np.ma.masked_where(mask == 0, error)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(error, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Error (cm)")
    fig.tight_layout()

    return wandb.Image(fig, caption=caption)

def wandb_combined_image(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    img: torch.Tensor,
    caption: str = "",
    cmap: str = "plasma",
    vmin: float = None,
    vmax: float = None,
    test = False
) -> wandb.Image:
    """
    Combine prediction, ground truth, and error map into a single image.

    Args:
        pred: torch.Tensor (H, W) prediction.
        gt: torch.Tensor (H, W) ground truth.
        mask: torch.Tensor (H, W), 0 = invalid.
        caption: Image caption.
        cmap: Colormap for the heightmaps.
        vmin/vmax: Optional fixed range for color normalization.

    Returns:
        wandb.Image: Combined image for logging.
    """
    # --- Convert tensors to numpy ---
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    img = img.detach().cpu()
    img = denormalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img = img.permute(1, 2, 0).numpy()

    # --- Mask invalid regions ---
    if test == False:
        pred = np.ma.masked_where(mask == 0, pred)    
    gt = np.ma.masked_where(mask == 0, gt)
    error = np.ma.masked_where(mask == 0, pred - gt)

    # --- Create the figure ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Prediction
    im_pred = axes[0].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Prediction")
    axes[0].axis("off")
    fig.colorbar(im_pred, ax=axes[0], fraction=0.046, pad=0.04)

    # Ground Truth
    im_gt = axes[1].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

    # Error Map
    im_error = axes[2].imshow(error, cmap="RdBu_r")
    axes[2].set_title("Error Map")
    axes[2].axis("off")
    fig.colorbar(im_error, ax=axes[2], fraction=0.046, pad=0.04, label="Error (cm)")

    image = axes[3].imshow(img)
    axes[3].set_title("GT image")
    axes[3].axis("off")


    fig.tight_layout()

    # --- Return as wandb.Image ---
    return wandb.Image(fig, caption=caption)

def wandb_heightmap_image(height_map, mask, caption, cmap="plasma", vmin = None, vmax = None):
    """
    height_map, mask: torch.Tensor (H, W)
    """

    print("....saving file wandb")
    height_map = height_map.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()


    height_map = np.ma.masked_where(mask == 0, height_map)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(height_map, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cm")
    fig.tight_layout()
    
    return wandb.Image(fig, caption=caption)

def wandb_rgb_image(img, caption):
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    return wandb.Image(img, caption=caption)

def get_fixed_samples(dataset, indices, device):
    samples = [dataset[i] for i in indices]

    imgs = torch.stack([s[0] for s in samples]).to(device)
    ele_gt = torch.stack([s[1] for s in samples]).to(device)
    ele_mask = torch.stack([s[2] for s in samples]).to(device)
    proj_idx = torch.stack([s[3] for s in samples]).to(device)

    return imgs, ele_gt, ele_mask, proj_idx

def denormalize(img, mean, std):
    """
    Denormalize a normalized image.

    Args:
        img: torch.Tensor (C, H, W) normalized image.
        mean: List of mean values for each channel.
        std: List of standard deviation values for each channel.

    Returns:
        torch.Tensor: Denormalized image.
    """
    mean = torch.tensor(mean).view(3, 1, 1)  # Reshape to (C, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)    # Reshape to (C, 1, 1)
    return img * std + mean

def sum_absolute_error(pred, gt):
    """
    Compute the Sum of Absolute Error (SAE) between two elevation maps.

    Parameters:
        pred (torch.Tensor): Predicted elevation map of shape (1, 164, 64).
        gt (torch.Tensor): Ground truth elevation map of shape (1, 164, 64).

    Returns:
        float: Sum of absolute error.
    """
    # Ensure both tensors are on the same device and type
    pred = pred.to(dtype=torch.float32)
    gt = gt.to(dtype=torch.float32)

    # Check shape
    assert pred.shape == gt.shape, f"got {pred.shape} and {gt.shape}"

    # Compute SAE
    sae = torch.sum(torch.abs(pred - gt)).item()
    return sae
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RoadBEV: Road Surface Reconstruction in Bird\'s Eye View')
    parser.add_argument('--dataset', help='dataset to use: add it to wandb runs')
    parser.add_argument('--stereo', action='store_true', help='if yes, use RoadBEV-stereo; otherwise, RoadBEV-mono')
    parser.add_argument('--cla_res', type=float, default=0.5, help='class resolution for elevation classification')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='maximum learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--logdir', default='./checkpoints/', help='the directory to save logs and checkpoints')
    parser.add_argument('--loadckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--summary_freq', type=int, default=40, help='summary_freq')
    parser.add_argument('--seed', type=int, default=307, metavar='S', help='random seed')
    parser.add_argument('--regression', action='store_true', help='regression or classification')
    parser.add_argument('--backbone',default='efficientnet', help='Use DepthAnything3 backbone or EfficientNet')
    parser.add_argument('--gradient_weight', type=float, default=0.01, help='weight for gradient loss in regression')
    parser.add_argument('--notes', type=str, default='', help='notes for wandb run')
    parser.add_argument('--scheduler', type=str, default='onecycle', help='type of lr scheduler to use: onecycle or reduceonplateau')
    parser.add_argument('--loss', type=str, default='L1', help='type of loss to use if regression: L1, gaussian NLL')
    parser.add_argument('--normalize', action='store_true', help='disable normalization')
    parser.add_argument('--name_run', type=str, default= ' ', help='give the name of the wandb run')
    parser.add_argument('--pred_head_dim', type=int, default=128, help='define the bottleneck between the transformer encoder and the CNN prediction head')



    # parse arguments, set seeds
    args = parser.parse_args()
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print("normalize", args.normalize)

    if args.stereo:
        args.down_scale = 2
        print('training RoadBEV-stereo!')
    else:
        args.down_scale = 4
        print('training RoadBEV-mono!')

    # dataset, dataloader
    if 'RSRD' in args.dataset:
        train_set = RSRD(training=True, stereo=args.stereo, down_scale=args.down_scale)
        test_set = RSRD(training=False, stereo=args.stereo, down_scale=args.down_scale)

    elif 'CARDSetV2Small' in args.dataset:
        test_set = CARDSetDatasetV2Smalldataset(root_dir='CARDSet/CARD_nice', mode='test', down_scale=args.down_scale)
        train_set = CARDSetDatasetV2Smalldataset(root_dir='CARDSet/CARD_nice', mode='train', down_scale=args.down_scale)
        args.batch_size = 1
        args.summary_freq = 1
        #args.epochs = 20

    elif 'CARDSet' in args.dataset:
        train_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/T7/cariad dataset/train_all_data_clean_NN_RHF.txt', mode='train', down_scale=args.down_scale)
        test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/T7/cariad dataset/val_all_data_clean_NN_RHF.txt', mode='test', down_scale=args.down_scale)
    
    
    
    else:
        print('unknown dataset!')
        exit(0)

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    
    #test_set = CARDSetDataset(root_dir='/media/T7/cariad dataset/Nardo', mode='test', down_scale=args.down_scale)
    test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    print('dataset size - train:%d, test:%d' % (len(train_set), len(test_set)))

    #get fixed sample for logging
    fixed_train_indices = [0, 1, 2]#, 2, 3]  
    fixed_eval_indices  = [0, 1, 2]#,6,7]

    device = "cuda"

    train_imgs_fixed, train_gt_fixed, train_mask_fixed, train_proj_fixed = get_fixed_samples(train_loader.dataset, fixed_train_indices, device)
    print(f"train mask fixed shape: {train_mask_fixed.shape}")

    eval_imgs_fixed, eval_gt_fixed, eval_mask_fixed, eval_proj_fixed = get_fixed_samples(test_loader.dataset, fixed_eval_indices, device)
    
    # model, optimizer
    ele_range = train_set.y_range
    voxel_ele_res = train_set.grid_res[1]
    num_grids = [train_set.num_grids_x, train_set.num_grids_y, train_set.num_grids_z]
    model = Elevation(args.stereo, num_grids, ele_range, args.cla_res, args.regression, args.backbone, args.normalize, args.pred_head_dim).cuda()
    early_stopping = EarlyStopping(patience=300, min_delta=0.001)
    print('num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    #print(model)
    model.train()
    
    if args.regression:
        if args.loss == 'L1':
            loss_func = LossReg(ele_range, args.normalize).cuda()
        else:
            loss_func = LossReg2(ele_range, args.gradient_weight).cuda()
    else:
        loss_func = MyLoss(ele_range, voxel_ele_res, args.cla_res).cuda()
    metric = Metric(ele_range, train_set.num_grids_z, distance_wise=False)

    url = 'https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth'
    try:
        weights = load_state_dict_from_url(url, progress=True)
    except:
        print('please manually download pretrained weights at:', url)
        exit(0)

    weights_new = {}
    target_keys = ['features.0', 'features.1', 'features.2', 'features.3', 'features.4']
    for key, value in weights.items():
        if any(k in key for k in target_keys):
            weights_new[key.replace('features.', 'l')] = value
    model.feature_extraction.load_state_dict(weights_new, strict=False)

    if args.loadckpt is not None:
        # load the checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        model.load_state_dict(state_dict, strict=True)

    scaler = GradScaler()

    lr_encoder = 1e-5
    encoder_params = list(model.feature_extraction.parameters())
    decoder_params = [param for name, param in model.named_parameters() if 'feature_extraction' not in name]
    print(f"number of decoder parameters: {sum(p.numel() for p in decoder_params)} vs number of parameter{sum(p.numel() for p in model.parameters())} vs number of encoder param {sum(p.numel() for p in encoder_params)}")

    optimizer = optim.AdamW([{"params":decoder_params, "lr":args.lr, "betas":(0.9, 0.999), "weight_decay":1e-4},
                             {"params":encoder_params, "lr":lr_encoder, "betas":(0.9, 0.999), "weight_decay":1e-4}])

    if args.scheduler == 'reduceonplaeau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    else: 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, pct_start=0.02,
                                                        three_phase=False,
                                                        div_factor=20, anneal_strategy='linear',
                                                        steps_per_epoch=len(train_loader))

    # logging
    args.logdir = os.path.join(args.logdir, datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    print('logging dir:', args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    # shutil.copy('./utils/dataset.py', os.path.join(args.logdir, 'dataset.py'))
    shutil.copy('./models/model.py', os.path.join(args.logdir, 'model.py'))
    shutil.copy('./models/efficientnet.py', os.path.join(args.logdir, 'efficientnet.py'))
    shutil.copy('./models/ele_head.py', os.path.join(args.logdir, 'ele_head.py'))
    shutil.copy('./models/patch2feature.py', os.path.join(args.logdir, 'patch2feature.py'))
    # shutil.copy('train.py', os.path.join(args.logdir, 'train.py'))
    log_file = open(os.path.join(args.logdir, 'log.txt'), 'a')

    if args.regression:
        train_regression()
    else:
        train()



