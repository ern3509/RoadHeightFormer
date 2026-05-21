"""
Debug script to verify if loss_func output matches abs_err from metrics.
Tests on identical train/test loaders (both using training data).
"""
import torch
import argparse
from torch.utils.data import DataLoader
from utils.dataset import RSRD
from models.model import Elevation
from models.loss import LossReg
from utils.metric import Metric
from cardset.dataset import CARDSetDataset, CARDSetDatasetV2Smalldataset

def compare_loss_vs_metric(model, loss_func, metric, train_loader, test_loader, num_batches=5):
    """
    Compare loss output with abs_err from metrics on identical data
    """
    print("=" * 80)
    print("COMPARING LOSS vs ABS_ERR (Metric)")
    print("=" * 80)
    
    model.eval()
    
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)
    
    for batch_idx in range(num_batches):
        print(f"\n{'='*80}")
        print(f"Batch {batch_idx + 1}/{num_batches}")
        print(f"{'='*80}")
        
        try:
            # Get samples from train and test loaders
            train_sample = next(train_iter)
            test_sample = next(test_iter)
            
            # Parse train sample
            if len(train_sample) == 5:
                imgs_left_train, ele_gt_train, ele_mask_train, proj_idx_train, _ = train_sample
                imgs_right_train = None
            else:
                imgs_left_train, imgs_right_train, ele_gt_train, ele_mask_train, proj_idx_train, _, _ = train_sample
            
            # Parse test sample
            if len(test_sample) == 5:
                imgs_left_test, ele_gt_test, ele_mask_test, proj_idx_test, _ = test_sample
                imgs_right_test = None
            else:
                imgs_left_test, imgs_right_test, ele_gt_test, ele_mask_test, proj_idx_test, _, _ = test_sample
            
            # Move to cuda
            imgs_left_train = imgs_left_train.cuda()
            ele_gt_train = ele_gt_train.cuda()
            ele_mask_train = ele_mask_train.cuda()
            proj_idx_train = proj_idx_train.cuda()
            
            imgs_left_test = imgs_left_test.cuda()
            ele_gt_test = ele_gt_test.cuda()
            ele_mask_test = ele_mask_test.cuda()
            proj_idx_test = proj_idx_test.cuda()
            
            # Forward pass
            with torch.no_grad():
                if imgs_right_train is not None:
                    imgs_right_train = imgs_right_train.cuda()
                    ele_pred_train = model(imgs_left_train, proj_idx_train, imgs_right_train)
                else:
                    ele_pred_train = model(imgs_left_train, proj_idx_train)
                
                if imgs_right_test is not None:
                    imgs_right_test = imgs_right_test.cuda()
                    ele_pred_test = model(imgs_left_test, proj_idx_test, imgs_right_test)
                else:
                    ele_pred_test = model(imgs_left_test, proj_idx_test)
            
            # Check if loaders are truly identical (shapes and values)
            print(f"\n>>> LOADER COMPARISON:")
            print(f"Train batch shape: {imgs_left_train.shape}")
            print(f"Test batch shape:  {imgs_left_test.shape}")
            print(f"Train images equal to test images: {torch.allclose(imgs_left_train, imgs_left_test)}")
            print(f"Train GT equal to test GT: {torch.allclose(ele_gt_train, ele_gt_test)}")
            print(f"Train mask equal to test mask: {torch.equal(ele_mask_train, ele_mask_test)}")
            
            # For each sample in batch, compare loss with metric
            batch_size_train = ele_pred_train.shape[0]
            batch_size_test = ele_pred_test.shape[0]
            
            for sample_idx in range(min(batch_size_train, batch_size_test, 2)):  # Compare first 2 samples
                print(f"\n--- Sample {sample_idx + 1} in Batch ---")
                
                # Extract single sample
                pred_train = ele_pred_train[sample_idx:sample_idx+1]
                gt_train = ele_gt_train[sample_idx:sample_idx+1]
                mask_train = ele_mask_train[sample_idx:sample_idx+1]
                
                pred_test = ele_pred_test[sample_idx:sample_idx+1]
                gt_test = ele_gt_test[sample_idx:sample_idx+1]
                mask_test = ele_mask_test[sample_idx:sample_idx+1]
                
                # Compute loss (from train loader)
                loss_value = loss_func(pred_train, gt_train, mask_train)
                print(f"Loss from loss_func: {loss_value.item():.6f}")
                
                # Compute loss from test loader (should be identical)
                loss_value_test = loss_func(pred_test, gt_test, mask_test)
                print(f"Loss from test:      {loss_value_test.item():.6f}")
                print(f"Losses equal: {torch.allclose(loss_value, loss_value_test)}")
                
                # Compute metric abs_err (from metric compute_values_rhf)
                metric.clear()
                metric.compute(pred_train, gt_train, mask_train)
                metric_vals = metric.get_metric()
                abs_err = metric_vals[0][0]  # First value is abs_err
                
                print(f"\nAbs_err from metric: {abs_err:.6f}")
                
                # Manual abs_err calculation
                pred_masked = pred_train[mask_train]
                gt_masked = gt_train[mask_train]
                manual_abs_err = torch.mean(torch.abs(gt_masked - pred_masked)).item()
                print(f"Manual abs_err calc: {manual_abs_err:.6f}")
                
                # Compare loss with abs_err
                print(f"\n>>> COMPARISON: Loss ({loss_value.item():.6f}) vs Abs_Err ({abs_err:.6f})")
                diff = abs(loss_value.item() - abs_err)
                print(f"Difference: {diff:.6f}")
                print(f"Match (diff < 1e-5): {diff < 1e-5}")
                
        except StopIteration:
            print(f"Reached end of loaders at batch {batch_idx}")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug Loss vs Metric')
    parser.add_argument('--loadckpt', default='./checkpoints/20240407064559/checkpoint_epoch50_007500.ckpt', 
                        help='load the weights from a specific checkpoint')
    parser.add_argument('--dataset', default='RSRD', help='dataset: RSRD, CARDSet, CARDSetSmall, CARDSetV2Small')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--stereo', action='store_true', help='use stereo')
    parser.add_argument('--down_scale', type=int, default=4, help='down scale')
    parser.add_argument('--regression', action='store_true', help='regression or classification')
    parser.add_argument('--backbone', default='efficientnet', help='backbone: efficientnet, dino')
    parser.add_argument('--normalize', action='store_true', help='normalize heights to [-1, 1]')
    parser.add_argument('--pred_head_dim', type=int, default=128, help='prediction head dim')
    parser.add_argument('--preprocessed', action='store_true', help='use preprocessed data')
    parser.add_argument('--ele_range', type=float, default=0.2, help='elevation range in meters')
    parser.add_argument('--loss', default='L1', help='loss type: L1, MSE')
    
    args = parser.parse_args()
    
    # Setup device
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    print(f"\n>>> Loading dataset: {args.dataset}")
    if "CARDSet" == args.dataset:
        train_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', 
                                   split_file='/data/T7/cariad dataset/train_all_data_clean_NN_RHF.txt', 
                                   mode='train', down_scale=args.down_scale, preprocessed_data=args.preprocessed)
        test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', 
                                  split_file='/data/T7/cariad dataset/val_all_data_clean_NN_RHF.txt', 
                                  mode='test', down_scale=args.down_scale, preprocessed_data=args.preprocessed)
    elif 'CARDSetV2Small' == args.dataset:
        train_set = CARDSetDatasetV2Smalldataset(root_dir='CARDSet/CARD_nice', mode='train', down_scale=args.down_scale)
        test_set = CARDSetDatasetV2Smalldataset(root_dir='CARDSet/CARD_nice', mode='test', down_scale=args.down_scale)
    elif 'CARDSetSmall' == args.dataset:
        train_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', 
                                   split_file='/data/rhf/val_small_dataset.txt', 
                                   mode='train', down_scale=args.down_scale, preprocessed_data=args.preprocessed)
        test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', 
                                  split_file='/data/rhf/val_small_dataset.txt', 
                                  mode='test', down_scale=args.down_scale, preprocessed_data=args.preprocessed)
    elif 'RSRD' == args.dataset:
        train_set = RSRD(training=True, stereo=args.stereo, down_scale=args.down_scale)
        test_set = RSRD(training=False, stereo=args.stereo, down_scale=args.down_scale)
    else:
        print("Unknown dataset!")
        exit(0)
    
    # Create identical DataLoaders
    print(f">>> Creating identical DataLoaders (batch_size={args.batch_size})")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=False, num_workers=0, 
                              drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=0, 
                             drop_last=True, pin_memory=True)
    
    # Load model
    print(f">>> Loading model from: {args.loadckpt}")
    ele_range = train_set.y_range
    voxel_ele_res = train_set.grid_res[1]
    num_grids = [train_set.num_grids_x, train_set.num_grids_y, train_set.num_grids_z]
    
    model = Elevation(args.stereo, num_grids, ele_range, cla_res=0.5, regression=args.regression, 
                      backbone=args.backbone, normalize=args.normalize, pred_dim=args.pred_head_dim).cuda()
    
    if args.loadckpt and torch.cuda.is_available():
        checkpoint = torch.load(args.loadckpt)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    # Create loss and metric
    print(f">>> Creating loss function: {args.loss}")
    loss_func = LossReg(ele_range, normalize=args.normalize, type_of_loss=args.loss).cuda()
    
    metric = Metric(ele_range, train_set.num_grids_z, distance_wise=False)
    
    # Run comparison
    compare_loss_vs_metric(model, loss_func, metric, train_loader, test_loader, num_batches=5)
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
