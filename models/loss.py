import torch.nn.functional as F
import torch
from torch import nn

class MyLoss(nn.Module):
    def __init__(self, ele_range, voxel_ele_res, cla_res=1):
        super(MyLoss, self).__init__()
        self.ele_range = ele_range*100  # to cm
        if (self.ele_range*20) % (cla_res*10) != 0:
            print('The class interval is improper')
            exit()
        self.cla_res = cla_res   # in cm
        self.voxel_ele_res = voxel_ele_res*100  # in cm
        self.num_voxels_ele = int(self.ele_range*2 / self.voxel_ele_res)

        self.num_classes = int(2*self.ele_range/cla_res)
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def label2class(self, ele_gt):
        # ele_gt: [N,]
        assert ele_gt.numel() > 0
        class_label = torch.floor((ele_gt + self.ele_range) / self.cla_res).type(torch.long)
        class_label = self.num_classes - class_label - 1

        return class_label

    def forward(self, ele_pred, ele_gt, ele_mask):
        # ele_pred: [B, num_classes, H, W]  without softmax
        # ele_gt:   [B, H, W]
        # ele_mask: [B, H, W]

        ele_mask_roi = torch.logical_and(ele_gt > -self.ele_range, ele_gt < self.ele_range)
        ele_mask = torch.logical_and(ele_mask_roi, ele_mask)

        ele_pred = ele_pred.permute(0, 2, 3, 1)
        ele_pred = ele_pred[ele_mask, :]
        ele_gt = ele_gt[ele_mask]

        # class_voxel = self.label2class(ele_gt, 'voxel')
        # loss_voxel = self.loss_func1(voxel_prob, class_voxel)
        class_ele = self.label2class(ele_gt)
        loss_ele = self.loss_func(ele_pred, class_ele)

        return loss_ele

class LossReg(nn.Module):
    def __init__(self, ele_range, normalize=False):
        super(LossReg, self).__init__()
        self.ele_range = ele_range*100
        self.loss_func = nn.SmoothL1Loss(reduction='mean')
        self.normalize = normalize


    def forward(self, ele_pred, ele_gt, ele_mask):
        # ele_pred: [B, H, W]
        # ele_gt:   [B, H, W]
        # ele_mask: [B, H, W]
        #print("Erwannnn",ele_pred.shape, ele_gt.shape, ele_mask.shape)
        print("ele_pred is nan", torch.sum(torch.isnan(ele_pred)))
        ele_mask_roi = torch.logical_and(ele_gt > -1000, ele_gt < 1000)
        ele_mask = torch.logical_and(ele_mask_roi, ele_mask)
        ele_pred = ele_pred[ele_mask]
        print("Regression Loss:L1")
        #ele_pred = ele_pred[:, 0:1].squeeze(1)[ele_mask]
        ele_gt = ele_gt[ele_mask]
        gt_min = - self.ele_range
        gt_max = self.ele_range
        if self.normalize:
            print("normalized")
            pred_scaled = (ele_pred * (gt_max - gt_min) / 2) + ((gt_max + gt_min) / 2)

            assert(pred_scaled.shape == ele_pred.shape)
            loss = self.loss_func(pred_scaled, ele_gt)
        else:
            loss = self.loss_func(ele_pred, ele_gt)

        return loss
    
class GradientLoss(nn.Module):
    """
    Computes L1 loss between spatial gradients of prediction and ground truth.
    """

    def __init__(self):
        super().__init__()
    @staticmethod
    def gradient_x(img):
        return img[:, :, :, 1:] - img[:, :, :, :-1]

    @staticmethod
    def gradient_y(img):
        return img[:, :, 1:, :] - img[:, :, :-1, :]

    def forward(self, pred, gt, mask):
        """
        pred: (B, 1, H, W)
        gt:   (B, 1, H, W)
        mask: (B, 1, H, W) boolean
        """

        pred_dx = self.gradient_x(pred)
        pred_dy = self.gradient_y(pred)

        gt_dx = self.gradient_x(gt)
        gt_dy = self.gradient_y(gt)

        # Gradient masks (both neighboring pixels must be valid)
        mask_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]

        loss_x = torch.abs(pred_dx - gt_dx)[mask_dx].mean()
        loss_y = torch.abs(pred_dy - gt_dy)[mask_dy].mean()

        return loss_x + loss_y

class HeteroscedasticNLLLoss(nn.Module):
    """
    Gaussian negative log-likelihood loss with learned per-pixel variance.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        pred: (B, 2, H, W)
        gt:   (B, 1, H, W)
        mask: (B, 1, H, W) boolean
        """
        mean = pred[:, 0:1]
        var  = pred[:, 1:2]

        var = F.softplus(var) + self.eps

        nll = (gt - mean) ** 2 / (2.0 * var) + 0.5 * torch.log(var)

        return nll[mask].mean()
    
    
class LossReg2(nn.Module): #neg loglik + gradient loss
    def __init__(self, ele_range, gradient_weight=0.01):
        super(LossReg2, self).__init__()
        self.gradientloss = GradientLoss()
        self.nll = HeteroscedasticNLLLoss()
        self.ele_range = ele_range*100
        self.gradient_weight = gradient_weight
        self.l1loss = nn.L1Loss(reduction='mean')
    def forward(self, ele_pred, ele_gt, ele_mask):
        # ele_pred: [B, 2, H, W]  mean and variance
        # ele_gt:   [B, H, W]
        # ele_mask: [B, H, W]

        # Valid value range mask
        roi_mask = torch.logical_and((ele_gt > -self.ele_range),(ele_gt < self.ele_range))
        mask = torch.logical_and(roi_mask,ele_mask)
        mask = mask.unsqueeze(1)  # (B, 1, H, W)

        ele_gt = ele_gt.unsqueeze(1)

        loss_nll = self.nll(ele_pred, ele_gt, mask)
        loss_grad = self.gradientloss(ele_pred[:, 0:1], ele_gt, mask)

        return loss_nll + self.gradient_weight * loss_grad
