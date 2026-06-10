import torch
import torch.nn.functional as F
import numpy as np
from manip_density_detect import density_to_boxes

def total_variation(x):
    # x: [B,1,H,W]
    return (x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean() + (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()

def compute_loss(pred_den, pred_msk_logit, gt_den, gt_msk, lam_msk=1.0, lam_tv=0.05):
    # density regression: robust
    loss_den = F.smooth_l1_loss(pred_den, gt_den)

    # mask supervision
    loss_msk = F.binary_cross_entropy_with_logits(pred_msk_logit, gt_msk)

    # smoothness on density
    loss_tv  = total_variation(pred_den)

    loss = loss_den + lam_msk*loss_msk + lam_tv*loss_tv
    return loss, {"den": float(loss_den.item()), "msk": float(loss_msk.item()), "tv": float(loss_tv.item())}

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    A=(a[2]-a[0])*(a[3]-a[1])
    B=(b[2]-b[0])*(b[3]-b[1])
    return inter/(A+B-inter+1e-6)

def eval_loader(model, loader, device, thr=0.5, min_area=20, iou_thr=0.5):
    model.eval()
    TP=FP=FN=0
    ious=[]
    abs_count_err=[]
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].to(device, non_blocking=True)
            gt_boxes_list = batch["boxes"]

            pred_den, pred_msk_logit = model(imgs)
            pd = pred_den.detach().cpu().numpy()  # [B,1,H,W]

            for i in range(pd.shape[0]):
                d = pd[i,0]
                d01 = d

                pred_boxes = density_to_boxes(d01, thr=thr, min_area=min_area)
                gt_boxes = gt_boxes_list[i]

                abs_count_err.append(abs(len(pred_boxes) - len(gt_boxes)))

                matched=set()
                for p in pred_boxes:
                    ok=False
                    best_iou=0.0
                    best_j=-1
                    for j,g in enumerate(gt_boxes):
                        if j in matched:
                            continue
                        v = iou(p,g)
                        if v > best_iou:
                            best_iou=v; best_j=j
                    if best_iou >= iou_thr:
                        TP += 1
                        matched.add(best_j)
                        ok=True
                        ious.append(best_iou)
                    if not ok:
                        FP += 1
                FN += (len(gt_boxes) - len(matched))

    prec = TP/(TP+FP+1e-6)
    rec  = TP/(TP+FN+1e-6)
    f1   = 2*prec*rec/(prec+rec+1e-6)
    map50_like = prec
    miou = float(np.mean(ious)) if len(ious)>0 else 0.0
    count_mae = float(np.mean(abs_count_err)) if len(abs_count_err)>0 else 0.0

    return {
        "Precision@0.5": float(prec),
        "Recall@0.5": float(rec),
        "F1@0.5": float(f1),
        "mAP50_like": float(map50_like),
        "mIoU_matched": miou,
        "Count_MAE": count_mae,
        "TP": int(TP), "FP": int(FP), "FN": int(FN),
    }
