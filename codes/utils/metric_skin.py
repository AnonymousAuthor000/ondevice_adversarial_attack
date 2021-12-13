import torch
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pred(bs, imgs, labels, tar_labels, net):
    tar_vec = torch.zeros(bs)
    untar_vec = torch.zeros(bs)
    with torch.no_grad():
        imgs = imgs.permute(0, 2, 3, 1)
        out = net.query(imgs, expand=99, dtype='float32')
        _, pred = torch.max(out, 1)
    pred = pred.to(device) - 1
    untar_vec = pred.eq(labels)
    tar_vec = pred.ne(tar_labels)
    # delete cache
    torch.cuda.empty_cache()
    gc.collect()
    return untar_vec, tar_vec

def compute_dist(bs, vec_adv, vec_ori, lf_norm):
    final_vec = torch.zeros(bs)
    lf_tmp = torch.zeros(bs)
    correct = 0.
    total = 0.
    final_vec = torch.add(vec_adv.float(), vec_ori.float())
    correct = (final_vec == 2).sum().float()
    total = (vec_ori == 1).sum()
    if correct == total:
        lf_dist = 0.
    else:
        lf_ori_correct = lf_norm.cpu() * vec_ori.cpu().float()
        lf_adv_correct = lf_norm.cpu() * (final_vec == 2).cpu().float()
        lf_dist = ((lf_ori_correct - lf_adv_correct).sum()) / (
                   (lf_ori_correct - lf_adv_correct).ne(lf_tmp).sum().float())
    return correct, total, lf_dist

def record_norm(adv_img, ori_img, bs, dist):
    perturb = (adv_img - ori_img) * 0.5 * 255
    if dist == 'l2':
        lf_norm = perturb.view(bs, -1).norm(2, 1)
    elif dist == 'l1':
        lf_norm = perturb.view(bs, -1).abs().sum(dim=1)
    else: # linf
        lf_norm, _ = perturb.view(bs, -1).abs().max(dim=1)
    return lf_norm

def record_tmp(correct, lf_dist, lf_norm, untar_adv, tar_adv, dist, target):
    if target is False:
        # untarget
        correct += untar_adv.sum().item()
        if dist == 'linf':
            if untar_adv.size(0) == 1:
                lf_tmp = (lf_norm * untar_adv.logical_not()).item()
            else:
                lf_tmp = (lf_norm * untar_adv.logical_not()).max().item()
            if lf_tmp > lf_dist:
                lf_dist = lf_tmp
        else:
            lf_dist += (lf_norm * untar_adv.logical_not()).sum().item()
    else:
        # target
        correct += tar_adv.sum().item()
        if dist == 'linf':
            if tar_adv.size(0) == 1:
                lf_tmp = (lf_norm * tar_adv.logical_not()).item()
            else:
                lf_tmp = (lf_norm * tar_adv.logical_not()).max().item()
            if lf_tmp > lf_dist:
                lf_dist = lf_tmp
        else:
            lf_dist += (lf_norm * tar_adv.logical_not()).sum().item()

    return correct, lf_dist
