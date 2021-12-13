import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gc
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def myeyematrix(eye_mt, k, cols):
    tmp = np.eye(cols)
    for i in range(cols - k):
        tmp[:, i+k] = eye_mt[:, i]
    for j in range(k):
        tmp[:, j] = eye_mt[:, cols-k+j]
    return tmp

def mean_square_distance(x1, x2, min_, max_):
    return np.mean((x1 - x2).cpu().numpy() ** 2) / ((max_ - min_) ** 2 + 1e-8)


class Attack(object):
    """
    DUAttack
    """
    def __init__(self, imgsize, net, clip_min, clip_max, mu=0.01, criterion=None, dtype='float32'):
        self.net = net
        self.dtype = dtype
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.clipmin = clip_min
        self.clipmax = clip_max
        self.imgsize = imgsize
        self.mu = mu
        self.eye_ori = np.eye(self.imgsize)

    # label-based with momentum
    def DD_label_m1(self, x, mask, y=None, targeted=False,
                    eps=0.2, iteration=1000, eps_attenuation=1500.0, dist=1., rand=False):

        # init some variables
        record_l = 0
        record_r = 0
        record = []
        mask_adv = mask
        x_adv = x
        bs = x.size()[0]
        img_ch = x.size()[1]
        flag = 0
        # recored the history
        momentum = torch.zeros(1, img_ch, self.imgsize, self.imgsize).to(device)
        least_remaining = 10000.0
        no_change = 0
        attenuation = 1.0
        # start
        for i in range(iteration):
            eps_final = eps * math.pow(0.5, i // eps_attenuation) * attenuation
            print("iteration:", i, "eps:", eps * math.pow(0.5, i // eps_attenuation))
            l_remaining = torch.zeros(bs)
            r_remaining = torch.zeros(bs)

            # 【1】select the c-th channel and the k-th eye matrix randomly
            # tmp_mask: the perturbation
            tmp = torch.zeros(1, img_ch, self.imgsize, self.imgsize).to(device)
            if flag  == 0:
                c = torch.randint(0, img_ch, (1,)).to(device)
                k = torch.randint(0, self.imgsize, (1,)).to(device)
            if rand is False:
                eye_mask = myeyematrix(self.eye_ori, k, self.imgsize)
                tmp[:, c, :, :] = torch.tensor(eye_mask).float().to(device)
            else:
                # mask under the random selection
                # random_mask = random.sample(range(0, self.imgsize*self.imgsize),
                #                             self.imgsize*self.imgsize)
                random_mask = random.sample(range(0, self.imgsize*self.imgsize),
                                                  128)
                tmp_ch = torch.zeros(self.imgsize * self.imgsize).to(device)
                tmp_ch[random_mask] = 1.0
                # random_mask = torch.tensor(random_mask)
                tmp_ch = tmp_ch.view(self.imgsize, self.imgsize)
                tmp[:, c, :, :] = tmp_ch.float().to(device)

                # random_ch = torch.randint(0, img_ch, (1,))
                # random_a = torch.randint(0, self.imgsize, (1,))
                # random_b = torch.randint(0, self.imgsize, (1,))
                # tmp[:, random_ch, random_a, random_b] = 1.0
            # tmp_mask = Variable(tmp).to(device)
            tmp_mask = tmp.to(device)

            # 【2.0】subtract
            left_mask = mask_adv - tmp_mask * (eps_final + 0.9*momentum)
            # keep the lf distance unchanged
            # l2
            left_dist = torch.norm(left_mask)
            left_mask = (left_mask / (left_dist + 1e-8)) * dist
            # # linf
            # left_mask = torch.sign(left_mask) * torch.min(abs(left_mask), (torch.ones(1)*dist).to(device))
            # get the adversarial output
            left_adv = (x + left_mask).clamp(self.clipmin, self.clipmax)
            with torch.no_grad():
                left_in = left_adv.permute(0, 2, 3, 1)
                left_out = self.net.query(left_in, expand=99, dtype='float32')
                left_out = left_out.to(device)
                _, left_preds = torch.max(left_out, 1)
                # left_preds = left_preds.to(device)
            # check
            if targeted:
                l_remaining = left_preds.ne(y)
                l_loss = nn.functional.cross_entropy(left_out, y, reduction='sum')
            else:
                l_remaining = left_preds.eq(y)
                l_loss = nn.functional.cross_entropy(left_out, y, reduction='sum') * -1.0
            # print("left preds:", left_preds)
            # if all images are misclassified, then break
            if l_remaining.sum() == 0:
                record.append((i,1))
                record_l += 1
                x_adv = left_adv
                mask_adv = (x_adv[0] - x[0]).unsqueeze(0)
                break

            torch.cuda.empty_cache()
            gc.collect()

            # 【2.1】add
            right_mask = mask_adv + (eps_final + 0.9*momentum)*tmp_mask
            # keep the lf distance unchanged
            # l2
            right_dist = torch.norm(right_mask)
            right_mask = (right_mask / (right_dist + 1e-8)) * dist
            # # linf
            # right_mask = torch.sign(right_mask) * torch.min(abs(right_mask), (torch.ones(1)*dist).to(device))
            # get the adversarial output
            right_adv = (x + right_mask).clamp(self.clipmin, self.clipmax)
            with torch.no_grad():
                right_in = right_adv.permute(0, 2, 3, 1)
                right_out = self.net.query(right_in, expand=99, dtype='float32')
                right_out = right_out.to(device)
                _, right_preds = torch.max(right_out, 1)
            # check
            if targeted:
                r_remaining = right_preds.ne(y)
                r_loss = nn.functional.cross_entropy(right_out, y, reduction='sum')
            else:
                r_remaining = right_preds.eq(y)
                r_loss = nn.functional.cross_entropy(right_out, y, reduction='sum') * -1.0
            # print("right preds:", right_preds)
            # if all images are misclassified, then break
            if r_remaining.sum() == 0:
                record.append((i,2))
                record_r += 1
                x_adv = right_adv
                mask_adv = (x_adv[0] - x[0]).unsqueeze(0)
                break

            # if least_remaining <= r_remaining.sum() and least_remaining <= l_remaining.sum():
            if least_remaining <= r_loss and least_remaining <= l_loss:
                no_change += 1
                attenuation = math.pow(0.95, no_change)
                print("no change")
                print("current attenuation：", attenuation)
            else:
                no_change = 0
                attenuation = 1.0
            # 【3】compare
            # if add has less improve than subtract
            # if l_remaining.sum() > r_remaining.sum():
            if l_loss > r_loss:
                record.append((i,2))
                record_r += 1
                x_adv = right_adv
                mask_adv = (x_adv[0] - x[0]).unsqueeze(0)
                # momentum += tmp_mask*eps
                momentum = 0                                    # test
                print("+unsuccess samples:", r_remaining.sum())
                print("-loss:", r_loss)
                # least_remaining = r_remaining.sum()
                least_remaining = r_loss
                # no_change = 0
                # attenuation = 1.0
            # elif r_remaining.sum() >= l_remaining.sum():
            elif r_loss >= l_loss:
                record.append((i,1))
                record_l += 1
                x_adv = left_adv
                mask_adv = (x_adv[0] - x[0]).unsqueeze(0)
                # momentum -= tmp_mask*eps
                momentum = 0                                    # test
                print("-unsuccess samples:", l_remaining.sum())
                print("-loss:", l_loss)
                # least_remaining = l_remaining.sum()
                least_remaining = l_loss
                # temp_adv = x_adv
                # no_change = 0
                # attenuation = 1.0
            flag = 0

            # print('DD_label_Iter %d: %.2f' % (i, torch.norm(mask_adv)))
            # print('DD_label_Iter %d: %.2f' % (i, mask_adv.max().item()))
            # print('record_r and record_l: %d, %d' % (record_r, record_l))
            # delete cache
            torch.cuda.empty_cache()
            gc.collect()

        return mask_adv
