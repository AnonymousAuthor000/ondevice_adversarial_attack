import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.utils.data.sampler as sp
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from scipy.io import savemat,loadmat
from PIL import Image

import numpy
import argparse
import os
import sys
import gc
import random
from utils.import_tflite_model import tf_inference
from utils.duattack_fruit import Attack
from utils import metric_fruit

seed = 1000
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass


def parse_args():

    parser = argparse.ArgumentParser(description='Test on Fruit')
    parser.add_argument('--root', dest='root', default='~/rm46/dataset', type=str)
    parser.add_argument('--model', dest='model', default='./models/mobile_models/fruit_graph.tflite', type=str)
    parser.add_argument('--nw', dest='nw', default=4, type=int)
    parser.add_argument('--bs', dest='bs', default=400, type=int)
    parser.add_argument('--tar_cls', dest='tar_cls', default=10, type=int)
    parser.add_argument('--tar', dest='tar', default=False, type=bool)
    parser.add_argument('--iter', dest='iter', default=1000, type=int)
    parser.add_argument('--eps', dest='eps', default=0.2, type=float)
    parser.add_argument('--attenuation', dest='attenuation', default=1500.0, type=float)
    parser.add_argument('--dist', dest='dist', default=20.0, type=float)
    parser.add_argument('--rand', dest='rand', default=False, type=bool)
    parser.add_argument('--version', dest='version', default=0, type=int) #1: rand
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    sys.stdout = Logger('fruit_' + str(args.dist) + '_' + str(args.version) + '.log', sys.stdout)
    print('Called with args:')
    print(args)

    # [1] Dataset
    train_path = os.path.join(args.root, 'fruit/train/')
    test_path = os.path.join(args.root, 'fruit/test/')
    img_size = 224
    clipmax = 1.
    clipmin = -1.
    num_classes = 40
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                     ])
    trainloader = DataLoader(ImageFolder(train_path, transform=transforms),
                             batch_size=args.bs,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=args.nw)
    testloader = DataLoader(ImageFolder(test_path, transform=transforms),
                             batch_size=args.bs,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=args.nw)


    # Net
    load_dir = args.model
    model = tf.lite.Interpreter(model_path=load_dir)
    tf_model = tf_inference(model)

    # [3] Test original results
    ori_total = 0.
    ori_correct = 0.
    cnt_test_size = 0
    for i, data in enumerate(testloader, 0):
        ori_imgs, ori_labels = data
        cnt_test_size += ori_labels.size()[0]
        ori_imgs = ori_imgs.permute(0, 2, 3, 1)
        outputs = tf_model.query(ori_imgs, expand=99, dtype='float32')
        _, predict = torch.max(outputs, 1)
        ori_total += ori_imgs.size(0)
        ori_correct += (predict == ori_labels).sum()
    acc = (100. * ori_correct.float() / ori_total)
    print('original accuracy:', acc)
    print('test size:', cnt_test_size)

    adv_mask = torch.zeros(1, 3, img_size, img_size).to(device)
    mask_path = os.path.join('./mask', 'fruit_zero.mat')
    savemat(mask_path, {'A':adv_mask.cpu().numpy()})

    # [4] Compute the UAP
    DUAttack = Attack(imgsize=img_size, net=tf_model,
                      clip_min=clipmin, clip_max=clipmax, mu=args.eps)
    adv_mask = torch.zeros(1, 3, img_size, img_size).to(device)
    print('computing the UAP by DUAttack for fruit')
    for i, data in enumerate(trainloader, 0):
        print("batch:", i)
        ori_img, ori_label = data
        # print('ori_label:', ori_label)
        ori_img = ori_img.to(device)
        ori_label = ori_label.to(device)
        # generate UAP
        if args.tar:
            tar_labeln = torch.ones(ori_img.size(0)) * args.tar_cls
            adv_labeln = tar_labeln.long().to(device)
        else:
            adv_labeln = ori_label
        adv_mask_tmp = DUAttack.DD_label_m1(x=ori_img, mask=adv_mask,
                                            y=adv_labeln, targeted=args.tar,
                                            eps=args.eps, iteration=args.iter,
                                            eps_attenuation=args.attenuation,
                                            dist=args.dist, rand=args.rand)
        adv_mask = adv_mask_tmp
        print('adv_mask: ', torch.norm(adv_mask).item())
    # delete cache
        if args.tar:
            mask_path1 = os.path.join('./mask', 'fruit_tar_{}.jpg'.format(args.dist))
            mask_path = os.path.join('./mask', 'fruit_tar_{}.mat'.format(args.dist))
        else:
            mask_path1 = os.path.join('./mask', 'fruit_{}.jpg'.format(args.bs))
            mask_path = os.path.join('./mask', 'fruit_{}.mat'.format(args.bs))
        # mask_path1 = os.path.join('./mask', 'fruit_{}.jpg'.format(args.dist))
        save_image(adv_mask_tmp, mask_path1)
        # mask_path = os.path.join('./mask', 'fruit_{}.mat'.format(args.dist))
        savemat(mask_path, {'A':adv_mask_tmp.cpu().numpy()})
    del ori_img, ori_label
    torch.cuda.empty_cache()
    gc.collect()

    # mat = loadmat('./mask_test/fruit_17.0_51.mat')['A']
    # adv_mask = torch.from_numpy(mat).cuda()
    # [5] Validate the performance of UAP
    correct = 0.
    total = 0.
    lf_dist = 0.
    cnt = 0.
    d_cnt = 0.
    for i, data in enumerate(testloader, 0):

        # original data
        imgs, labels = data
        imgs = imgs.to(device)
        # show_img = (imgs[0] + adv_mask) / 2.0 + 0.5
        # for k in range(15):
        #     show_adv = (imgs[k] + adv_mask) / 2.0 + 0.5
        #     show_img = imgs[k] / 2.0 + 0.5
        #     if args.tar:
        #         mask_path1 = os.path.join('./outputs', 'fruit_adv_tar_{}.jpg'.format(k))
        #         mask_path2 = os.path.join('./outputs', 'fruit_ori_tar_{}.jpg'.format(k))
        #     else:
        #         mask_path1 = os.path.join('./outputs', 'fruit_adv_{}.jpg'.format(k))
        #         mask_path2 = os.path.join('./outputs', 'fruit_ori_{}.jpg'.format(k))
        #     save_image(show_adv, mask_path1)
        #     save_image(show_img, mask_path2)
        labels = labels.to(device)

        # target labels
        bs = imgs.size()[0]
        tar_labels = torch.ones(bs) * args.tar_cls
        tar_labels = tar_labels.long().to(device)
        # tar_labels = torch.randint(0, num_classes-1, (bs,))
        # tar_labels = tar_labels.long().to(device)

        # original output
        untar_ori, tar_ori = metric_fruit.pred(bs=bs, imgs=imgs, labels=labels,
                                               tar_labels=tar_labels, net=tf_model)

        # generate adv_imgs
        if args.tar:
            robust_ori = tar_ori
            ori_robust = imgs[robust_ori]
        else:
            robust_ori = untar_ori
            ori_robust = imgs[robust_ori]
        adv_imgs = (ori_robust + adv_mask).clamp(clipmin, clipmax)

        # compute attack distance for this batch
        bs_robust = ori_robust.size(0)
        cnt = cnt + bs_robust
        lf_norm = metric_fruit.record_norm(adv_img=adv_imgs, ori_img=ori_robust,
                                           bs=bs_robust, dist='l2')
        # adversarial output
        untar_adv, tar_adv = metric_fruit.pred(bs=bs_robust, imgs=adv_imgs,
                                               labels=labels[robust_ori],
                                               tar_labels=tar_labels[robust_ori],
                                               net=tf_model)
        # delete cache
        torch.cuda.empty_cache()
        gc.collect()

        # compute asr and distance
        correct, lf_dist = metric_fruit.record_tmp(correct, lf_dist, lf_norm,
                                                   untar_adv, tar_adv,
                                                   dist='l2', target=args.tar)

    print('Total imgs for validation: ', cnt)
    asr = 100. - 100. * correct / cnt
    if args.tar is False:
        print('Untarget ASR on net: %.2f %%' % (asr))
    else:
        print('Target ASR on net: %.2f %%' % (asr))
    print('Attack Distance: %.2f' % (lf_dist / (cnt - correct)))
