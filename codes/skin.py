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
from glob import glob
import numpy
import argparse
import os
import sys
import gc
import random
from utils.import_tflite_model import tf_inference
from utils.duattack_skin import Attack
from utils import metric_skin
from utils.read_data_skin import get_data, CustomDataset

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


def find_unique_class(samples, labels, n=1000):
    data_n = torch.zeros_like(samples).to(device)
    label_n = torch.zeros_like(labels).to(device)
    # for i in range(1500):
    #     for j in range(1500):
    #         temp = (label_n - labels[j]).abs()
    #         if (temp == 0).nonzero().numel() == 0:
    #             data_n[i] = samples[j]
    #             label_n[i] = labels[j]
    for i in range(5):
        sign = 0
        for j in range(100):
            while(1):
                sign += 1
                if labels[sign - 1] == i:
                    label_n[j + i * 100] = labels[sign - 1]
                    data_n[j + i * 100] = samples[sign - 1]
                    # print(i)
                    break

    # print(label_n[0:n])
    return data_n[0:n], label_n[0:n]


def parse_args():

    parser = argparse.ArgumentParser(description='Test on Fruit')
    parser.add_argument('--root', dest='root', default='~/rm46/dataset', type=str)
    parser.add_argument('--model', dest='model', default='./models/mobile_models/skin_cancer.lite', type=str)
    parser.add_argument('--nw', dest='nw', default=4, type=int)
    parser.add_argument('--bs', dest='bs', default=400, type=int)
    parser.add_argument('--tar_cls', dest='tar_cls', default=1, type=int)
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
    sys.stdout = Logger('skin_' + str(args.dist) + '_' + str(args.version) + '.log', sys.stdout)
    print('Called with args:')
    print(args)

    # [1] Dataset
    # train_path = os.path.join(args.root, 'fruit/train/')
    # test_path = os.path.join(args.root, 'fruit/test/')

    base_dir = '/fs03/rm46/dataset/HAM10000/All/'
    all_image_path = glob(os.path.join(base_dir, '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    df_train, df_val = get_data(base_dir, imageid_path_dict)

    img_size = 224
    clipmax = 1.
    clipmin = -1.
    num_classes = 5
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    # transforms = transforms.Compose([transforms.Resize(256),
    #                                  transforms.CenterCrop(224),
    #                                  transforms.ToTensor(),
    #                                  normalize,
    #                                  ])
    train_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor(),
                                        normalize
                                        ])
    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor(),
                                        normalize
                                        ])
    training_set = CustomDataset(df_train, transform=train_transform)
    if args.tar:
        bs = 50
        print("bs:", bs)
    else:
        bs = 1000
    trainloader = DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=args.nw)
    # Same for the validation set:
    test_set = CustomDataset(df_val, transform=train_transform)
    testloader = DataLoader(test_set, batch_size=256, shuffle=True, num_workers=args.nw)

    # Net
    load_dir = args.model
    model = tf.lite.Interpreter(model_path=load_dir)
    tf_model = tf_inference(model)

    # [3] Test original results
    ori_total = 0.
    ori_correct = 0.
    for i, data in enumerate(testloader, 0):
        ori_imgs, ori_labels = data
        # print(ori_labels)
        ori_imgs = ori_imgs.permute(0, 2, 3, 1)
        outputs = tf_model.query(ori_imgs, expand=99, dtype='float32')
        _, predict = torch.max(outputs, 1)
        ori_total += ori_imgs.size(0)
        ori_correct += ((predict-1) == ori_labels).sum()
    acc = (100. * ori_correct.float() / ori_total)
    print('original accuracy:', acc)

    # [4] Compute the UAP
    DUAttack = Attack(imgsize=img_size, net=tf_model,
                      clip_min=clipmin, clip_max=clipmax, mu=args.eps)
    adv_mask = torch.zeros(1, 3, img_size, img_size).to(device)
    # mat = loadmat('./mask/skin_10.0.mat')['A']
    # adv_mask = torch.from_numpy(mat).cuda()
    print('computing the UAP by DUAttack for skin')
    for i, data in enumerate(trainloader, 0):
        print("batch:", i)
        ori_img, ori_label = data
        # print('ori_label:', ori_label)
        ori_img = ori_img.to(device)
        ori_label = ori_label.to(device)
        bs = ori_img.size()[0]

        if args.tar:
            tar_labeln = torch.ones(ori_img.size(0)) * args.tar_cls
            ori_labeln = tar_labeln.long().to(device)
        else:
            ori_labeln = ori_label

        untar_ori, tar_ori = metric_skin.pred(bs=bs, imgs=ori_img, labels=ori_label,
                                               tar_labels=ori_labeln, net=tf_model)

        # generate adv_imgs
        if args.tar:
            robust_ori = tar_ori
            ori_robust = ori_img[robust_ori]
        else:
            robust_ori = untar_ori
            ori_robust = ori_img[robust_ori]
        print("training size:", ori_robust.size(0))
        print("training size:", ori_labeln[robust_ori].size(0))

        # ori_imgn, ori_labeln = find_unique_class(samples=ori_robust, labels=ori_label, n=500)
        # print(ori_labeln)
        # generate UAP

        adv_mask_tmp = DUAttack.DD_label_m1(x=ori_robust, mask=adv_mask,
                                            y=ori_labeln[robust_ori], targeted=args.tar,
                                            eps=args.eps, iteration=args.iter,
                                            eps_attenuation=args.attenuation,
                                            dist=args.dist, rand=args.rand)
        adv_mask = adv_mask_tmp
        print('adv_mask: ', torch.norm(adv_mask).item())
    # delete cache
        if args.tar:
            mask_path1 = os.path.join('./mask', 'skin_tar_{}.jpg'.format(args.dist))
            mask_path = os.path.join('./mask', 'skin_tar{}_50.mat'.format(args.dist))
        else:
            mask_path1 = os.path.join('./mask', 'skin_{}.jpg'.format(args.dist))
            mask_path = os.path.join('./mask', 'skin_{}.mat'.format(args.dist))
        # mask_path1 = os.path.join('./mask', 'skin_{}.jpg'.format(args.dist))
        save_image(adv_mask_tmp, mask_path1)
        # mask_path = os.path.join('./mask', 'skin_{}.mat'.format(args.dist))
        savemat(mask_path, {'A':adv_mask_tmp.cpu().numpy()})
        break
    del ori_img, ori_label
    torch.cuda.empty_cache()
    gc.collect()

    # mat = loadmat('./mask_test/skin_tar17.0.mat')['A']
    # adv_mask = torch.from_numpy(mat).cuda()
    # [5] Validate the performance of UAP
    correct = 0.
    total = 0.
    lf_dist = 0.
    cnt = 0.
    d_cnt = 0.
    image_saver = 0
    for i, data in enumerate(testloader, 0):

        # original data
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        # target labels
        bs = imgs.size()[0]
        tar_labels = torch.ones(bs) * args.tar_cls
        tar_labels = tar_labels.long().to(device)
        # tar_labels = torch.randint(0, num_classes-1, (bs,))
        # tar_labels = tar_labels.long().to(device)

        # original output
        untar_ori, tar_ori = metric_skin.pred(bs=bs, imgs=imgs, labels=labels,
                                               tar_labels=tar_labels, net=tf_model)

        # generate adv_imgs
        if args.tar:
            robust_ori = tar_ori
            ori_robust = imgs[robust_ori]
        else:
            robust_ori = untar_ori
            ori_robust = imgs[robust_ori]
        adv_imgs = (ori_robust + adv_mask).clamp(clipmin, clipmax)
        for k in range(ori_robust.size(0)):
            show_adv = adv_imgs[k] / 2.0 + 0.5
            show_img = ori_robust[k] / 2.0 + 0.5
            if args.tar:
                mask_path1 = os.path.join('./outputs', 'skin_adv_tar_{}.jpg'.format(image_saver))
                mask_path2 = os.path.join('./outputs', 'skin_ori_tar_{}.jpg'.format(image_saver))
            else:
                mask_path1 = os.path.join('./outputs', 'skin_adv_{}.jpg'.format(image_saver))
                mask_path2 = os.path.join('./outputs', 'skin_ori_{}.jpg'.format(image_saver))
            save_image(show_adv, mask_path1)
            save_image(show_img, mask_path2)
            image_saver += 1

        # compute attack distance for this batch
        bs_robust = ori_robust.size(0)
        cnt = cnt + bs_robust
        lf_norm = metric_skin.record_norm(adv_img=adv_imgs, ori_img=ori_robust,
                                           bs=bs_robust, dist='l2')
        # adversarial output
        untar_adv, tar_adv = metric_skin.pred(bs=bs_robust, imgs=adv_imgs,
                                               labels=labels[robust_ori],
                                               tar_labels=tar_labels[robust_ori],
                                               net=tf_model)
        # delete cache
        torch.cuda.empty_cache()
        gc.collect()

        # compute asr and distance
        correct, lf_dist = metric_skin.record_tmp(correct, lf_dist, lf_norm,
                                                   untar_adv, tar_adv,
                                                   dist='l2', target=args.tar)

    print('Total imgs for validation: ', cnt)
    asr = 100. - 100. * correct / cnt
    if args.tar is False:
        print('Untarget ASR on net: %.2f %%' % (asr))
    else:
        print('Target ASR on net: %.2f %%' % (asr))
    print('Attack Distance: %.2f' % (lf_dist / (cnt - correct)))
