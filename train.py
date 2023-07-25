# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import argparse
import glob
import json
import random
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from datasets import WHUAugBox
from networks.GAN import Discriminator, Decoder, Segmenter
from networks.drn import drn_a_50
from utils.metrics import scores
from utils.util import get_cosine_schedule_with_warmup, deep_grabcut
from config import Config
from loss import SureRegionLoss, ContentLoss


def set_seed(seed=42):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_loss(sigma, loss):
    return 0.5 / (sigma ** 2) * loss + torch.log(1 + sigma ** 2)


@torch.no_grad()
def val(encoder, classifier, dataloader, cfg):
    print("Start Validation...")
    labels_true = []
    labels_pred = []

    encoder.eval()
    classifier.eval()
    for img_id, img_a, img_b, gt_mask, _, _ in tqdm(dataloader):
        img_a, img_b = img_a.to(cfg.device), img_b.to(cfg.device)
        gt_mask = gt_mask.to(cfg.device)

        a_features = encoder(img_a)
        b_features = encoder(img_b)
        label_pred = classifier(a_features, b_features)

        label_pred = (torch.sigmoid(label_pred) > 0.5).float()
        labels_pred.append(label_pred.cpu().numpy().squeeze())
        labels_true.append(gt_mask.cpu().numpy().squeeze())
            
    score = scores(labels_true, labels_pred, cfg.num_classes)
    return score


@torch.no_grad()
def test(encoder, segmentor, dataloader, cfg):
    encoder.eval()
    segmentor.eval()
    label_preds = []
    label_trues = []
    
    pbar = tqdm(dataloader, 
                bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| [{elapsed}|{rate_fmt}{postfix}]',
                desc='[Test] ')
    for img_id, img_a, img_b, gt_mask, _, _ in pbar:
        img_a, img_b = img_a.to(cfg.device), img_b.to(cfg.device)
        gt_mask = gt_mask.to(cfg.device)
        
        a_encoder_out = encoder(img_a)
        b_encoder_out = encoder(img_b)
        y_pred = segmentor(a_encoder_out, b_encoder_out)
        y_pred = (torch.sigmoid(y_pred) > 0.5).float()
        y_pred = y_pred.cpu().numpy().squeeze().astype(np.int)
        gt_mask = gt_mask.cpu().numpy().squeeze()
        gt_mask[gt_mask == 255] = 1
        label_preds.append(y_pred)
        label_trues.append(gt_mask)
        
    cls_iou = scores(label_trues, label_preds, 2)
    return cls_iou


@torch.no_grad()
def predict(encoder, segmenter, img_a, img_b):
    img_a, img_b = T.ToTensor()(img_a)[None], T.ToTensor()(img_b)[None, ...]
    img_a = torch.sub(torch.multiply(torch.tensor(2.), 
                                     torch.div(img_a, torch.tensor(255.))), torch.tensor(1.))
    img_b = torch.sub(torch.multiply(torch.tensor(2.), 
                                     torch.div(img_b, torch.tensor(255.))), torch.tensor(1.))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    segmenter.eval()
    
    a_encoder_out = encoder(img_a.to(device))
    b_encoder_out = encoder(img_b.to(device))
    a_decoder_out = segmenter(a_encoder_out[-1])
    b_decoder_out = segmenter(b_encoder_out[-1])
    a_decoder_out = a_decoder_out.cpu().numpy().squeeze()  # (3, 256, 256)
    b_decoder_out = b_decoder_out.cpu().numpy().squeeze()  # (3, 256, 256)
    a_decoder_out = a_decoder_out.reshape(3, 256 * 256).transpose().reshape(256, 256, 3)
    b_decoder_out = b_decoder_out.reshape(3, 256 * 256).transpose().reshape(256, 256, 3)

    return a_decoder_out, b_decoder_out


def main():
    set_seed(Config.seed)

    # -----------
    # Build Model
    # -----------

    encoder = drn_a_50(pretrained=True)
    decoder = Decoder(2048, 3)
    segmenter = Segmenter()
    discriminator_pre = Discriminator(Config.input_shape)
    encoder.to(Config.device)
    decoder.to(Config.device)
    segmenter.to(Config.device)
    discriminator_pre.to(Config.device)

    if Config.pretrained:
        print("Loading pretrained parameters......")
        encoder.load_state_dict(torch.load(Config.pretrained)['encoder'])
        decoder.load_state_dict(torch.load(Config.pretrained)['decoder'])
        segmenter.load_state_dict(torch.load(Config.pretrained)['segmenter'], strict=True)
        discriminator_pre.load_state_dict(torch.load(Config.pretrained)['discriminator_pre'])

    # -----------------
    # Loss
    # -----------------

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    sr_loss = SureRegionLoss()
    con_loss = ContentLoss()

    # -------------------------------------
    # Optimiser and scheduler configuration
    # -------------------------------------

    optimizer_E = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                    lr=Config.lr_encoder)
    optimizer_Decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()),
                                    lr=Config.lr_decoder)
    optimizer_Seg = torch.optim.Adam(filter(lambda p: p.requires_grad, segmenter.parameters()),
                                    lr=Config.lr_seg)
    optimizer_D2 = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator_pre.parameters()),
                                    lr=Config.lr_d2)

    scheduler_E = get_cosine_schedule_with_warmup(optimizer_E, Config.warm_steps, Config.iter_max * Config.epoch)
    scheduler_Decoder = get_cosine_schedule_with_warmup(optimizer_Decoder, Config.warm_steps, Config.iter_max * Config.epoch)
    scheduler_Seg = get_cosine_schedule_with_warmup(optimizer_Seg, Config.warm_steps, Config.iter_max * Config.epoch)
    scheduler_D2 = get_cosine_schedule_with_warmup(optimizer_D2, Config.warm_steps, Config.iter_max * Config.epoch)

    best_iou = 0.0

    train_dataset = WHUAugBox(root=Config.root,
                              split=Config.split_train,
                              ignore_label=Config.ignore_label,
                              mean_bgr=Config.mean_bgr)
    val_dataset = WHUAugBox(root=Config.root,
                            split=Config.split_val,
                            ignore_label=Config.ignore_label,
                            mean_bgr=Config.mean_bgr,
                            augment=False)
    train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=Config.batch_size,
                                  shuffle=True, num_workers=Config.num_worker,
                                  pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1, shuffle=False, num_workers=Config.num_worker,
                            pin_memory=True, drop_last=False)
    train_iter = iter(train_loader)

    for p_iter in range(0, Config.epoch):
        print(f"---------------------Epoch: {p_iter + 1} / {Config.epoch}---------------------\n")

        out_save_path = os.path.join(Config.save_path, f"epoch_{p_iter + 1}")
        os.makedirs(out_save_path, exist_ok=True)

        save_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = open(os.path.join(out_save_path, save_name + '.txt'), 'w')
        log_json = open(os.path.join(out_save_path, save_name + '.json'), 'w')
        log_file.write("pytorch version: {} cuda version: {}\n".format(torch.__version__, torch.version.cuda))
        log_file.flush()
        
        # -------------------------------------------
        # The order of training TICDN：
        # E -> Seg -> Decoder -> D2
        # -------------------------------------------

        for iteration in range(1, Config.iter_max + 1):

            try:
                img_id, img_a, img_b, gt_mask, bbox_mask, sure_mask = next(train_iter)
            except:
                train_iter = iter(train_loader)
                img_id, img_a, img_b, gt_mask, bbox_mask, sure_mask = next(train_iter)
            img_a, img_b = img_a.to(Config.device), img_b.to(Config.device)
            gt_mask = gt_mask.float().to(Config.device)
            bbox_mask = bbox_mask.float().to(Config.device)
            sure_mask = sure_mask.to(Config.device)

            # Adversarial ground truths
            valid = torch.ones((img_a.size(0), *discriminator_pre.output_shape), requires_grad=False,
                            device=gt_mask.device)
            fake = torch.zeros((img_a.size(0), *discriminator_pre.output_shape), requires_grad=False,
                            device=gt_mask.device)

            # -------------
            # train Encoder
            # -------------

            encoder.train(True)
            segmenter.train(True)
            decoder.train(True)
            optimizer_E.zero_grad()
            optimizer_Seg.zero_grad()
            optimizer_Decoder.zero_grad()

            # encoder out
            a_encoder_out = encoder(img_a)
            b_encoder_out = encoder(img_b)

            # decoder out
            decode_a_out = decoder(a_encoder_out[-1])
            decode_b_out = decoder(b_encoder_out[-1])

            # segmenter out
            seg_out = segmenter(a_encoder_out, b_encoder_out)

            # loss
            loss_adv_s = 0.5 * (mse_loss(discriminator_pre(decode_b_out), valid)
                                + mse_loss(discriminator_pre(decode_a_out), fake))
            loss_cont = 0.5 * (con_loss(img_a, decode_a_out) + con_loss(img_b, decode_b_out))
            loss_cycle = l1_loss(decode_a_out, img_a)
            loss_seg = sr_loss(seg_out, gt_mask, sure_mask)

            loss_encoder = loss_seg + 0.002 * loss_cont + loss_cycle + 0.1 * loss_adv_s

            loss_encoder.backward()
            optimizer_E.step()
            scheduler_E.step()
            optimizer_Seg.step()
            scheduler_Seg.step()
            optimizer_Decoder.step()
            scheduler_Decoder.step()

            # -------------
            # train D2
            # -------------

            discriminator_pre.train(True)
            optimizer_D2.zero_grad()

            # encoder out
            a_encoder_out = encoder(img_a)
            b_encoder_out = encoder(img_b)

            # decoder out
            decode_a_out = decoder(a_encoder_out[-1])
            decode_b_out = decoder(b_encoder_out[-1])

            loss_adv_s = 0.5 * (mse_loss(discriminator_pre(decode_a_out.detach()), valid) +
                                mse_loss(discriminator_pre(decode_b_out.detach()), fake))
            loss_D2 = loss_adv_s

            loss_D2.backward()
            optimizer_D2.step()
            scheduler_D2.step()

            # ---------------
            # save output
            # ---------------

            if iteration % Config.iter_save_txt == 0:
                log_str = "time: {}, iteration: {}, "\
                    "lr_encoder: {}, lr_decoder: {}, lr_seg: {}, lr_d2: {}, "\
                    "loss_cont: {}, loss_cycle: {}, loss_adv_s: {}, loss_D2: {}, "\
                    "loss_seg: {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), iteration,
                    optimizer_E.param_groups[0]["lr"], optimizer_Decoder.param_groups[0]["lr"],
                    optimizer_Seg.param_groups[0]["lr"], optimizer_D2.param_groups[0]["lr"],
                    0.002 * loss_cont.item(), loss_cycle.item(), 0.1 * loss_adv_s.item(), loss_D2.item(),
                    loss_seg.item()
                )
                print(log_str)
                log_file.write(log_str + '\n')
                log_file.flush()

                json_dict = {'iteration': iteration,
                             "loss_seg": loss_seg.item(),
                             "loss_cont": 0.002 * loss_cont.item(),
                             "loss_cycle": loss_cycle.item(),
                             "loss_adv_s": 0.1 * loss_adv_s.item(),
                             "loss_D2": loss_D2.item(),
                             }
                json_str = json.dumps(json_dict)
                log_json.write(json_str + '\n')
                log_json.flush()

            # Testing on validation set
            if iteration % Config.iter_val == 0:
                score = val(encoder, segmenter, val_loader, Config)
                val_iou = score['Class IoU'][1]
                log_str = "time: {}, iteration: {}, iou: {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), iteration, val_iou)
                print(log_str + '\n')
                log_file.write(log_str + '\n')
                log_file.write("\n")
                log_file.flush()

                json_dict = {'iteration': iteration, 'score': score}
                json_str = json.dumps(json_dict)
                log_json.write(json_str + '\n')
                log_json.flush()

                # save weights
                if val_iou > best_iou:
                    best_iou = val_iou
                    current_checkpoints = glob.glob(out_save_path + '/*.tar')
                    if len(current_checkpoints) > Config.save_max_num:
                        os.remove(current_checkpoints[0])
                    torch.save({
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'segmenter': segmenter.state_dict(),
                        'discriminator_pre': discriminator_pre.state_dict(),
                    }, os.path.join(out_save_path, 'iter_{}_{:.5f}.tar'.format(iteration, best_iou)))

        # Generate pseudo-labels and save
        print("End of training round, start updating pseudo-labels.......")
        if p_iter == 0:
            train_dataset = WHUAugBox(root=Config.root,
                                    split=Config.split_train,
                                    ignore_label=Config.ignore_label,
                                    mean_bgr=Config.mean_bgr,
                                    augment=False)
        else:
            train_dataset = WHUAugBox(pickle_file=os.path.join(Config.save_path, f"epoch_{p_iter}", "pseudo_label_unmask.pkl"),
                                    root=Config.root,
                                    split=Config.split_train,
                                    ignore_label=Config.ignore_label,
                                    mean_bgr=Config.mean_bgr,
                                    augment=False)
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=Config.p_batch_size,
                                shuffle=False, num_workers=Config.num_worker,
                                pin_memory=True, drop_last=False)
        pseudo_label_unmask = deep_grabcut(Config, encoder, train_loader, Config.root)
        with open(os.path.join(out_save_path, f"pseudo_label_unmask.pkl"), 'wb') as f:
            pickle.dump(pseudo_label_unmask, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--weight', type=str, default='')
    opt = parser.parse_args()

    if opt.test == False:
        main()
    else:
        test_dataset = WHUAugBox(root=Config.root,
                                 split=Config.split_test,
                                 ignore_label=Config.ignore_label,
                                 mean_bgr=Config.mean_bgr,
                                 augment=False)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1, shuffle=False, num_workers=Config.num_worker,
                                 pin_memory=True, drop_last=True)
        encoder = drn_a_50(pretrained=True)
        segmenter = Segmenter()
        encoder.load_state_dict(torch.load(opt.weight)['encoder'])
        segmenter.load_state_dict(torch.load(opt.weight)['segmenter'])
        encoder.to(Config.device)
        segmenter.to(Config.device)
        res = test(encoder, segmenter, test_loader, Config)
        print(res)
