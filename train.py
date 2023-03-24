import numpy as np
import os
from torch.utils.data import DataLoader
from vad_datasets import TrainForegroundDataset
from misc import AverageMeter
import torch
from model.cae import ReconCAE
import torch.optim as optim
import torch.nn as nn
from configparser import ConfigParser
from utils import visualize_pair, visualize_batch, visualize_pair_map
import argparse


def train(dataset_name, mode, data_root_dir, epochs, batch_size, use_flow, nf, context_frame_num,
          lambda_raw, lambda_of, sp_mode, shrink_rate, warmup_epoch, w_block, h_block):
    loss_func = nn.MSELoss(reduction='none')
    score_func = nn.MSELoss(reduce=False)
    init_std_range = 4
    stc_folder = '{}_{}_set_for_training'.format(dataset_name, mode)

    model_set = [[[] for ww in range(w_block)] for hh in range(h_block)]
    raw_training_scores_set = [[[] for ww in range(w_block)] for hh in range(h_block)]
    if use_flow:
        of_training_scores_set = [[[] for ww in range(w_block)] for hh in range(h_block)]
    for h_idx in range(h_block):
        for w_idx in range(w_block):
            raw_losses = AverageMeter()
            of_losses = AverageMeter()

            cur_dataset = TrainForegroundDataset(fore_dir=os.path.join(data_root_dir, stc_folder), h_idx=h_idx, w_idx=w_idx)
            cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
            tot_frame_num = 2 * context_frame_num + 1
            cur_model = torch.nn.DataParallel(ReconCAE(tot_frame_num=tot_frame_num, features_root=nf, use_flow=use_flow)).cuda()
            optimizer = optim.Adam(cur_model.parameters(), weight_decay=0.0, lr=0.001)

            cur_model.train()
            t = 0
            for epoch in range(epochs):
                for idx, (raw, of) in enumerate(cur_dataloader):
                    raw = raw.cuda()
                    of = of.cuda()
                    weights = torch.ones(raw.size(0)).cuda().type(torch.cuda.FloatTensor)

                    if epoch >= warmup_epoch:
                        cur_model.eval()
                        with torch.no_grad():
                            target_raw, target_of, output_raw, output_of = cur_model(raw, of)

                            raw_scores = score_func(output_raw, target_raw)
                            raw_scores = torch.sum(torch.sum(torch.sum(raw_scores, dim=3), dim=2), dim=1)

                            scores = raw_scores

                            weights = torch.ones_like(scores)
                            avg = torch.mean(scores)
                            std = torch.std(scores)
                            if sp_mode != 'mixture':
                                lambda_sp = avg + (init_std_range - t * shrink_rate) * std
                                if lambda_sp < avg + std:
                                    lambda_sp = avg + std
                                    print('min lambda sp', t, lambda_sp)
                                t += 1
                                easy_idx = torch.where(scores < lambda_sp)[0]
                                hard_idx = torch.where(scores >= lambda_sp)[0]

                                if sp_mode == 'hard':
                                    weights[easy_idx] = 1.0
                                    if len(hard_idx) > 0:
                                        weights[hard_idx] = 0.0
                                        # visualize_batch(batch=img_batch_tensor2numpy(raw[hard_idx].cpu().detach()[:, 6:9, :, :]))
                                elif sp_mode == 'linear':
                                    weights[easy_idx] = 1.0 - scores[easy_idx] / lambda_sp
                                    if len(hard_idx) > 0:
                                        weights[hard_idx] = 0.0
                                        # visualize_batch(batch=img_batch_tensor2numpy(raw[hard_idx].cpu().detach()[:, 6:9, :, :]))
                                elif sp_mode == 'huber':
                                    weights[easy_idx] = 1.0
                                    if len(hard_idx) > 0:
                                        weights[hard_idx] = lambda_sp / scores[hard_idx]

                            else:
                                lambda1_sp = avg + 1.0 * std
                                lambda2_sp = avg + (init_std_range - t * shrink_rate) * std
                                if lambda2_sp < lambda1_sp:
                                    lambda2_sp = lambda1_sp
                                t += 1
                                easy_idx = torch.where(scores <= lambda1_sp)[0]
                                median_idx = torch.where((scores > lambda1_sp) & (scores < lambda2_sp))[0]
                                hard_idx = torch.where(scores >= lambda2_sp)[0]
                                weights[easy_idx] = 1.0
                                if len(hard_idx) > 0:
                                    weights[hard_idx] = 0.0
                                    # visualize_batch(batch=img_batch_tensor2numpy(raw[hard_idx].cpu().detach()[:, 6:9, :, :]))
                                if len(median_idx) > 0:
                                    weights[median_idx] = (lambda1_sp * (lambda2_sp - scores[median_idx])) / (scores[median_idx] * (lambda2_sp - lambda1_sp))

                        cur_model.train()

                    target_raw, target_of, output_raw, output_of = cur_model(raw, of)
                    # visualization
                    # max_num = 30
                    # visualize_pair_map(
                    #     batch_1=img_batch_tensor2numpy(target_raw.cpu().detach()[:max_num, 12:15, :, :]),
                    #     batch_2=img_batch_tensor2numpy(output_raw.cpu().detach()[:max_num, 12:15, :, :]))
                    # visualize_pair_map(
                    #     batch_1=img_batch_tensor2numpy(target_of.cpu().detach()[:max_num, 8:10, :, :]),
                    #     batch_2=img_batch_tensor2numpy(output_of.cpu().detach()[:max_num, 8:10, :, :]))

                    loss_raw = loss_func(output_raw, target_raw)
                    loss_raw = loss_raw.sum(dim=3).sum(dim=2).sum(dim=1)
                    loss_raw = loss_raw @ weights / target_raw.size(0) / target_raw.size(1) / target_raw.size(2) / target_raw.size(3)
                    raw_losses.update(loss_raw.item(), raw.size(0))
                    if use_flow:
                        loss_of = loss_func(output_of, target_of)
                        loss_of = loss_of.sum(dim=3).sum(dim=2).sum(dim=1)
                        loss_of = loss_of @ weights / target_of.size(0) / target_of.size(1) / target_of.size(2) / target_of.size(3)
                        of_losses.update(loss_of.item(), of.size(0))
                    else:
                        of_losses.update(0., of.size(0))

                    if use_flow:
                        loss = lambda_raw * loss_raw + lambda_of * loss_of
                    else:
                        loss = loss_raw

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if idx % 5 == 0:
                        print('Block: ({}, {}), epoch {}, batch {} of {}, raw loss: {}, of loss: {}'.format(
                            h_idx, w_idx, epoch, idx, cur_dataset.__len__() // batch_size, raw_losses.avg, of_losses.avg))

            model_set[h_idx][w_idx].append(cur_model.module.state_dict())
            #  /*--  A forward pass to store the training scores of optical flow and raw datasets respectively*/
            forward_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=False, num_workers=8)  # 32
            cur_model.eval()
            for idx, (raw, of) in enumerate(forward_dataloader):
                raw = raw.cuda().type(torch.cuda.FloatTensor)
                of = of.cuda().type(torch.cuda.FloatTensor)

                target_raw, target_of, output_raw, output_of = cur_model(raw, of)

                raw_scores = score_func(output_raw, target_raw).cpu().data.numpy()
                raw_scores = np.sum(np.sum(np.sum(raw_scores, axis=3), axis=2), axis=1)  # mse
                raw_training_scores_set[h_idx][w_idx].append(raw_scores)
                if use_flow:
                    of_scores = score_func(output_of, target_of).cpu().data.numpy()
                    of_scores = np.sum(np.sum(np.sum(of_scores, axis=3), axis=2), axis=1)  # mse
                    of_training_scores_set[h_idx][w_idx].append(of_scores)

            raw_training_scores_set[h_idx][w_idx] = np.concatenate(raw_training_scores_set[h_idx][w_idx], axis=0)
            if use_flow:
                of_training_scores_set[h_idx][w_idx] = np.concatenate(of_training_scores_set[h_idx][w_idx], axis=0)

    torch.save(raw_training_scores_set, os.path.join(data_root_dir, '{}_raw_training_scores.npy'.format(dataset_name)))
    if use_flow:
        torch.save(of_training_scores_set, os.path.join(data_root_dir, '{}_of_training_scores.npy'.format(dataset_name)))
    torch.save(model_set, os.path.join(data_root_dir, '{}_model.npy'.format(dataset_name)))
    print('Training of LBR_SPR for dataset: {} has completed!'.format(dataset_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, default="UCSDped2",
                        help="dataset name: UCSDped1, UCSDped2, avenue or ShanghaiTech")
    parser.add_argument("--mode", type=str, required=True, default="test",
                        help="the testing set or the merged set of training and testing set: test or merge "
                             "(the training set will not be processed separately due to unsupervised setting)")
    args = parser.parse_args()

    cp = ConfigParser()
    cp.read(os.path.join('configs', "{}.cfg".format(args.dataset_name)))
    epochs = cp.getint('LBR_SPR_parameters', 'epochs')
    batch_size = cp.getint('LBR_SPR_parameters', 'batch_size')
    use_flow = cp.getboolean('LBR_SPR_parameters', 'use_flow')
    nf = cp.getint('LBR_SPR_parameters', 'num_feature_root')
    context_frame_num = cp.getint('data_parameters', 'context_frame_num')
    lambda_raw = cp.getfloat('LBR_SPR_parameters', 'lambda_raw')
    lambda_of = cp.getfloat('LBR_SPR_parameters', 'lambda_of')
    sp_mode = cp.get('LBR_SPR_parameters', 'sp_mode')
    shrink_rate = cp.getfloat('LBR_SPR_parameters', 'shrink_rate')
    warmup_epoch = cp.getint('LBR_SPR_parameters', 'warmup_epoch')
    h_block = cp.getint('data_parameters', 'h_block')
    w_block = cp.getint('data_parameters', 'w_block')
    data_root_dir = cp.get('data_parameters', 'data_root_dir')

    train(args.dataset_name, args.mode, data_root_dir, epochs, batch_size, use_flow, nf, context_frame_num,
          lambda_raw, lambda_of, sp_mode, shrink_rate, warmup_epoch, w_block, h_block)

