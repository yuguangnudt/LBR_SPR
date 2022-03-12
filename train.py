import numpy as np
import os
from torch.utils.data import DataLoader
from vad_datasets import unified_dataset_interface, TrainForegroundDataset, img_tensor2numpy, img_batch_tensor2numpy, frame_size
from fore_det.inference import init_detector
from fore_det.obj_det_with_motion import imshow_bboxes, getObBboxes, getGdBboxes, getOfBboxes, delCoverBboxes
from misc import AverageMeter
import torch
from model.cae import ReconCAE
import torch.optim as optim
import torch.nn as nn
from configparser import ConfigParser
from utils import calc_block_idx, visualize_pair, visualize_batch, visualize_pair_map
import shutil

#  /*-------------------------------------------------Overall parameter setting-----------------------------------------------------*/
cp = ConfigParser()
cp.read("config.cfg")

dataset_name = cp.get('shared_parameters', 'dataset_name')  # The name of dataset: UCSDped1/UCSDped2/avenue/ShanghaiTech.
raw_dataset_dir = cp.get('shared_parameters', 'raw_dataset_dir')  # Fixed
foreground_extraction_mode = cp.get('shared_parameters', 'foreground_extraction_mode')  # Foreground extraction method: obj_det_with_gd/obj_det_with_of/obj_det/frame.
data_root_dir = cp.get('shared_parameters', 'data_root_dir')  # Fixed: A folder that stores the data such as foreground produced by the program.
modality = cp.get('shared_parameters', 'modality')  # Fixed
mode = cp.get('train_parameters', 'mode')  # merge/test
method = cp.get('shared_parameters', 'method')  # Fixed
try:
    patch_size = cp.getint(dataset_name, 'patch_size')  # Resize the foreground bounding boxes.
    train_block_mode = cp.getint(dataset_name, 'train_block_mode')  # Fixed
    motionThr = cp.getfloat(dataset_name, 'motionThr')  # Fixed
    # Define h_block * w_block sub-regions of video frames for localized training
    h_block = cp.getint(dataset_name, 'h_block')
    w_block = cp.getint(dataset_name, 'w_block')

    # Set 'bbox_save=False' and 'foreground_saved=False' at first to calculate and store the bboxes and foreground,
    # then set them to True to load the stored bboxes and foreground directly, if the foreground parameters remain unchanged.
    bbox_saved = cp.getboolean(dataset_name, '{}_bbox_saved'.format(mode))
    foreground_saved = cp.getboolean(dataset_name, 'Train_{}_foreground_saved'.format(mode))
except:
    raise NotImplementedError

#  /*--------------------------------------------------Foreground localization-----------------------------------------------------*/
config_file = 'fore_det/obj_det_config/yolov3_d53_mstrain-608_273e_coco.py'
checkpoint_file = 'fore_det/obj_det_checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'

# set dataset for foreground extraction
# raw dataset
dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name), context_frame_num=1, mode=mode, border_mode='hard')
# optical flow dataset
dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name), context_frame_num=1, mode=mode, border_mode='hard', file_format='.npy')
print(dataset.__len__(), dataset2.__len__())
if not bbox_saved:
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    all_bboxes = list()
    for idx in range(dataset.__len__()):
        batch, _ = dataset.__getitem__(idx)
        batch2, _ = dataset2.__getitem__(idx)
        print('Extracting bboxes of {}-th frame in total {} frame'.format(idx + 1, dataset.__len__()))
        cur_img = img_tensor2numpy(batch[1])
        cur_of = img_tensor2numpy(batch2[1])

        if foreground_extraction_mode == 'obj_det_with_of':
            # A coarse detection of bboxes by pretrained object detector
            ob_bboxes = getObBboxes(cur_img, model, dataset_name, verbose=False)
            ob_bboxes = delCoverBboxes(ob_bboxes, dataset_name)

            # visual object detection bounding boxes with covering filter
            # imshow_bboxes(cur_img, ob_bboxes, win_name='del_cover_bboxes')

            # further foreground detection by optical flow
            of_bboxes = getOfBboxes(cur_of, cur_img, ob_bboxes, dataset_name, verbose=False)

            if of_bboxes.shape[0] > 0:
                cur_bboxes = np.concatenate((ob_bboxes, of_bboxes), axis=0)
            else:
                cur_bboxes = ob_bboxes
        else:
            raise NotImplementedError

        # imshow_bboxes(cur_img, cur_bboxes)
        all_bboxes.append(cur_bboxes)
    np.save(os.path.join(dataset.dir, 'bboxes_{}_{}.npy'.format(mode, foreground_extraction_mode)), all_bboxes)
    print('bboxes for training data saved!')
elif not foreground_saved:
    all_bboxes = np.load(os.path.join(dataset.dir, 'bboxes_{}_{}.npy'.format(mode, foreground_extraction_mode)), allow_pickle=True)
    print('bboxes for training data loaded!')

# /*--------------------------------------------------Foreground extraction--------------------------------------------------------*/
border_mode = cp.get(method, 'border_mode')
if not foreground_saved:
    context_frame_num = cp.getint(method, 'context_frame_num')
    context_of_num = cp.getint(method, 'context_of_num')
    if modality == 'raw_datasets':
        file_format = frame_size[dataset_name][2]
    elif modality == 'raw2flow':
        file_format1 = frame_size[dataset_name][2]
        file_format2 = '.npy'
    else:
        file_format = '.npy'

    if modality == 'raw2flow':
        dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                            context_frame_num=context_frame_num, mode=mode,
                                            border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                            file_format=file_format1)
        dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name),
                                             context_frame_num=context_of_num, mode=mode,
                                             border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                             file_format=file_format2)
    else:
        dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(modality, dataset_name),
                                            context_frame_num=context_frame_num, mode=mode,
                                            border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                            file_format=file_format)

    h_step, w_step = frame_size[dataset_name][0] / h_block, frame_size[dataset_name][1] / w_block

    # Create folders to store foreground.
    fore_folder_name = dataset_name + '_' + 'Train_' + mode + '_' + 'foreground' + '_{}'.format(foreground_extraction_mode)
    if os.path.exists(os.path.join(data_root_dir, modality, fore_folder_name)):
        shutil.rmtree(os.path.join(data_root_dir, modality, fore_folder_name))
    for h in range(h_block):
        for w in range(w_block):
            raw_fore_dir = os.path.join(data_root_dir, modality, fore_folder_name, 'block_{}_{}'.format(h, w), 'raw')
            of_fore_dir = os.path.join(data_root_dir, modality, fore_folder_name, 'block_{}_{}'.format(h, w), 'of')
            os.makedirs(raw_fore_dir)
            os.makedirs(of_fore_dir)

    fore_idx = 0
    for idx in range(dataset.__len__()):
        batch, _ = dataset.__getitem__(idx)
        if modality == 'raw2flow':
            batch2, _ = dataset2.__getitem__(idx)

        print('Extracting foreground in {}-th batch, {} in total'.format(idx + 1, dataset.__len__() // 1))

        cur_bboxes = all_bboxes[idx]
        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            if modality == 'raw2flow':
                batch2 = img_batch_tensor2numpy(batch2)

            if modality == 'optical_flow':
                if len(batch.shape) == 4:
                    mag = np.sum(np.sum(np.sum(batch ** 2, axis=3), axis=2), axis=1)
                else:
                    mag = np.mean(np.sum(np.sum(np.sum(batch ** 2, axis=4), axis=3), axis=2), axis=1)
            elif modality == 'raw2flow':
                if len(batch2.shape) == 4:
                    mag = np.sum(np.sum(np.sum(batch2 ** 2, axis=3), axis=2), axis=1)

                else:
                    mag = np.mean(np.sum(np.sum(np.sum(batch2 ** 2, axis=4), axis=3), axis=2), axis=1)
            else:
                mag = np.ones(batch.shape[0]) * 10000

            for idx_bbox in range(cur_bboxes.shape[0]):
                if mag[idx_bbox] > motionThr:
                    all_blocks = calc_block_idx(cur_bboxes[idx_bbox, 0], cur_bboxes[idx_bbox, 2], cur_bboxes[idx_bbox, 1], cur_bboxes[idx_bbox, 3], h_step, w_step, mode=train_block_mode)
                    for (h_block_idx, w_block_idx) in all_blocks:
                        np.save(os.path.join(data_root_dir, modality, fore_folder_name, 'block_{}_{}'.format(h_block_idx, w_block_idx), 'raw', '{}.npy'.format(fore_idx)), batch[idx_bbox])
                        if modality == 'raw2flow':
                            np.save(os.path.join(data_root_dir, modality, fore_folder_name, 'block_{}_{}'.format(h_block_idx, w_block_idx), 'of', '{}.npy'.format(fore_idx)), batch2[idx_bbox])
                        fore_idx += 1

    print('foreground for training data saved!')

#  /*----------------------------------------------------End-to-End training--------------------------------------------------------*/

if method == 'LBR_SPR':
    epochs = cp.getint(method, 'epochs')
    batch_size = cp.getint(method, 'batch_size')
    useFlow = cp.getboolean(method, 'useFlow')
    nf = cp.getint(method, 'nf')
    if border_mode == 'predict':
        tot_frame_num = cp.getint(method, 'context_frame_num') + 1
    else:
        tot_frame_num = 2 * cp.getint(method, 'context_frame_num') + 1
    assert modality == 'raw2flow'

    lambda_raw = cp.getfloat(method, 'lambda_raw')
    lambda_of = cp.getfloat(method, 'lambda_of')
    loss_func = nn.MSELoss(reduction='none')
    score_func = nn.MSELoss(reduce=False)

    sp_mode = cp.get(method, 'sp_mode')
    shrink_rate = cp.getfloat(method, 'shrink_rate')
    warmup_epoch = cp.getint(method, 'warmup_epoch')
    init_std_range = 4

    model_set = [[[] for ww in range(w_block)] for hh in range(h_block)]
    raw_training_scores_set = [[[] for ww in range(w_block)] for hh in range(h_block)]
    if useFlow:
        of_training_scores_set = [[[] for ww in range(w_block)] for hh in range(h_block)]
    fore_folder_name = dataset_name + '_' + 'Train_' + mode + '_' + 'foreground' + '_{}'.format(foreground_extraction_mode)
    for h_idx in range(h_block):
        for w_idx in range(w_block):
            raw_losses = AverageMeter()
            of_losses = AverageMeter()

            cur_dataset = TrainForegroundDataset(fore_dir=os.path.join(data_root_dir, modality, fore_folder_name), h_idx=h_idx, w_idx=w_idx)
            cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

            cur_model = torch.nn.DataParallel(ReconCAE(tot_frame_num=tot_frame_num, features_root=nf, use_flow=useFlow)).cuda()

            optimizer = optim.Adam(cur_model.parameters(), weight_decay=0.0, lr=0.001)

            cur_model.train()
            t = 0
            for epoch in range(epochs):
                for idx, (raw, of) in enumerate(cur_dataloader):
                    raw = raw.cuda().type(torch.cuda.FloatTensor)
                    of = of.cuda().type(torch.cuda.FloatTensor)
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
                                # print(t, lambda_sp)
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
                                elif sp_mode =='huber':
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
                                    # print(weights[median_idx])

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
                    if useFlow:
                        loss_of = loss_func(output_of, target_of)
                        loss_of = loss_of.sum(dim=3).sum(dim=2).sum(dim=1)
                        loss_of = loss_of @ weights / target_of.size(0) / target_of.size(1) / target_of.size(2) / target_of.size(3)
                        of_losses.update(loss_of.item(), of.size(0))
                    else:
                        of_losses.update(0., of.size(0))

                    if useFlow:
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
                if useFlow:
                    of_scores = score_func(output_of, target_of).cpu().data.numpy()
                    of_scores = np.sum(np.sum(np.sum(of_scores, axis=3), axis=2), axis=1)  # mse
                    of_training_scores_set[h_idx][w_idx].append(of_scores)

            raw_training_scores_set[h_idx][w_idx] = np.concatenate(raw_training_scores_set[h_idx][w_idx], axis=0)
            if useFlow:
                of_training_scores_set[h_idx][w_idx] = np.concatenate(of_training_scores_set[h_idx][w_idx], axis=0)

    torch.save(raw_training_scores_set, os.path.join(data_root_dir, modality, '{}_raw_training_scores_{}.npy'.format(dataset_name, method)))
    if useFlow:
        torch.save(of_training_scores_set, os.path.join(data_root_dir, modality, '{}_of_training_scores_{}.npy'.format(dataset_name, method)))
    torch.save(model_set, os.path.join(data_root_dir, modality, '{}_model_{}.npy'.format(dataset_name, method)))
    print('Training of {} for dataset: {} has completed!'.format(method, dataset_name))

else:
    raise NotImplementedError
