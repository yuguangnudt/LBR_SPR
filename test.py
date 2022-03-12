import numpy as np
import os
from torch.utils.data import DataLoader
from vad_datasets import unified_dataset_interface, FrameForegroundDataset, TestForegroundDataset, img_tensor2numpy, img_batch_tensor2numpy, frame_size
from fore_det.inference import init_detector
from fore_det.obj_det_with_motion import imshow_bboxes, getObBboxes, getGdBboxes, getOfBboxes, delCoverBboxes
import torch
from model.cae import ReconCAE
import torch.nn as nn
from utils import save_roc_pr_curve_data, calc_block_idx, moving_average, visualize_pair, visualize_batch, visualize_pair_map, visualize_score, visualize_img
from configparser import ConfigParser

#  /*-------------------------------------------------Overall parameter setting-----------------------------------------------------*/
cp = ConfigParser()
cp.read("config.cfg")

dataset_name = cp.get('shared_parameters', 'dataset_name')
raw_dataset_dir = cp.get('shared_parameters', 'raw_dataset_dir')
foreground_extraction_mode = cp.get('shared_parameters', 'foreground_extraction_mode')
data_root_dir = cp.get('shared_parameters', 'data_root_dir')
modality = cp.get('shared_parameters', 'modality')
mode = cp.get('test_parameters', 'mode')  # merge/test

method = cp.get('shared_parameters', 'method')
try:
    patch_size = cp.getint(dataset_name, 'patch_size')
    h_block = cp.getint(dataset_name, 'h_block')
    w_block = cp.getint(dataset_name, 'w_block')
    test_block_mode = cp.getint(dataset_name, 'test_block_mode')
    bbox_saved = cp.getboolean(dataset_name, mode + '_bbox_saved')
    foreground_saved = cp.getboolean(dataset_name, 'Test_' + mode + '_foreground_saved')
    motionThr = cp.getfloat(dataset_name, 'motionThr')
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
    print('bboxes for testing data saved!')
elif not foreground_saved:
    all_bboxes = np.load(os.path.join(dataset.dir, 'bboxes_{}_{}.npy'.format(mode, foreground_extraction_mode)), allow_pickle=True)
    print('bboxes for testing data loaded!')

#  /*--------------------------------------------------Foreground extraction--------------------------------------------------------*/
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

    # set dataset for foreground bbox extraction
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
        dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(modality, dataset_name), context_frame_num=context_frame_num, mode=mode,
                                            border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format)

    h_step, w_step = frame_size[dataset_name][0] / h_block, frame_size[dataset_name][1] / w_block

    # To store bboxes corresponding to foreground in each frame.
    foreground_bbox_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]

    # Create folders to store foreground.
    fore_folder_name = dataset_name + '_' + 'Test_' + mode + '_' + 'foreground' + '_{}'.format(foreground_extraction_mode)
    for h in range(h_block):
        for w in range(w_block):
            raw_fore_dir = os.path.join(data_root_dir, modality, fore_folder_name, 'raw')
            of_fore_dir = os.path.join(data_root_dir, modality, fore_folder_name, 'of')
            os.makedirs(raw_fore_dir, exist_ok=True)
            os.makedirs(of_fore_dir, exist_ok=True)

    for idx in range(dataset.__len__()):
        print('Extracting foreground in {}-th batch, {} in total'.format(idx + 1, dataset.__len__() // 1))

        batch, _ = dataset.__getitem__(idx)
        if modality == 'raw2flow':
            batch2, _ = dataset2.__getitem__(idx)
        cur_bboxes = all_bboxes[idx]

        frame_foreground = [[[] for ww in range(w_block)] for hh in range(h_block)]
        if modality == 'raw2flow':
            frame_foreground2 = [[[] for ww in range(w_block)] for hh in range(h_block)]
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
                    all_blocks = calc_block_idx(cur_bboxes[idx_bbox, 0], cur_bboxes[idx_bbox, 2], cur_bboxes[idx_bbox, 1], cur_bboxes[idx_bbox, 3], h_step, w_step, mode=test_block_mode)
                    for (h_block_idx, w_block_idx) in all_blocks:
                        frame_foreground[h_block_idx][w_block_idx].append(batch[idx_bbox])
                        if modality == 'raw2flow':
                            frame_foreground2[h_block_idx][w_block_idx].append(batch2[idx_bbox])
                        foreground_bbox_set[idx][h_block_idx][w_block_idx].append(cur_bboxes[idx_bbox])

        frame_foreground = [[np.array(frame_foreground[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
        frame_foreground2 = [[np.array(frame_foreground2[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
        np.save(os.path.join(data_root_dir, modality, fore_folder_name, 'raw', '{}.npy'.format(idx)), frame_foreground)
        np.save(os.path.join(data_root_dir, modality, fore_folder_name, 'of', '{}.npy'.format(idx)), frame_foreground2)

    foreground_bbox_set = [[[np.array(foreground_bbox_set[ii][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]
    np.save(os.path.join(data_root_dir, modality, fore_folder_name, 'foreground_bbox.npy'), foreground_bbox_set)
    print('foreground for testing data saved!')
else:
    fore_folder_name = dataset_name + '_' + 'Test_' + mode + '_' + 'foreground' + '_{}'.format(foreground_extraction_mode)
    foreground_bbox_set = np.load(os.path.join(data_root_dir, modality, fore_folder_name, 'foreground_bbox.npy'), allow_pickle=True)
    print('foreground bboxes for testing data loaded! Will load saved testing foreground with dataset interface.')

#  /*-------------------------------------------------Abnormal event detection-----------------------------------------------------*/
scores_saved = cp.getboolean(dataset_name, 'scores_saved')
if scores_saved is False:
    h, w, _, sn = frame_size[dataset_name]
    border_mode = cp.get(method, 'border_mode')
    nf = cp.getint(method, 'nf')
    if border_mode == 'predict':
        tot_frame_num = cp.getint(method, 'context_frame_num') + 1
        tot_of_num = cp.getint(method, 'context_of_num') + 1
    else:
        tot_frame_num = 2 * cp.getint(method, 'context_frame_num') + 1
        tot_of_num = 2 * cp.getint(method, 'context_of_num') + 1
    useFlow = cp.getboolean(method, 'useFlow')
    w_raw = cp.getfloat(method, 'w_raw')
    w_of = cp.getfloat(method, 'w_of')

    assert modality == 'raw2flow'
    score_func = nn.MSELoss(reduce=False)
    big_number = 20

    results_dir = 'results'
    frame_result_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(frame_result_dir, exist_ok=True)
    pixel_result_dir = os.path.join(results_dir, dataset_name, 'score_mask')
    os.makedirs(pixel_result_dir, exist_ok=True)

    model_weights = torch.load(os.path.join(data_root_dir, modality, '{}_model_{}.npy'.format(dataset_name, method)))

    model_set = [[[] for ww in range(len(model_weights[hh]))] for hh in range(len(model_weights))]
    for hh in range(len(model_weights)):
        for ww in range(len(model_weights[hh])):
            if len(model_weights[hh][ww]) > 0:
                cur_model = ReconCAE(tot_frame_num=tot_frame_num, features_root=nf, use_flow=useFlow).cuda()
                cur_model.load_state_dict(model_weights[hh][ww][0])
                model_set[hh][ww].append(cur_model.eval())

    #  get training scores statistics
    raw_training_scores_set = torch.load(os.path.join(data_root_dir, modality, '{}_raw_training_scores_{}.npy'.format(dataset_name, method)))
    if useFlow:
        of_training_scores_set = torch.load(os.path.join(data_root_dir, modality, '{}_of_training_scores_{}.npy'.format(dataset_name, method)))

    # mean and std of training scores
    raw_stats_set = [[(np.mean(raw_training_scores_set[hh][ww]), np.std(raw_training_scores_set[hh][ww])) for ww in range(w_block)] for hh in range(h_block)]
    if useFlow:
        of_stats_set = [[(np.mean(of_training_scores_set[hh][ww]), np.std(of_training_scores_set[hh][ww])) for ww in range(w_block)] for hh in range(h_block)]

    fore_folder_name = dataset_name + '_' + 'Test_' + mode + '_' + 'foreground' + '_{}'.format(foreground_extraction_mode)
    frame_fore_dataset = FrameForegroundDataset(frame_fore_dir=os.path.join(data_root_dir, modality, fore_folder_name))
    # Get scores.
    if dataset_name == 'avenue' or dataset_name == 'ShanghaiTech':
        frame_scores = []
    for frame_idx in range(frame_fore_dataset.__len__()):
        print('Calculating scores for {}-th frame in total {} frames'.format(frame_idx, frame_fore_dataset.__len__()))
        frame_raw_fore, frame_of_fore = frame_fore_dataset.__getitem__(frame_idx)
        cur_bboxes = foreground_bbox_set[frame_idx]
        # normal: no objects in test set
        cur_pixel_results = -1 * np.ones(shape=(h, w)) * big_number

        for h_idx in range(h_block):
            for w_idx in range(w_block):
                if len(frame_raw_fore[h_idx][w_idx]) > 0:
                    if len(model_set[h_idx][w_idx]) > 0:
                        cur_model = model_set[h_idx][w_idx][0]
                        cur_dataset = TestForegroundDataset(raw_fore=frame_raw_fore[h_idx][w_idx], of_fore=frame_of_fore[h_idx][w_idx])
                        cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=frame_raw_fore[h_idx][w_idx].shape[0], shuffle=False, num_workers=0)

                        for idx, (raw, of) in enumerate(cur_dataloader):
                            raw = raw.cuda().type(torch.cuda.FloatTensor)
                            of = of.cuda().type(torch.cuda.FloatTensor)
                            # print(raw.shape)

                            target_raw, target_of, output_raw, output_of = cur_model(raw, of)

                            # mse
                            raw_scores = score_func(output_raw, target_raw).cpu().data.numpy()
                            raw_scores = np.sum(np.sum(np.sum(raw_scores, axis=3), axis=2), axis=1)
                            # print(raw_scores)
                            if useFlow:
                                of_scores = score_func(output_of, target_of).cpu().data.numpy()
                                of_scores = np.sum(np.sum(np.sum(of_scores, axis=3), axis=2), axis=1)

                            # visualization
                            # max_num = 30
                            # visualize_pair_map(
                            #     batch_1=img_batch_tensor2numpy(target_raw.cpu().detach()[:max_num, 6:9, :, :]),
                            #     batch_2=img_batch_tensor2numpy(output_raw.cpu().detach()[:max_num, 6:9, :, :]))
                            # visualize_pair(
                            #     batch_1=img_batch_tensor2numpy(target_of.cpu().detach()[:max_num, 4:6, :, :]),
                            #     batch_2=img_batch_tensor2numpy(output_of.cpu().detach()[:max_num, 4:6, :, :]))

                            # normalize scores using training scores
                            raw_scores = (raw_scores - raw_stats_set[h_idx][w_idx][0]) / raw_stats_set[h_idx][w_idx][1]
                            if useFlow:
                                of_scores = (of_scores - of_stats_set[h_idx][w_idx][0]) / of_stats_set[h_idx][w_idx][1]

                            if useFlow:
                                scores = w_raw * raw_scores + w_of * of_scores
                            else:
                                scores = raw_scores
                    else:
                        # anomaly: no objects in training set while objects occur in this block
                        scores = np.ones(frame_raw_fore[h_idx][w_idx].shape[0], ) * big_number

                    for m in range(scores.shape[0]):
                        cur_score_mask = -1 * np.ones(shape=(h, w)) * big_number
                        cur_score = scores[m]
                        bbox = cur_bboxes[h_idx][w_idx][m]
                        x_min, x_max = np.int(np.ceil(bbox[0])), np.int(np.ceil(bbox[2]))
                        y_min, y_max = np.int(np.ceil(bbox[1])), np.int(np.ceil(bbox[3]))
                        cur_score_mask[y_min:y_max, x_min:x_max] = cur_score
                        cur_pixel_results = np.max(np.concatenate([cur_pixel_results[:, :, np.newaxis], cur_score_mask[:, :, np.newaxis]], axis=2), axis=2)
        if dataset_name == 'UCSDped1' or dataset_name == 'UCSDped2':
            torch.save(cur_pixel_results, os.path.join(pixel_result_dir, '{}'.format(frame_idx)))
        elif dataset_name == 'avenue' or dataset_name == 'ShanghaiTech':
            frame_scores.append(cur_pixel_results.max())
    if dataset_name == 'avenue' or dataset_name == 'ShanghaiTech':
        torch.save(frame_scores, os.path.join(frame_result_dir, 'frame_scores.npy'))


#  /*-------------------------------------------------------Evaluation-----------------------------------------------------------*/
criterion = 'frame'
batch_size = 1
# set dataset for evaluation
dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(raw_dataset_dir, dataset_name), context_frame_num=0, mode=mode, border_mode='hard')
dataset_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)

print('Evaluating {} by {}-criterion:'.format(dataset_name, criterion))
if criterion == 'frame':
    if dataset_name == 'UCSDped1' or dataset_name == 'UCSDped2':
        all_frame_scores = list()
    elif dataset_name == 'avenue' or dataset_name == 'ShanghaiTech':
        all_frame_scores = torch.load(os.path.join(results_dir, dataset_name, 'frame_scores.npy'))
    all_targets = list()
    for idx, (_, target) in enumerate(dataset_loader):
        print('Processing {}-th frame'.format(idx))
        if dataset_name == 'UCSDped1' or dataset_name == 'UCSDped2':
            cur_pixel_results = torch.load(os.path.join(results_dir, dataset_name, 'score_mask', '{}'.format(idx)))
            all_frame_scores.append(cur_pixel_results.max())
        all_targets.append(target[0].numpy().max())
    all_frame_scores = np.array(all_frame_scores)
    all_targets = np.array(all_targets)
    all_targets = all_targets > 0
    nor = all_targets[all_targets == 0]
    ano = all_targets[all_targets != 0]
    print(len(nor), len(ano))
    results_path = os.path.join(results_dir, dataset_name, '{}_{}_{}_frame_results.npz'.format(modality, foreground_extraction_mode, method))
    print('Results written to {}:'.format(results_path))

    # moving average
    all_frame_scores = moving_average(all_frame_scores, window_size=1, decay=1)
    auc = save_roc_pr_curve_data(all_frame_scores, all_targets, results_path)
elif criterion == 'pixel':
    if dataset_name == 'UCSDped1' or dataset_name == 'UCSDped2':
        all_pixel_scores = list()
        all_targets = list()
        thr = 0.4
        for idx, (_, target) in enumerate(dataset_loader):
            print('Processing {}-th frame'.format(idx))
            cur_pixel_results = torch.load(os.path.join(results_dir, dataset_name, 'score_mask', '{}'.format(idx)))
            target_mask = target[0].numpy()
            all_targets.append(target[0].numpy().max())
            if all_targets[-1] > 0:
                cur_effective_scores = cur_pixel_results[target_mask > 0]
                sorted_score = np.sort(cur_effective_scores)
                cut_off_idx = np.int(np.round((1 - thr) * cur_effective_scores.shape[0]))
                cut_off_score = sorted_score[cut_off_idx]
            else:
                cut_off_score = cur_pixel_results.max()
            all_pixel_scores.append(cut_off_score)
        all_frame_scores = np.array(all_pixel_scores)
        all_targets = np.array(all_targets)
        all_targets = all_targets > 0
        results_path = os.path.join(results_dir, dataset_name,
                                    '{}_{}_{}_pixel_results.npz'.format(modality, foreground_extraction_mode, method))
        print('Results written to {}:'.format(results_path))
        results = save_roc_pr_curve_data(all_frame_scores, all_targets, results_path)
    else:
        raise NotImplementedError
else:
    raise NotImplementedError
