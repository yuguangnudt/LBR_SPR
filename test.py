import numpy as np
import os
from torch.utils.data import DataLoader
from vad_datasets import unified_dataset_interface, FrameForegroundDataset, TestForegroundDataset, frame_info, img_batch_tensor2numpy
import torch
from model.cae import ReconCAE
import torch.nn as nn
from utils import save_roc_pr_curve_data, moving_average, visualize_pair_map
from configparser import ConfigParser
import argparse


def test(dataset_name, mode, nf, context_frame_num, use_flow, w_raw, w_of, data_root_dir, w_block, h_block, window_size):
    results_dir = 'results'
    frame_result_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(frame_result_dir, exist_ok=True)
    pixel_result_dir = os.path.join(results_dir, dataset_name, 'score_mask')
    os.makedirs(pixel_result_dir, exist_ok=True)

    model_weights = torch.load(os.path.join(data_root_dir, '{}_model.npy'.format(dataset_name)))
    model_set = [[[] for ww in range(len(model_weights[hh]))] for hh in range(len(model_weights))]
    tot_frame_num = 2 * context_frame_num + 1
    for hh in range(len(model_weights)):
        for ww in range(len(model_weights[hh])):
            if len(model_weights[hh][ww]) > 0:
                cur_model = ReconCAE(tot_frame_num=tot_frame_num, features_root=nf, use_flow=use_flow).cuda()
                cur_model.load_state_dict(model_weights[hh][ww][0])
                model_set[hh][ww].append(cur_model.eval())

    #  get training scores statistics
    raw_training_scores_set = torch.load(os.path.join(data_root_dir, '{}_raw_training_scores.npy'.format(dataset_name)))
    if use_flow:
        of_training_scores_set = torch.load(os.path.join(data_root_dir, '{}_of_training_scores.npy'.format(dataset_name)))
    # mean and std of training scores
    raw_stats_set = [[(np.mean(raw_training_scores_set[hh][ww]), np.std(raw_training_scores_set[hh][ww])) for ww in range(w_block)] for hh in range(h_block)]
    if use_flow:
        of_stats_set = [[(np.mean(of_training_scores_set[hh][ww]), np.std(of_training_scores_set[hh][ww])) for ww in range(w_block)] for hh in range(h_block)]

    h, w, _, _ = frame_info[dataset_name]
    score_func = nn.MSELoss(reduce=False)
    big_number = 20
    stc_folder = '{}_{}_set_for_testing'.format(dataset_name, mode)
    foreground_bbox_set = np.load(os.path.join(data_root_dir, stc_folder, 'foreground_bbox.npy'), allow_pickle=True)
    frame_fore_dataset = FrameForegroundDataset(frame_fore_dir=os.path.join(data_root_dir, stc_folder))
    # To calculate anomaly score for each frame.
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
                            raw = raw.cuda()
                            of = of.cuda()

                            target_raw, target_of, output_raw, output_of = cur_model(raw, of)

                            raw_scores = score_func(output_raw, target_raw).cpu().data.numpy()
                            raw_scores = np.sum(raw_scores, axis=(1, 2, 3))
                            if use_flow:
                                of_scores = score_func(output_of, target_of).cpu().data.numpy()
                                of_scores = np.sum(of_scores, axis=(1, 2, 3))

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
                            if use_flow:
                                of_scores = (of_scores - of_stats_set[h_idx][w_idx][0]) / of_stats_set[h_idx][w_idx][1]

                            if use_flow:
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

    # evaluation
    dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                        context_frame_num=0, mode=mode, border_mode='hard')
    dataset_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8)
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
    results_path = os.path.join(results_dir, dataset_name, 'frame_results.npz')
    print('Results written to {}:'.format(results_path))
    all_frame_scores = moving_average(all_frame_scores, window_size=window_size, decay=1)
    auc = save_roc_pr_curve_data(all_frame_scores, all_targets, results_path)


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
    nf = cp.getint('LBR_SPR_parameters', 'num_feature_root')
    context_frame_num = cp.getint('data_parameters', 'context_frame_num')
    use_flow = cp.getboolean('LBR_SPR_parameters', 'use_flow')
    w_raw = cp.getfloat('LBR_SPR_parameters', 'w_raw')
    w_of = cp.getfloat('LBR_SPR_parameters', 'w_of')
    data_root_dir = cp.get('data_parameters', 'data_root_dir')
    h_block = cp.getint('data_parameters', 'h_block')
    w_block = cp.getint('data_parameters', 'w_block')
    window_size = cp.getint('LBR_SPR_parameters', 'window_size')

    test(args.dataset_name, args.mode, nf, context_frame_num, use_flow, w_raw, w_of, data_root_dir, w_block, h_block, window_size)