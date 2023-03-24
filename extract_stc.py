import numpy as np
import os
from vad_datasets import unified_dataset_interface, img_batch_tensor2numpy, frame_info
from configparser import ConfigParser
from utils import calc_block_idx
import shutil
import argparse


def extract_training_stc(data_root_dir, dataset_name, mode, all_bboxes, border_mode,
                         context_frame_num, context_of_num, patch_size, h_block, w_block):

    file_format1 = frame_info[dataset_name][2]
    file_format2 = '.npy'
    dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                        context_frame_num=context_frame_num, mode=mode,
                                        border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                        file_format=file_format1)
    dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name),
                                         context_frame_num=context_of_num, mode=mode,
                                         border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                         file_format=file_format2)

    h_step, w_step = frame_info[dataset_name][0] / h_block, frame_info[dataset_name][1] / w_block

    # Create folders to store STCs.
    stc_folder = '{}_{}_set_for_training'.format(dataset_name, mode)
    if os.path.exists(os.path.join(data_root_dir, stc_folder)):
        shutil.rmtree(os.path.join(data_root_dir, stc_folder))
    for h in range(h_block):
        for w in range(w_block):
            raw_fore_dir = os.path.join(data_root_dir, stc_folder, 'block_{}_{}'.format(h, w), 'raw')
            of_fore_dir = os.path.join(data_root_dir, stc_folder, 'block_{}_{}'.format(h, w), 'of')
            os.makedirs(raw_fore_dir)
            os.makedirs(of_fore_dir)

    fore_idx = 0
    for idx in range(dataset.__len__()):
        batch, _ = dataset.__getitem__(idx)
        batch2, _ = dataset2.__getitem__(idx)

        print('Extracting STCs in {}-th batch, {} in total'.format(idx + 1, dataset.__len__() // 1))

        cur_bboxes = all_bboxes[idx]
        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            batch2 = img_batch_tensor2numpy(batch2)

            for idx_bbox in range(cur_bboxes.shape[0]):
                all_blocks = calc_block_idx(cur_bboxes[idx_bbox, 0], cur_bboxes[idx_bbox, 2],
                                            cur_bboxes[idx_bbox, 1], cur_bboxes[idx_bbox, 3], h_step, w_step, 1)
                for (h_block_idx, w_block_idx) in all_blocks:
                    np.save(os.path.join(data_root_dir, stc_folder, 'block_{}_{}'.format(h_block_idx, w_block_idx),
                                         'raw', '{}.npy'.format(fore_idx)), batch[idx_bbox])
                    np.save(os.path.join(data_root_dir, stc_folder, 'block_{}_{}'.format(h_block_idx, w_block_idx),
                                         'of', '{}.npy'.format(fore_idx)), batch2[idx_bbox])
                    fore_idx += 1

    print('STCs for training saved!')


def extract_testing_stc(data_root_dir, dataset_name, mode, all_bboxes, border_mode,
                         context_frame_num, context_of_num, patch_size, h_block, w_block):
    file_format1 = frame_info[dataset_name][2]
    file_format2 = '.npy'

    dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                        context_frame_num=context_frame_num, mode=mode,
                                        border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                        file_format=file_format1)
    dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name),
                                         context_frame_num=context_of_num, mode=mode,
                                         border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                         file_format=file_format2)

    h_step, w_step = frame_info[dataset_name][0] / h_block, frame_info[dataset_name][1] / w_block

    # To store bboxes corresponding to the STCs in each frame.
    foreground_bbox_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]

    # Create folders to store STCs.
    stc_folder = '{}_{}_set_for_testing'.format(dataset_name, mode)
    for h in range(h_block):
        for w in range(w_block):
            raw_fore_dir = os.path.join(data_root_dir, stc_folder, 'raw')
            of_fore_dir = os.path.join(data_root_dir, stc_folder, 'of')
            os.makedirs(raw_fore_dir, exist_ok=True)
            os.makedirs(of_fore_dir, exist_ok=True)

    for idx in range(dataset.__len__()):
        print('Extracting STCs in {}-th batch, {} in total'.format(idx + 1, dataset.__len__() // 1))

        batch, _ = dataset.__getitem__(idx)
        batch2, _ = dataset2.__getitem__(idx)
        cur_bboxes = all_bboxes[idx]

        frame_foreground = [[[] for ww in range(w_block)] for hh in range(h_block)]
        frame_foreground2 = [[[] for ww in range(w_block)] for hh in range(h_block)]
        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            batch2 = img_batch_tensor2numpy(batch2)

            for idx_bbox in range(cur_bboxes.shape[0]):
                all_blocks = calc_block_idx(cur_bboxes[idx_bbox, 0], cur_bboxes[idx_bbox, 2],
                                            cur_bboxes[idx_bbox, 1], cur_bboxes[idx_bbox, 3], h_step, w_step, 1)
                for (h_block_idx, w_block_idx) in all_blocks:
                    frame_foreground[h_block_idx][w_block_idx].append(batch[idx_bbox])
                    frame_foreground2[h_block_idx][w_block_idx].append(batch2[idx_bbox])
                    foreground_bbox_set[idx][h_block_idx][w_block_idx].append(cur_bboxes[idx_bbox])

        frame_foreground = [[np.array(frame_foreground[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
        frame_foreground2 = [[np.array(frame_foreground2[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
        np.save(os.path.join(data_root_dir, stc_folder, 'raw', '{}.npy'.format(idx)), frame_foreground)
        np.save(os.path.join(data_root_dir, stc_folder, 'of', '{}.npy'.format(idx)), frame_foreground2)

    foreground_bbox_set = [[[np.array(foreground_bbox_set[ii][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]
    np.save(os.path.join(data_root_dir, stc_folder, 'foreground_bbox.npy'), foreground_bbox_set)
    print('STCs for testing data saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, default="UCSDped2",
                        help="dataset name: UCSDped1, UCSDped2, avenue or ShanghaiTech")
    parser.add_argument("--mode", type=str, required=True, default="test",
                        help="the testing set or the merged set of training and testing set: test or merge "
                             "(the training set will not be processed separately due to unsupervised setting)")
    parser.add_argument("--extract_training_stc", action="store_true", help="extract training STCs")
    parser.add_argument("--extract_testing_stc", action="store_true", help="extract testing STCs")
    args = parser.parse_args()

    cp = ConfigParser()
    cp.read(os.path.join('configs', "{}.cfg".format(args.dataset_name)))
    data_root_dir = cp.get('data_parameters', 'data_root_dir')
    border_mode = cp.get('data_parameters', 'border_mode')
    context_frame_num = cp.getint('data_parameters', 'context_frame_num')
    context_of_num = cp.getint('data_parameters', 'context_of_num')
    patch_size = cp.getint('data_parameters', 'patch_size')
    h_block = cp.getint('data_parameters', 'h_block')
    w_block = cp.getint('data_parameters', 'w_block')

    all_bboxes = np.load(os.path.join('raw_datasets', args.dataset_name, '{}_{}_bboxes.npy'.format(args.dataset_name, args.mode)), allow_pickle=True)

    if args.extract_training_stc:
        extract_training_stc(data_root_dir, args.dataset_name, args.mode, all_bboxes, border_mode,
                             context_frame_num, context_of_num, patch_size, h_block, w_block)
    elif args.extract_testing_stc:
        extract_testing_stc(data_root_dir, args.dataset_name, args.mode, all_bboxes, border_mode,
                            context_frame_num, context_of_num, patch_size, h_block, w_block)
    else:
        raise NotImplementedError
