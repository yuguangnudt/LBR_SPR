import numpy as np
import os
from vad_datasets import unified_dataset_interface, img_tensor2numpy
from fore_det.inference import init_detector
from fore_det.obj_det_with_motion import imshow_bboxes, getObBboxes, getOfBboxes, delCoverBboxes
import argparse


def extract_bbox(dataset_name, mode):
    config_file = 'fore_det/obj_det_config/yolov3_d53_mstrain-608_273e_coco.py'
    checkpoint_file = 'fore_det/obj_det_checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'

    # set dataset for foreground extraction
    # raw dataset
    dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name), context_frame_num=1, mode=mode, border_mode='hard')
    # optical flow dataset
    dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name), context_frame_num=1, mode=mode, border_mode='hard',
                                         file_format='.npy')

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    all_bboxes = list()
    for idx in range(dataset.__len__()):
        batch, _ = dataset.__getitem__(idx)
        batch2, _ = dataset2.__getitem__(idx)
        print('Extracting bboxes of {}-th frame in total {} frame'.format(idx + 1, dataset.__len__()))
        cur_img = img_tensor2numpy(batch[1])
        cur_of = img_tensor2numpy(batch2[1])

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

        # imshow_bboxes(cur_img, cur_bboxes)
        all_bboxes.append(cur_bboxes)
    np.save(os.path.join(dataset.dir, '{}_{}_bboxes.npy'.format(dataset_name, mode)), all_bboxes)
    print('bboxes saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, default="UCSDped2",
                        help="dataset name: UCSDped1, UCSDped2, avenue or ShanghaiTech")
    parser.add_argument("--mode", type=str, required=True, default="test",
                        help="the testing set or the merged set of training and testing set: test or merge "
                             "(the training set will not be processed separately due to unsupervised setting)")
    args = parser.parse_args()

    extract_bbox(args.dataset_name, args.mode)