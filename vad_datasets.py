import torch
import numpy as np
import cv2
from collections import OrderedDict
import os
import glob
import scipy.io as sio
from torch.utils.data import Dataset
import torchvision.transforms as transforms


transform = transforms.Compose([
        transforms.ToTensor(),
    ])
# frame_info: the frame information of each dataset: (h, w, file_format, scene_num)
frame_info = {'UCSDped1' : (158, 238, '.tif', 1), 'UCSDped2': (240, 360, '.tif', 1), 'avenue': (360, 640, '.jpg', 1),
              'ShanghaiTech': (480, 856, '.jpg', 1)}


def get_inputs(file_addr, color=cv2.IMREAD_COLOR):
    file_format = file_addr.split('.')[-1]
    if file_format == 'mat':
        return sio.loadmat(file_addr, verify_compressed_data_integrity=False)['uv']
    elif file_format == 'npy':
        return np.load(file_addr)
    else:
        if color == cv2.IMREAD_GRAYSCALE:
            img = cv2.imread(file_addr, color)
            img = np.expand_dims(img, 2)  # to unify the interface of gray images (2D, one channel)
        else:
            img = cv2.imread(file_addr, color)
        return img


def img_tensor2numpy(img):
    # mutual transformation between ndarray-like imgs and Tensor-like images
    # both intensity and rgb images are represented by 3-dim data
    if isinstance(img, np.ndarray):
        return torch.from_numpy(np.transpose(img, [2, 0, 1]))
    else:
        return np.transpose(img, [1, 2, 0]).numpy()


def img_batch_tensor2numpy(img_batch):
    # both intensity and rgb image batch are represented by 4-dim data
    if isinstance(img_batch, np.ndarray):
        if len(img_batch.shape) == 4:
            return torch.from_numpy(np.transpose(img_batch, [0, 3, 1, 2]))
        else:
            return torch.from_numpy(np.transpose(img_batch, [0, 1, 4, 2, 3]))
    else:
        if len(img_batch.numpy().shape) == 4:
            return np.transpose(img_batch, [0, 2, 3, 1]).numpy()
        else:
            return np.transpose(img_batch, [0, 1, 3, 4, 2]).numpy()


def get_foreground(img, bboxes, patch_size):
    img_patches = list()
    if len(img.shape) == 3:
        for i in range(len(bboxes)):
            x_min, x_max = np.int(np.ceil(bboxes[i][0])), np.int(np.ceil(bboxes[i][2]))
            y_min, y_max = np.int(np.ceil(bboxes[i][1])), np.int(np.ceil(bboxes[i][3]))
            cur_patch = img[:, y_min:y_max, x_min:x_max]
            cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]), (patch_size, patch_size))
            if len(cur_patch.shape) == 2:
                cur_patch = np.expand_dims(cur_patch, 2)  # to unify the interface of gray images (2D, one channel)
            img_patches.append(np.transpose(cur_patch, [2, 0, 1]))
        img_patches = np.array(img_patches)
    elif len(img.shape) == 4:
        for i in range(len(bboxes)):
            x_min, x_max = np.int(np.ceil(bboxes[i][0])), np.int(np.ceil(bboxes[i][2]))
            y_min, y_max = np.int(np.ceil(bboxes[i][1])), np.int(np.ceil(bboxes[i][3]))
            cur_patch_set = img[:, :, y_min:y_max, x_min:x_max]
            tmp_set = list()
            for j in range(img.shape[0]):
                cur_patch = cur_patch_set[j]
                cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]), (patch_size, patch_size))
                if len(cur_patch.shape) == 2:
                    cur_patch = np.expand_dims(cur_patch, 2)  # to unify the interface of gray images (2D, one channel)
                tmp_set.append(np.transpose(cur_patch, [2, 0, 1]))
            cur_cube = np.array(tmp_set)
            img_patches.append(cur_cube)
        img_patches = np.array(img_patches)
    return img_patches


def unified_dataset_interface(dataset_name, dir, mode='train', context_frame_num=0, border_mode='elastic', file_format=None, all_bboxes=None, patch_size=32, color=cv2.IMREAD_COLOR):

    if file_format is None:
        if dataset_name in ['UCSDped1', 'UCSDped2']:
            file_format = '.tif'
        elif dataset_name in ['avenue', 'ShanghaiTech']:
            file_format = '.jpg'
        else:
            raise NotImplementedError

    if dataset_name in ['UCSDped1', 'UCSDped2']:
        dataset = ped_dataset(dir=dir, context_frame_num=context_frame_num, mode=mode, border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format, color=color)
    elif dataset_name == 'avenue':
        dataset = avenue_dataset(dir=dir, context_frame_num=context_frame_num, mode=mode, border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format, color=color)
    elif dataset_name == 'ShanghaiTech':
        dataset = shanghaiTech_dataset(dir=dir, context_frame_num=context_frame_num, mode=mode, border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size, file_format=file_format, color=color)
    else:
        raise NotImplementedError

    return dataset


class TrainForegroundDataset(Dataset):
    def __init__(self, fore_dir, h_idx, w_idx, transform=transform):
        self.fore_dir = fore_dir
        self.raw_fore_dir = os.path.join(self.fore_dir, 'block_{}_{}'.format(h_idx, w_idx), 'raw')
        self.of_fore_dir = os.path.join(self.fore_dir, 'block_{}_{}'.format(h_idx, w_idx), 'of')
        self.raw_fore_idx = os.listdir(self.raw_fore_dir)
        self.raw_fore_idx.sort(key=lambda x: int(x[:-4]))  # Sort files by int type (instead of str type).
        self.of_fore_idx = os.listdir(self.of_fore_dir)
        self.of_fore_idx.sort(key=lambda x: int(x[:-4]))   # Sort files by int type (instead of str type).
        self.transform = transform

    def __len__(self):
        return len(self.raw_fore_idx)

    def __getitem__(self, indice):
        fore_idx = self.raw_fore_idx[indice]
        raw_fore_path = os.path.join(self.raw_fore_dir, fore_idx)
        of_fore_path = os.path.join(self.of_fore_dir, fore_idx)
        raw_fore = np.load(raw_fore_path)
        of_fore = np.load(of_fore_path)

        if len(raw_fore.shape) == 3:
            raw_fore = raw_fore[np.newaxis, :, :, :]
            of_fore = of_fore[np.newaxis, :, :, :]

        raw_fore = np.transpose(raw_fore, [1, 2, 0, 3])
        raw_fore = np.reshape(raw_fore, (raw_fore.shape[0], raw_fore.shape[1], -1))
        of_fore = np.transpose(of_fore, [1, 2, 0, 3])
        of_fore = np.reshape(of_fore, (of_fore.shape[0], of_fore.shape[1], -1))

        if self.transform is not None:
            return self.transform(raw_fore), self.transform(of_fore)
        else:
            return raw_fore, of_fore


class FrameForegroundDataset(Dataset):
    def __init__(self, frame_fore_dir):
        self.frame_fore_dir = frame_fore_dir
        self.frame_raw_fore_dir = os.path.join(self.frame_fore_dir, 'raw')
        self.frame_of_fore_dir = os.path.join(self.frame_fore_dir, 'of')
        self.frame_raw_fore_idx = os.listdir(self.frame_raw_fore_dir)
        self.frame_raw_fore_idx.sort(key=lambda x: int(x[:-4]))   # Sort files by int type (instead of str type).
        self.frame_of_fore_idx = os.listdir(self.frame_of_fore_dir)
        self.frame_of_fore_idx.sort(key=lambda x: int(x[:-4]))

    def __len__(self):
        return len(self.frame_raw_fore_idx)

    def __getitem__(self, indice):
        frame_idx = self.frame_raw_fore_idx[indice]
        frame_raw_fore_path = os.path.join(self.frame_raw_fore_dir, frame_idx)
        frame_of_fore_path = os.path.join(self.frame_of_fore_dir, frame_idx)
        frame_raw_fore = np.load(frame_raw_fore_path, allow_pickle=True)
        frame_of_fore = np.load(frame_of_fore_path, allow_pickle=True)

        return frame_raw_fore, frame_of_fore


class TestForegroundDataset(Dataset):
    def __init__(self, raw_fore, of_fore, transform=transform):
        if len(raw_fore.shape) == 4:
            raw_fore = raw_fore[:, np.newaxis, :, :, :]
        if len(of_fore.shape) == 4:
            of_fore = of_fore[:, np.newaxis, :, :, :]
        self.raw_fore = raw_fore
        self.of_fore = of_fore
        self.transform = transform

    def __len__(self):
        return len(self.raw_fore)

    def __getitem__(self, indice):
        cur_raw_fore = self.raw_fore[indice]
        cur_of_fore = self.of_fore[indice]

        cur_raw_fore = np.transpose(cur_raw_fore, [1, 2, 0, 3])
        cur_raw_fore = np.reshape(cur_raw_fore, (cur_raw_fore.shape[0], cur_raw_fore.shape[1], -1))
        cur_of_fore = np.transpose(cur_of_fore, [1, 2, 0, 3])
        cur_of_fore = np.reshape(cur_of_fore, (cur_of_fore.shape[0], cur_of_fore.shape[1], -1))

        if self.transform is not None:
            return self.transform(cur_raw_fore), self.transform(cur_of_fore)
        else:
            return cur_raw_fore, cur_of_fore


class ped_dataset(Dataset):
    '''
    Loading dataset for UCSD ped1/ped2
    '''
    def __init__(self, dir, mode='train', context_frame_num=0, border_mode='elastic', file_format='.tif', all_bboxes=None, patch_size=32, color=cv2.IMREAD_COLOR):
        '''
        :param dir: The directory to load UCSD ped2 dataset
        '''
        self.dir = dir
        self.mode = mode
        if mode == 'train' or mode == 'test':
            self.videos = OrderedDict()
        else:
            self.videos1 = OrderedDict()
            self.videos2 = OrderedDict()
        self.all_frame_addr = list()
        self.frame_video_idx = list()
        self.tot_frame_num = 0
        self.train_frame_num = 0
        
        self.context_frame_num = context_frame_num
        self.border_mode = border_mode
        self.file_format = file_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size
        self.return_gt = False
        self.color = color
        if mode == 'test' or mode == 'merge':
            self.all_gt_addr = list()
            self.gts = OrderedDict()
        if self.dir[-1] == '1':
            self.h = 158
            self.w = 238
        else:
            self.h = 240
            self.w = 360
        self.dataset_init()

    def __len__(self):
        return self.tot_frame_num

    def dataset_init(self):
        if self.mode == 'train':
            data_dir = os.path.join(self.dir, 'Train')
        elif self.mode == 'test':
            data_dir = os.path.join(self.dir, 'Test')
        elif self.mode == 'merge':  # unsupervised
            data_dir1 = os.path.join(self.dir, 'Train')
            data_dir2 = os.path.join(self.dir, 'Test')
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                if 'Train' in video_name:
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                    self.videos[video_name]['frame'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

        elif self.mode == 'test':
            dir_list = glob.glob(os.path.join(data_dir, '*'))
            video_dir_list = []
            gt_dir_list = []
            for dir in sorted(dir_list):
                if '_gt' in dir:
                    gt_dir_list.append(dir)
                    self.return_gt = True
                else:
                    name = dir.split('/')[-1]
                    if 'Test' in name:
                        video_dir_list.append(dir)

            # load frames for test
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # load ground truth of frames
            if self.return_gt:
                for gt in sorted(gt_dir_list):
                    gt_name = gt.split('/')[-1]
                    self.gts[gt_name] = {}
                    self.gts[gt_name]['gt_frame'] = glob.glob(os.path.join(gt, '*.bmp'))
                    self.gts[gt_name]['gt_frame'].sort()

                # merge different frames of different videos into one list
                for _, cont in self.gts.items():
                    self.all_gt_addr += cont['gt_frame']

        elif self.mode == 'merge':
            video_dir_list1 = glob.glob(os.path.join(data_dir1, '*'))
            idx = 1
            for video in sorted(video_dir_list1):
                video_name = video.split('/')[-1]
                if 'Train' in video_name:
                    self.videos1[video_name] = {}
                    self.videos1[video_name]['path'] = video
                    self.videos1[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                    self.videos1[video_name]['frame'].sort()
                    self.videos1[video_name]['length'] = len(self.videos1[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos1[video_name]['length']
                    idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos1.items():
                self.all_frame_addr += cont['frame']
            self.train_frame_num = len(self.all_frame_addr)

            dir_list = glob.glob(os.path.join(data_dir2, '*'))
            video_dir_list2 = []
            gt_dir_list = []
            for dir in sorted(dir_list):
                if '_gt' in dir:
                    gt_dir_list.append(dir)
                    self.return_gt = True
                else:
                    name = dir.split('/')[-1]
                    if 'Test' in name:
                        video_dir_list2.append(dir)

            # load test frames
            for video in sorted(video_dir_list2):
                video_name = video.split('/')[-1]
                self.videos2[video_name] = {}
                self.videos2[video_name]['path'] = video
                self.videos2[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos2[video_name]['frame'].sort()
                self.videos2[video_name]['length'] = len(self.videos2[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos2[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos2.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # load ground truth of frames
            if self.return_gt:
                for gt in sorted(gt_dir_list):
                    gt_name = gt.split('/')[-1]
                    self.gts[gt_name] = {}
                    self.gts[gt_name]['gt_frame'] = glob.glob(os.path.join(gt, '*.bmp'))
                    self.gts[gt_name]['gt_frame'].sort()

                # merge different frames of different videos into one list
                for _, cont in self.gts.items():
                    self.all_gt_addr += cont['gt_frame']
            
        else:
            raise NotImplementedError

    def context_range(self, indice):
        if self.border_mode == 'elastic':
            # check head and tail
            if indice - self.context_frame_num < 0:
                indice = self.context_frame_num
            elif indice + self.context_frame_num > self.tot_frame_num - 1:
                indice = self.tot_frame_num - 1 - self.context_frame_num
            start_idx = indice - self.context_frame_num
            end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1
        elif self.border_mode == 'predict':
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num
            end_idx = indice
            need_context_num = self.context_frame_num + 1
        else:
            # check head and tail
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num

            if indice + self.context_frame_num > self.tot_frame_num - 1:
                end_idx = self.tot_frame_num - 1
            else:
                end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1

        center_idx = self.frame_video_idx[indice]
        video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        pad = need_context_num - len(video_idx)
        if pad > 0:
            if start_idx == 0:
                video_idx = [video_idx[0]] * pad + video_idx
            else:
                video_idx = video_idx + [video_idx[-1]] * pad
        tmp = np.array(video_idx) - center_idx
        offset = tmp.sum()
        if tmp[0] != 0 and tmp[-1] != 0:  # extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError
        if pad == 0 and offset == 0:  # all frames are from the same video
            idx = [x for x in range(start_idx, end_idx+1)]
            return idx
        else:
            if self.border_mode == 'elastic':
                idx = [x for x in range(start_idx - offset, end_idx - offset + 1)]
                return idx
            elif self.border_mode == 'predict':
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), pad) + idx
                return idx
            else:
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if pad > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx - offset, end_idx + 1)]
                        idx = [idx[0]] * pad + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx - offset + 1)]
                        idx = idx + [idx[-1]] * pad
                        return idx

    def __getitem__(self, indice):

        if self.mode == 'train':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice], self.color), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    # print(get_inputs(self.all_frame_addr[idx], self.color).shape)
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx], self.color), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            return img_batch, torch.zeros(1)  # to unify the interface
        elif self.mode == 'test':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice], self.color), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = cv2.imread(self.all_gt_addr[indice], cv2.IMREAD_GRAYSCALE)
                    gt_batch = torch.from_numpy(gt_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx], self.color), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = cv2.imread(self.all_gt_addr[indice], cv2.IMREAD_GRAYSCALE)
                    gt_batch = torch.from_numpy(gt_batch)
            if self.return_gt:
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)  # to unify the interface
        elif self.mode == 'merge':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice], self.color), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)                    
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx], self.color), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            if self.return_gt:
                if indice < self.train_frame_num:  # training set: create gt of all normal frames
                    gt_frame = cv2.imread(self.all_gt_addr[-1], cv2.IMREAD_GRAYSCALE)  # to get gt size
                    h, w = gt_frame.shape
                    gt = torch.zeros(h, w)  # normal frames
                else:  # test set: get gt
                    gt_frame = cv2.imread(self.all_gt_addr[indice-self.train_frame_num], cv2.IMREAD_GRAYSCALE)  # test set
                    gt = torch.from_numpy(gt_frame)

            if self.return_gt:
                return img_batch, gt
            else:
                return img_batch, torch.zeros(1)
        else:
            raise NotImplementedError


class avenue_dataset(Dataset):
    '''
    Loading dataset for Avenue
    '''
    def __init__(self, dir, mode='train', context_frame_num=0, border_mode='elastic', file_format='.jpg', all_bboxes=None, patch_size=32, color=cv2.IMREAD_COLOR):
        '''
        :param dir: The directory to load Avenue dataset
        '''
        self.dir = dir
        self.mode = mode
        if mode == 'train' or mode == 'test':
            self.videos = OrderedDict()
        else:
            self.videos1 = OrderedDict()
            self.videos2 = OrderedDict()
        self.all_frame_addr = list()
        self.frame_video_idx = list()
        self.tot_frame_num = 0
        self.train_frame_num = 0
        self.context_frame_num = context_frame_num
        self.border_mode = border_mode
        self.file_format = file_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size
        self.return_gt = False
        self.color = color
        if mode == 'test' or mode == 'merge':
            self.all_gt = list()
        self.dataset_init()
        pass

    def __len__(self):
        return self.tot_frame_num

    def dataset_init(self):
        if self.mode == 'train':
            data_dir = os.path.join(self.dir, 'training', 'frames')
        elif self.mode == 'test':
            data_dir = os.path.join(self.dir, 'testing', 'frames')
            gt_dir = os.path.join(self.dir, 'ground_truth_demo', 'testing_label_mask')
            if os.path.exists(gt_dir):
                self.return_gt = True
        elif self.mode == 'merge':
            data_dir1 = os.path.join(self.dir, 'training', 'frames')
            data_dir2 = os.path.join(self.dir, 'testing', 'frames')
            gt_dir = os.path.join(self.dir, 'ground_truth_demo', 'testing_label_mask')
            if os.path.exists(gt_dir):
                self.return_gt = True
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

        elif self.mode == 'test':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # set address of ground truth of frames
            if self.return_gt:
                self.all_gt = [sio.loadmat(os.path.join(gt_dir, str(x + 1)+'_label.mat'))['volLabel'] for x in range(len(self.videos))]
                self.all_gt = np.concatenate(self.all_gt, axis=1)
                
        elif self.mode == 'merge':
            video_dir_list1 = glob.glob(os.path.join(data_dir1, '*'))
            idx = 1
            for video in sorted(video_dir_list1):
                video_name = video.split('/')[-1]
                self.videos1[video_name] = {}
                self.videos1[video_name]['path'] = video
                self.videos1[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos1[video_name]['frame'].sort()
                self.videos1[video_name]['length'] = len(self.videos1[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos1[video_name]['length']
                idx += 1
            # merge different frames of different videos into one list
            for _, cont in self.videos1.items():
                self.all_frame_addr += cont['frame']
            self.train_frame_num = len(self.all_frame_addr)
            
            video_dir_list2 = glob.glob(os.path.join(data_dir2, '*'))
            for video in sorted(video_dir_list2):
                video_name = video.split('/')[-1]
                self.videos2[video_name] = {}
                self.videos2[video_name]['path'] = video
                self.videos2[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos2[video_name]['frame'].sort()
                self.videos2[video_name]['length'] = len(self.videos2[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos2[video_name]['length']
                idx += 1
            # merge different frames of different videos into one list
            for _, cont in self.videos2.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # set address of ground truth of frames
            if self.return_gt:
                self.all_gt = [sio.loadmat(os.path.join(gt_dir, str(x + 1)+'_label.mat'))['volLabel'] for x in range(len(self.videos2))]
                self.all_gt = np.concatenate(self.all_gt, axis=1)
        else:
            raise NotImplementedError

    def context_range(self, indice):
        if self.border_mode == 'elastic':
            # check head and tail
            if indice - self.context_frame_num < 0:
                indice = self.context_frame_num
            elif indice + self.context_frame_num > self.tot_frame_num - 1:
                indice = self.tot_frame_num - 1 - self.context_frame_num
            start_idx = indice - self.context_frame_num
            end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1
        elif self.border_mode == 'predict':
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num
            end_idx = indice
            need_context_num = self.context_frame_num + 1
        else:
            # check head and tail
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num

            if indice + self.context_frame_num > self.tot_frame_num - 1:
                end_idx = self.tot_frame_num - 1
            else:
                end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1

        center_idx = self.frame_video_idx[indice]
        video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        pad = need_context_num - len(video_idx)
        if pad > 0:
            if start_idx == 0:
                video_idx = [video_idx[0]] * pad + video_idx
            else:
                video_idx = video_idx + [video_idx[-1]] * pad
        tmp = np.array(video_idx) - center_idx
        offset = tmp.sum()
        if tmp[0] != 0 and tmp[-1] != 0:  # extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError
        if pad == 0 and offset == 0:  # all frames are from the same video
            idx = [x for x in range(start_idx, end_idx+1)]
            return idx
        else:
            if self.border_mode == 'elastic':
                idx = [x for x in range(start_idx - offset, end_idx - offset + 1)]
                return idx
            elif self.border_mode == 'predict':
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), pad) + idx
                return idx
            else:
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if pad > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx - offset, end_idx + 1)]
                        idx = [idx[0]] * pad + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx - offset + 1)]
                        idx = idx + [idx[-1]] * pad
                        return idx

    def __getitem__(self, indice):
        if self.mode == 'train':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice], self.color), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx], self.color), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            return img_batch, torch.zeros(1)  # to unify the interface
        elif self.mode == 'test':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice], self.color), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = self.all_gt[0, indice]
                    gt_batch = torch.from_numpy(gt_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx], self.color), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = self.all_gt[0, indice]
                    gt_batch = torch.from_numpy(gt_batch)
            if self.return_gt:
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)
                
        elif self.mode == 'merge':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice], self.color), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)                    
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx], self.color), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            if self.return_gt:
                if indice < self.train_frame_num:
                    gt_frame = self.all_gt[0, -1]  # to get gt size
                    h, w = gt_frame.shape
                    gt = torch.zeros(h, w)  # normal frames
                else:
                    gt = self.all_gt[0, indice - self.train_frame_num]
                    gt = torch.from_numpy(gt)

            if self.return_gt:
                return img_batch, gt
            else:
                return img_batch, torch.zeros(1)
        else:
            raise NotImplementedError


class shanghaiTech_dataset(Dataset):
    '''
    Loading dataset for ShanghaiTech
    '''
    def __init__(self, dir, mode='train', context_frame_num=0, border_mode='elastic', file_format='.jpg', all_bboxes=None, patch_size=32, color=cv2.IMREAD_COLOR):
        '''
        :param dir: The directory to load ShanghaiTech dataset
        '''
        self.dir = dir
        self.mode = mode
        if mode == 'train' or mode == 'test':
            self.videos = OrderedDict()
        elif mode == 'merge':
            self.videos1 = OrderedDict()
            self.videos2 = OrderedDict()
        else:
            raise NotImplementedError
        self.all_frame_addr = list()
        self.frame_video_idx = list()
        self.tot_frame_num = 0
        self.train_frame_num = 0
        self.context_frame_num = context_frame_num
        self.border_mode = border_mode
        self.file_format = file_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size
        self.return_gt = False
        self.scene_num = 0
        self.color = color
        if mode == 'test' or mode == 'merge':
            self.all_gt = list()
        self.dataset_init()
        pass

    def __len__(self):
        return self.tot_frame_num

    def dataset_init(self):
        if self.mode == 'train':
            data_dir = os.path.join(self.dir, 'training', 'videosFrame')
        elif self.mode == 'test':
            data_dir = os.path.join(self.dir, 'Testing', 'frames_part')
            gt_dir = os.path.join(self.dir, 'Testing', 'test_frame_mask')
            if os.path.exists(gt_dir):
                self.return_gt = True
        elif self.mode == 'merge':
            data_dir1 = os.path.join(self.dir, 'training', 'videosFrame')
            data_dir2 = os.path.join(self.dir, 'Testing', 'frames_part')
            gt_dir = os.path.join(self.dir, 'Testing', 'test_frame_mask')
            if os.path.exists(gt_dir):
                self.return_gt = True
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_dir_list = glob.glob(os.path.join(data_dir, '*'))
            idx = 1
            for video in sorted(video_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

        elif self.mode == 'test':
            idx = 1
            for j in [1, 2]:
                video_dir_list = glob.glob(os.path.join(data_dir+str(j), '*'))
                for video in sorted(video_dir_list):
                    video_name = video.split('/')[-1]
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                    self.videos[video_name]['frame'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # load ground truth of frames
            if self.return_gt:
                gt_dir_list = glob.glob(os.path.join(gt_dir, '*'))
                for gt in sorted(gt_dir_list):
                    self.all_gt.append(np.load(gt))

                # merge different frames of different videos into one list, only support frame gt now due to memory issue
                self.all_gt = np.concatenate(self.all_gt, axis=0)

        elif self.mode == 'merge':
            # Load training dataset.
            video_dir_list1 = glob.glob(os.path.join(data_dir1, '*'))
            idx = 1
            for video in sorted(video_dir_list1):
                video_name = video.split('/')[-1]
                self.videos1[video_name] = {}
                self.videos1[video_name]['path'] = video
                self.videos1[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.file_format))
                self.videos1[video_name]['frame'].sort()
                self.videos1[video_name]['length'] = len(self.videos1[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos1[video_name]['length']
                idx += 1
            # merge different frames of different videos into one list
            for _, cont in self.videos1.items():
                self.all_frame_addr += cont['frame']
            self.train_frame_num = len(self.all_frame_addr)

            # Load testing dataset.
            for j in [1, 2]:
                video_dir_list2 = glob.glob(os.path.join(data_dir2+str(j), '*'))
                for video in sorted(video_dir_list2):
                    video_name = video.split('/')[-1]
                    self.videos2[video_name] = {}
                    self.videos2[video_name]['path'] = video
                    self.videos2[video_name]['frame'] = glob.glob(os.path.join(video, '*'+self.file_format))
                    self.videos2[video_name]['frame'].sort()
                    self.videos2[video_name]['length'] = len(self.videos2[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos2[video_name]['length']
                    idx += 1
            # merge different frames of different videos into one list
            for _, cont in self.videos2.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            # load ground truth of frames
            if self.return_gt:
                gt_dir_list = glob.glob(os.path.join(gt_dir, '*'))
                for gt in sorted(gt_dir_list):
                    self.all_gt.append(np.load(gt))
                # merge different frames of different videos into one list, only support frame gt now due to memory issue
                self.all_gt = np.concatenate(self.all_gt, axis=0)

        else:
            raise NotImplementedError

    def context_range(self, indice):
        if self.border_mode == 'elastic':
            # check head and tail
            if indice - self.context_frame_num < 0:
                indice = self.context_frame_num
            elif indice + self.context_frame_num > self.tot_frame_num - 1:
                indice = self.tot_frame_num - 1 - self.context_frame_num
            start_idx = indice - self.context_frame_num
            end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1
        elif self.border_mode == 'predict':
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num
            end_idx = indice
            need_context_num = self.context_frame_num + 1
        else:
            # check head and tail
            if indice - self.context_frame_num < 0:
                start_idx = 0
            else:
                start_idx = indice - self.context_frame_num

            if indice + self.context_frame_num > self.tot_frame_num - 1:
                end_idx = self.tot_frame_num - 1
            else:
                end_idx = indice + self.context_frame_num
            need_context_num = 2 * self.context_frame_num + 1

        center_idx = self.frame_video_idx[indice]
        video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        pad = need_context_num - len(video_idx)
        if pad > 0:
            if start_idx == 0:
                video_idx = [video_idx[0]] * pad + video_idx
            else:
                video_idx = video_idx + [video_idx[-1]] * pad
        tmp = np.array(video_idx) - center_idx
        offset = tmp.sum()
        if tmp[0] != 0 and tmp[-1] != 0:  # extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError
        if pad == 0 and offset == 0:  # all frames are from the same video
            idx = [x for x in range(start_idx, end_idx+1)]
            return idx
        else:
            if self.border_mode == 'elastic':
                idx = [x for x in range(start_idx - offset, end_idx - offset + 1)]
                return idx
            elif self.border_mode == 'predict':
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), pad) + idx
                return idx
            else:
                if pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if pad > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx - offset, end_idx + 1)]
                        idx = [idx[0]] * pad + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx - offset + 1)]
                        idx = idx + [idx[-1]] * pad
                        return idx

    def __getitem__(self, indice):
        if self.mode == 'train':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice], self.color), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx], self.color), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            return img_batch, torch.zeros(1)  # to unify the interface
        elif self.mode == 'test':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice], self.color), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = np.array([self.all_gt[indice]])
                    gt_batch = torch.from_numpy(gt_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx], self.color), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
                if self.return_gt:
                    gt_batch = np.array([self.all_gt[indice]])
                    gt_batch = torch.from_numpy(gt_batch)
            if self.return_gt:
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)  # to unify the interface
        elif self.mode == 'merge':
            if self.context_frame_num == 0:
                img_batch = np.transpose(get_inputs(self.all_frame_addr[indice], self.color), [2, 0, 1])
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            else:
                frame_range = self.context_range(indice=indice)
                img_batch = []
                for idx in frame_range:
                    cur_img = np.transpose(get_inputs(self.all_frame_addr[idx], self.color), [2, 0, 1])
                    img_batch.append(cur_img)
                img_batch = np.array(img_batch)
                if self.all_bboxes is not None:
                    img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                img_batch = torch.from_numpy(img_batch)
            if self.return_gt:
                if indice < self.train_frame_num:  # training set: create gt of all normal frames.
                    gt_batch = torch.zeros(1,)
                else:  # testing set: get gt.
                    gt_batch = np.array([self.all_gt[indice - self.train_frame_num]])
                    gt_batch = torch.from_numpy(gt_batch)
            if self.return_gt:
                return img_batch, gt_batch
            else:
                return img_batch, torch.zeros(1)  # to unify the interface
        else:
            raise NotImplementedError
    

