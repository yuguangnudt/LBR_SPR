import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import cv2
from flowlib import flow_to_image
import torch
from torch import nn


def calc_block_idx(x_min, x_max, y_min, y_max, h_step, w_step, mode):
    all_blocks = list()
    center = np.array([(y_min + y_max) / 2, (x_min + x_max) / 2])
    all_blocks.append(center + center)
    if mode > 1:
        all_blocks.append(np.array([y_min, center[1]]) + center)
        all_blocks.append(np.array([y_max, center[1]]) + center)
        all_blocks.append(np.array([center[0], x_min]) + center)
        all_blocks.append(np.array([center[0], x_max]) + center)
    if mode >= 9:
        all_blocks.append(np.array([y_min, x_min]) + center)
        all_blocks.append(np.array([y_max, x_max]) + center)
        all_blocks.append(np.array([y_max, x_min]) + center)
        all_blocks.append(np.array([y_min, x_max]) + center)
    all_blocks = np.array(all_blocks) / 2
    h_block_idxes = all_blocks[:, 0] / h_step
    w_block_idxes = all_blocks[:, 1] / w_step
    h_block_idxes, w_block_idxes = list(h_block_idxes.astype(np.int)), list(w_block_idxes.astype(np.int))
    # delete repeated elements
    all_blocks = set([x for x in zip(h_block_idxes, w_block_idxes)])
    all_blocks = [x for x in all_blocks]
    return all_blocks


def save_roc_pr_curve_data(scores, labels, file_path, verbose=True):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    # calculate EER
    fnr = 1 - tpr
    eer1 = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    eer2 = fnr[np.nanargmin(np.absolute(fnr - fpr))]

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    if verbose is True:
        print('AUC@ROC is {}'.format(roc_auc), 'EER1 is {}'.format(eer1), 'EER2 is {}'.format(eer2))

    np.savez_compressed(file_path,
                        preds=preds, truth=truth,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom)

    return roc_auc


def moving_average(values, window_size, decay):
    # window_size >= 1 0<decay<=1
    window = np.logspace(0, window_size-1, window_size, base=decay)
    # print(window)
    window = window / np.sum(window)
    # print(window)
    return np.convolve(values, window, 'same')


def visualize_score(score_map, big_number):
    lower_bound = -1 * big_number
    upper_bound = big_number
    all_values = np.reshape(score_map, (-1, ))
    all_values = all_values[all_values > lower_bound]
    all_values = all_values[all_values < upper_bound]
    max_val = all_values.max()
    min_val = all_values.min()
    visual_map = (score_map - min_val) / (max_val - min_val)
    visual_map[score_map == lower_bound] = 0
    visual_map[score_map == upper_bound] = 1
    visual_map *= 255
    visual_map = visual_map.astype(np.uint8)
    return visual_map


def visualize_img(img):
    if img.shape[2] == 2:
        cv2.imshow('Optical flow', flow_to_image(img))
    else:
        cv2.imshow('Image', img)
    cv2.waitKey(0)


def visualize_batch(batch):
    if len(batch.shape) == 4:
        if batch.shape[3] == 2:
            batch = [flow_to_image(batch[i]) for i in range(batch.shape[0])]
            cv2.imshow('Optical flow set', np.hstack(batch))
        else:
            batch = [batch[i] for i in range(batch.shape[0])]
            cv2.imshow('Image sets', np.hstack(batch))
        cv2.waitKey(0)
    else:
        if batch.shape[4] == 2:
            batch = [np.hstack([flow_to_image(batch[j][i]) for i in range(batch[j].shape[0])]) for j in range(batch.shape[0])]
            cv2.imshow('Optical flow set', np.vstack(batch))
        else:
            batch = [np.hstack([batch[j][i] for i in range(batch[j].shape[0])]) for j in range(batch.shape[0])]
            cv2.imshow('Image sets', np.vstack(batch))
        cv2.waitKey(0)


def visualize_pair(batch_1, batch_2):
    if len(batch_1.shape) == 4:
        if batch_1.shape[3] == 2:
            batch_1 = [flow_to_image(batch_1[i]) for i in range(batch_1.shape[0])]
        else:
            batch_1 = [batch_1[i] for i in range(batch_1.shape[0])]
        if batch_2.shape[3] == 2:
            batch_2 = [flow_to_image(batch_2[i]) for i in range(batch_2.shape[0])]
        else:
            batch_2 = [batch_2[i] for i in range(batch_2.shape[0])]
        cv2.imshow('Pair comparison', np.vstack([np.hstack(batch_1), np.hstack(batch_2)]))
        cv2.waitKey(0)
    else:
        if batch_1.shape[4] == 2:
            batch_1 = [flow_to_image(batch_1[-1][i]) for i in range(batch_1[-1].shape[0])]
        else:
            batch_1 = [batch_1[-1][i] for i in range(batch_1[-1].shape[0])]
        if batch_2.shape[4] == 2:
            batch_2 = [flow_to_image(batch_2[-1][i]) for i in range(batch_2[-1].shape[0])]
        else:
            batch_2 = [batch_2[-1][i] for i in range(batch_2[-1].shape[0])]
        cv2.imshow('Pair comparison', np.vstack([np.hstack(batch_1), np.hstack(batch_2)]))
        cv2.waitKey(0)