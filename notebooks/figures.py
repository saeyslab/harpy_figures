# Visualization code adapted from: https://github.com/vanvalenlab/publication-figures/blob/master/2021-Greenwald_Miller_et_al-Mesmer/figures.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def get_cmap():
    coolwarm = plt.colormaps.get_cmap('coolwarm')
    newcolors = coolwarm(np.linspace(0, 1, 256))
    black = np.array([0, 0, 0, 1])
    newcolors[1:2, :] = black
    newcmp = ListedColormap(newcolors)
    return newcmp

def apply_colormap_to_img(label_img):
    newcmp = get_cmap()

    transformed = np.copy(label_img)
    transformed -= np.min(transformed)
    transformed /= np.max(transformed)

    transformed = newcmp(transformed)

    return transformed

def label_image_by_ratio(true_label, pred_label, threshold=2):
    from skimage.segmentation import find_boundaries

    def get_matching_true_ids(true_label, pred_label):
        true_ids, pred_ids = [], []
        for pred_cell in np.unique(pred_label[pred_label > 0]):
            pred_mask = pred_label == pred_cell
            overlap_ids, overlap_counts = np.unique(true_label[pred_mask], return_counts=True)
            true_id = overlap_ids[np.argmax(overlap_counts)]
            true_ids.append(true_id)
            pred_ids.append(pred_cell)
        return true_ids, pred_ids

    def get_cell_size(label_list, label_map):
        size_list = []
        for label in label_list:
            size = np.sum(label_map == label)
            size_list.append(size)
        return size_list

    true_ids, pred_ids = get_matching_true_ids(true_label, pred_label)
    true_sizes = get_cell_size(true_ids, true_label)
    pred_sizes = get_cell_size(pred_ids, pred_label)
    fill_val = -threshold + 0.02
    disp_img = np.full_like(pred_label.astype('float32'), fill_val)
    for i in range(len(pred_ids)):
        current_id = pred_ids[i]
        true_id = true_ids[i]
        if true_id == 0:
            ratio = threshold
        else:
            ratio = np.log2(pred_sizes[i] / true_sizes[i])
        mask = pred_label == current_id
        boundaries = find_boundaries(mask, mode='inner')
        mask[boundaries > 0] = 0
        if ratio > threshold:
            ratio = threshold
        if ratio < -threshold:
            ratio = -threshold
        disp_img[mask] = ratio

    disp_img[-1, -1] = -threshold
    disp_img[-1, -2] = threshold

    return disp_img