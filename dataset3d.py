# coding=utf-8
import random

import numpy
from torch.utils.data import Dataset
import torchvision
import numpy as np
from os.path import join, isdir, isfile
import os as os
import pandas as pd
import psutil
import torch
import nibabel as nib
import math
import itertools
from operator import itemgetter
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import ast
from PIL import Image, ImageFilter
from misalignment import augment_misalign


def drawContour(m, s, c, RGB):
    """Draw edges of contour 'c' from segmented image 's' onto 'm' in colour 'RGB'"""

    # Fill contour "c" with white, make all else black
    thisContour = s.point(lambda p: p == c and 250)

    # Find edges of this contour and make into Numpy array
    thisEdges = thisContour.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))
    thisEdgesN = np.array(thisEdges)

    # Paint locations of found edges in color "RGB" onto "main"

    m[np.nonzero(thisEdgesN)] = RGB
    return m

def correct_bbox(img, bbox):

    new_patch_img = np.zeros((1, 24, 126, 126))
    coords = [0, 24, 0, 126, 0, 126]

    if bbox[0] < 0:                                 # first case: depth coordinate is larger at the beginning

        center = int(bbox[1] / 2)

        if bbox[1] > 2 * center:
            coords[0] = 12 - center
            coords[1] = 12 + center + 1

        elif bbox[1] < 2 * center:
            coords[0] = 12 - center
            coords[1] = 12 + center - 1
        else:
            coords[0] = 12 - center
            coords[1] = 12 + center

    if bbox[1] > img.shape[1]:                      # second case: depth coordinate is larger at the end

        center = int((img.shape[1] - bbox[0]) / 2)

        if img.shape[1] - bbox[0] > 2 * center:
            coords[0] = 12 - center
            coords[1] = 12 + center + 1

        elif img.shape[1] - bbox[0] < 2 * center:
            coords[0] = 12 - center
            coords[1] = 12 + center - 1

        else:
            coords[0] = 12 - center
            coords[1] = 12 + center

    if bbox[2] < 0:
        coords[2] = np.abs(bbox[2])

    if bbox[3] > img.shape[2]:
        coords[3] = 126 - (bbox[3] - img.shape[2])

    if bbox[4] < 0:
        coords[4] = np.abs(bbox[4])

    if bbox[5] > img.shape[2]:
        coords[5] = 126 - (bbox[5] - img.shape[3])

    new_patch_img[:,coords[0]:coords[1],coords[2]:coords[3],coords[4]:coords[5]] = img[:,
                                                                                    max(bbox[0],0):min(img.shape[1],bbox[1]),
                                                                                    max(bbox[2],0):min(img.shape[2],bbox[3]),
                                                                                    max(bbox[4],0):min(bbox[5],img.shape[3])]

    return new_patch_img



def compute_size_batch2(df):
    ### create the column of size_batch
    size_batch = []

    separations_nohcc = np.percentile(df[df['has_hcc'] == 0]['diameter_2d'], q=50)#19
    separations_hcc = np.percentile(df[df['has_hcc'] == 1]['diameter_2d'], q=50)#19

    for _, row in df.iterrows():
        if row['has_hcc'] == 0:
            if row['diameter_2d'] <= separations_nohcc:
                size_batch.append(1)
            else:
                size_batch.append(2)

        elif row['has_hcc'] == 1:
            if row['diameter_2d'] <= separations_hcc:
                size_batch.append(3)
            else:
                size_batch.append(4)

    df['size_batch'] = size_batch

    return df


def generate_bbox(bbox):
    z, y, x = 24, 96, 96
    z_min, z_max, y_min, y_max, x_min, x_max = bbox

    center_lesion_x = int((x_max - x_min) / 2)
    center_lesion_y = int((y_max - y_min) / 2)
    center_lesion_z = int((z_max - z_min) / 2)

    new_x_min = x_min + center_lesion_x - int(x / 2)
    new_x_max = x_max - center_lesion_x + int(x / 2)

    new_y_min = y_min + center_lesion_y - int(y / 2)
    new_y_max = y_max - center_lesion_y + int(y / 2)

    new_z_min = z_min + center_lesion_z - int(z / 2)
    new_z_max = z_max - center_lesion_z + int(z / 2)

    if new_x_max - new_x_min == 97:
        new_x_max -= 1
    if new_y_max - new_y_min == 97:
        new_y_max -= 1
    if new_z_max - new_z_min == 25:
        new_z_max -= 1

    bb = [new_z_min, new_z_max, new_y_min, new_y_max, new_x_min, new_x_max]
    return bb


class DatasetCL(Dataset):  # output size : (N,(512,512,2),1,1)

    def __init__(self, config, training=False, validation=False, test=False, chicp=False,
                 grenoble=False, dimension=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.grenoble = grenoble
        self.dimension = dimension

        if self.grenoble:
            self.config.database = 'grenoble'
            self.config.path_to_data = "/datadrive/emma/2022_contrastive/2_datasets/2_grenoble_ct_cirrhosis__volume"
            #self.config.path_to_data = "/datadrive/emma/2022_contrastive/2_datasets/1_lihc_ct_cirrhosis__volume"

        else:
            self.config.database = 'calv'

        if self.config.cross_val:
            if self.config.database == 'grenoble':
                self.df = pd.read_csv(join(self.config.lght_dir, '1_subjects_label_hcc_grenoble_cv.csv'), delimiter=",",
                                      index_col='datapoint').drop(['Unnamed: 0'], axis=1)
                dct = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
                self.df['label'] = [*map(dct.get, self.df['class'])]
            else:
                self.df = pd.read_csv(join(self.config.lght_dir, config.dir,
                                           '1_subjects_label_hcc_chic_cv.csv'),
                                      delimiter=",").drop(
                    ['Unnamed: 0'], axis=1).reset_index(drop=True)
                self.df = self.df.dropna(subset=['aphe','npw','ec']).reset_index(drop=True)

                lr1vols = ['bat-chic-1_sub-CH10267_ses-00001_pro-CT-Portal__1', 'bat-chic-1_sub-CH10226_ses-00002_pro-CT-Portal__1',
                           'bat-chic-1_sub-CH10017_ses-00001_pro-CT-Portal__1', 'bat-chic-1_sub-CH10388_ses-00001_pro-CT-Portal__1',
                           'bat-chic-1_sub-CH10343_ses-00002_pro-CT-Portal__1', 'bat-chic-1_sub-CH10343_ses-00002_pro-CT-Portal__2',
                           'bat-chic-1_sub-CH10131_ses-00002_pro-CT-Portal__1', 'bat-chic-1_sub-CH10288_ses-00002_pro-CT-Portal__1',
                           'bat-chic-1_sub-CH10288_ses-00002_pro-CT-Portal__2', 'bat-chic-1_sub-CH10028_ses-00001_pro-CT-Portal__1',
                           'bat-chic-2_sub-CH20076_ses-00001_pro-CT-Portal__1', 'bat-chic-2_sub-CH20076_ses-00001_pro-CT-Portal__2',
                           'bat-chic-2_sub-CH20011_ses-00001_pro-CT-Portal__1', 'bat-chic-2_sub-CH20011_ses-00001_pro-CT-Portal__3',
                           'bat-chic-1_sub-CH10097_ses-00001_pro-CT-Portal__1', 'bat-chic-1_sub-CH10129_ses-00001_pro-CT-Portal__1',
                           'bat-chic-2_sub-CH20120_ses-00001_pro-CT-Portal__1', 'bat-chic-1_sub-CH10396_ses-00001_pro-CT-Portal__1',
                           'bat-chic-2_sub-CH20100_ses-00001_pro-CT-Portal__1', 'bat-chic-1_sub-CH10297_ses-00001_pro-CT-Portal__1',
                           'bat-chic-2_sub-CH20139_ses-00001_pro-CT-Portal__1', 'bat-chic-1_sub-CH10319_ses-00001_pro-CT-Portal__1',
                           'bat-chic-1_sub-CH10056_ses-00001_pro-CT-Portal__1', 'bat-chic-1_sub-CH10425_ses-00002_pro-CT-Portal__1',
                           'bat-chic-1_sub-CH10103_ses-00001_pro-CT-Portal__1', 'bat-chic-1_sub-CH10103_ses-00001_pro-CT-Portal__2',
                           'bat-chic-2_sub-CH20053_ses-00001_pro-CT-Portal__1', 'bat-chic-1_sub-CH10249_ses-00002_pro-CT-Portal__1', 'bat-chic-1_sub-CH10249_ses-00002_pro-CT-Portal__2']
                self.df = self.df[~self.df['dp_lesion'].isin(lr1vols)].reset_index(drop=True)
        else:
            if self.config.database == 'grenoble':
                self.df = pd.read_csv(join(self.config.path_to_data, '1_subjects_label_hcc_grenoble.csv'), delimiter=",").drop(['Unnamed: 0'], axis=1)
                dct = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
                self.df['label'] = [*map(dct.get, self.df['class'])]
            else:
                self.df = pd.read_csv(join(self.config.path_to_data, '1_subjects_label_hcc_deeptekcalv_chic.csv'),
                                      delimiter=",").drop(
                    ['Unnamed: 0'], axis=1).reset_index(drop=True)

        print('constructing dataset')

        types = self.df[self.df['diameter_2d'].isna()]['lesion_type'].unique()
        mean_values = self.df[self.df['lesion_type'].isin(types)].groupby('lesion_type').mean(numeric_only=True).reindex(types)[
            'diameter_2d']
        self.df['diameter_2d'] = self.df['diameter_2d'].fillna(self.df['lesion_type'].map(dict(mean_values)))
        self.df = self.df[self.df['is_registered'] == 1].reset_index(drop=True)


        bboxes = np.load('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/bboxes.npy',
                         allow_pickle=True)
        bboxes_volumes = np.load('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/bboxes_volumes.npy',
                         allow_pickle=True)

        print('Computing pre and del bboxes...')

        bboxes_del = np.load(
            '/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/bboxes_del.npy',
            allow_pickle=True).item()
        bboxes_pre = np.load(
            '/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/bboxes_pre.npy',
            allow_pickle=True).item()

        bbs_del = {}
        bbs_pre = {}

        for dp, bbox in tqdm(bboxes_del.items()):
            bb = generate_bbox(bbox)
            bbs_del[dp] = bb

        for dp, bbox in tqdm(bboxes_pre.items()):
            bb = generate_bbox(bbox)
            bbs_pre[dp] = bb

        print('Done!')

        print('Generating the coordinates of the patches for each lesion...')
        bbs = []

        for bbox in tqdm(bboxes):

            bb = generate_bbox(bbox)
            bbs.append(bb)


        good_lesions = np.argwhere((bboxes[:, 1] - bboxes[:, 0] < 25) &
                                   (bboxes[:, 3] - bboxes[:, 2] < 97) &
                                   (bboxes[:, 5] - bboxes[:, 4] < 97))
        good_volumes = np.concatenate(bboxes_volumes[good_lesions])
        good_bboxes = np.concatenate(np.array(bbs)[good_lesions]).tolist()

        idx = []
        idx_bboxes = []
        for i, dp in enumerate(good_volumes):
            if dp in list(self.df['dp_lesion']):
                idx.append(self.df[self.df['dp_lesion'] == dp].index)
                idx_bboxes.append(i)

        self.df = self.df.loc[np.concatenate(idx), :].reset_index(drop=True)
        self.df['bbox'] = np.array(good_bboxes)[np.array(idx_bboxes)].tolist()

        print('df shape', self.df.shape)

        print('Coordinates ok!')

        ## we eliminate the bbox only "correct" on art or ven phase
        # keep only the bboxes that are good on both
        for x in self.df[self.df['base'] == 'calv']['dp_lesion']:
            if x.replace('VEN', 'ART') not in list(self.df['dp_lesion']):
                self.df.drop(self.df[self.df['dp_lesion'] == x].index, inplace=True)
        for x in self.df[self.df['base'] == 'calv']['dp_lesion']:
            if x.replace('ART', 'VEN') not in list(self.df['dp_lesion']):
                self.df.drop(self.df[self.df['dp_lesion'] == x].index, inplace=True)

        if training:
            self.labels = self.df[self.df[config.train_set] == 1].reset_index(drop=True)

        elif validation:
            self.labels = self.df[self.df[config.val_set] == 1].reset_index(drop=True)

        elif test:
            self.labels = self.df[self.df[config.test_set] == 1].reset_index(drop=True)

        elif chicp:
            self.labels = self.df[self.df['center'] == 'chicp'].reset_index(drop=True)

        print('self.labels shape', self.labels.shape)

        #self.labels = compute_size_batch2(self.labels)

        print('Avg HCC size', self.labels[self.labels['has_hcc'] == 1]['diameter_2d'].mean())

        print('Loading the data...')

        # you can calculate percentage of available memory
        print('available memory', psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)

        idx = [i for i in self.labels['datapoint'].index if
               self.labels.loc[i, 'datapoint'].endswith('__VEN.nii.gz') or self.labels.loc[
                   i, 'datapoint'].startswith('bat')]
        df_ven = self.labels.loc[idx, :].reset_index(drop=True)

        idx = [i for i in self.labels['datapoint'].index if
                         self.labels.loc[i, 'datapoint'].endswith('__ART.nii.gz') or str(
                             self.labels.loc[i, 'datapoint_art']).endswith('_ART.nii.gz')]
        df_art = self.labels.loc[idx, :].reset_index(drop=True)

        print(df_ven)
        print(df_art)


        ## reorder to get the exact same dataset for art and ven phases
        idxs_art = []
        idxs_ven = []
        for i, x in enumerate(df_ven['dp_lesion']):
            if x.replace('VEN', 'ART') in list(df_art['dp_lesion']):
                idxs_art.append(df_art[df_art['dp_lesion'] == x.replace('VEN', 'ART')].index.item())
                idxs_ven.append(df_ven[df_ven['dp_lesion'] == x].index.item())
            else:
                print(x)
        df_art = df_art.loc[idxs_art, :].reset_index(drop=True)
        self.df_ven = df_ven.loc[idxs_ven, :].reset_index(drop=True)


        print('dataframe ven shape', self.df_ven.shape)
        print('dataframe art shape', df_art.shape)

        for x in list(self.df_ven[self.df_ven['train_set'] == 1]['dp_lesion']):
            assert (x not in list(self.df_ven[self.df_ven['val_set'] == 1]['dp_lesion']))
        for x in list(self.df_ven[self.df_ven['val_set'] == 1]['dp_lesion']):
            assert(x not in list(self.df_ven[self.df_ven['train_set'] == 1]['dp_lesion']))

        s = self.df_ven.shape[0]

        r = list(range(0, s, s // 12))
        if r[-1] != s:
            r = r + [s]

        volumes = []
        has_hcc = []
        z_pos = []
        classes = []

        ven_patches_ = []
        art_patches_ = []
        del_patches_ = []
        pre_patches_ = []
        seg_patches_ = []
        #liverseg_patches_ = []

        delphase = []
        prephase = []
        center = []

        for i, k in enumerate(r[:-1]):

            df_ven_ = self.df_ven.loc[r[i]:r[i + 1] - 1, :]
            df_art_ = df_art.loc[r[i]:r[i + 1] - 1, :]

            ven_index = [i for i in df_ven_['datapoint'].index if
                         df_ven_.loc[i, 'datapoint'].endswith('__VEN.nii.gz') or df_ven_.loc[i, 'datapoint'].startswith(
                             'bat')]
            art_index = [i for i in df_art_['datapoint'].index if
                         df_art_.loc[i, 'datapoint'].endswith('__ART.nii.gz') or df_art_.loc[i, 'datapoint_art'].endswith('_ART.nii.gz')]

            self.df_vols_ven = df_ven_.loc[ven_index, :].reset_index(drop=True)
            self.df_vols_art = df_art_.loc[art_index, :].reset_index(drop=True)

            assert list(self.df_vols_ven['dp_lesion']) == [x.replace('ART', 'VEN') for x in list(self.df_vols_art['dp_lesion'])]

            ven_volumes = [k.replace('.nii.gz', '.npy') for k in self.df_vols_ven['datapoint']]
            art_volumes = [str(k).replace('.nii.gz', '.npy') for k in self.df_vols_art['datapoint'] if str(k).endswith('__ART.nii.gz')] + [str(k).replace('.nii.gz', '.npy') for k in self.df_vols_art['datapoint_art'] if str(k).endswith('_ART.nii.gz')]

            bboxes_ven = np.array([x for x in self.df_vols_ven['bbox']])    # list of portal venous coordinates of patches
            bboxes_art = np.array([x for x in self.df_vols_art['bbox']])    # list of arterial coordinates of patches

            print('Extracting the patches...')

            """



            print('PRE patches...')

            pre_patches = []

            for k,bbox_ven in zip(tqdm(self.df_vols_ven['dp_lesion']),bboxes_ven):

                if k.startswith('bat'):
                    point = '__'.join(k.split('__')[:-1]) + '_PRE.npy'
                    if point in os.listdir(self.config.path_to_data):
                        prephase.append(k)
                        img = np.load(join(self.config.path_to_data, point), allow_pickle=True)
                        bbox_ven = [bbox_ven[0], bbox_ven[1], bbox_ven[2] - 15,
                                    bbox_ven[3] + 15, bbox_ven[4] - 15, bbox_ven[5] + 15]
                        if bbox_ven[0] < 0 or bbox_ven[1] > img.shape[1] or bbox_ven[2] < 0 or bbox_ven[3] > img.shape[2] or bbox_ven[4] < 0 or \
                            bbox_ven[5] > img.shape[3]:
                            img = correct_bbox(img, bbox_ven)
                        else:
                            img = img[:, bbox_ven[0]:bbox_ven[1], bbox_ven[2]:bbox_ven[3], bbox_ven[4]:bbox_ven[5]]
                    else:
                        img = np.zeros((1,24,126,126))

                else:
                    point = '__'.join(k.split('__')[:-2]) + '__PRE.npy'
                    if point in os.listdir(self.config.path_to_data):
                        img = np.load(join(self.config.path_to_data, point), allow_pickle=True)
                        if k.replace('VEN','PRE') in list(bbs_pre.keys()):
                            prephase.append(k)
                            bbox = bbs_pre[k.replace('VEN','PRE')]
                            bbox = [bbox[0], bbox[1], bbox[2] - 15, bbox[3] + 15, bbox[4] - 15, bbox[5] + 15]
                            if bbox[0] < 0 or bbox[1] > img.shape[1] or bbox[2] < 0 or bbox[3] > img.shape[2] or bbox[4] < 0 or \
                                bbox[5] > img.shape[3]:
                                img = correct_bbox(img, bbox)
                            else:
                                img = img[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
                        else:
                            img = np.zeros((1, 24, 126, 126))
                    else:
                        img = np.zeros((1,24,126,126))

                pre_patches.append(img)

            print('PRE done!')
            
            """

            print('VEN patches...')
            ven_patches = []
            seg_patches = []
            #liverseg_patches = []
            for k, bbox, lesion in zip(tqdm(ven_volumes), bboxes_ven, self.df_vols_ven['lesion_id']):
                ## put here a new bbox with coordinates extended to -30 and +30
                bbox = [bbox[0], bbox[1], bbox[2] - 15, bbox[3] + 15, bbox[4] - 15, bbox[5] + 15]

                img = np.load(join(self.config.path_to_data, k), allow_pickle=True)
                seg = np.load(join(self.config.path_to_data, k.replace('.npy', '__lesions.npy')), allow_pickle=True)
                #liverseg = np.load(join(self.config.path_to_data, k.replace('.npy', '__liver.npy')), allow_pickle=True)

                if bbox[0] < 0 or bbox[1] > seg.shape[1] or bbox[2] < 0 or bbox[3] > seg.shape[2] or bbox[4] < 0 or \
                        bbox[5] > seg.shape[3]:
                    img = correct_bbox(img, bbox)
                    seg = correct_bbox(seg, bbox)
                    #liverseg = correct_bbox(liverseg, bbox)
                else:
                    img = img[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
                    seg = seg[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
                    #liverseg = liverseg[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

                seg = np.where(seg == lesion, 250, 0)
                #liverseg = np.where(liverseg == 1, 250, 0)

                ven_patches.append(img)
                seg_patches.append(seg)
                #liverseg_patches.append(liverseg)

            ven_patches = np.stack(ven_patches, axis=0)
            ven_patches_.append(ven_patches)
            seg_patches = np.stack(seg_patches, axis=0)
            seg_patches_.append(seg_patches)
            #liverseg_patches = np.stack(liverseg_patches, axis=0)
            #liverseg_patches_.append(liverseg_patches)

            print('VEN done!')

            print('ART patches...')
            art_patches = []
            artsegs = [str(k).replace('.nii.gz', '.npy') for k in self.df_vols_art['datapoint']]
            for k, bbox, s, lesion in zip(tqdm(art_volumes), bboxes_art, artsegs, self.df_vols_art['lesion_id']):

                ## put here a new bbox with coordinates extended to -30 and +30
                bbox = [bbox[0], bbox[1], bbox[2] - 15, bbox[3] + 15, bbox[4] - 15, bbox[5] + 15]
                img = np.load(join(self.config.path_to_data, k), allow_pickle=True)

                if bbox[0] < 0 or bbox[1] > img.shape[1] or bbox[2] < 0 or bbox[3] > img.shape[2] or bbox[4] < 0 or \
                        bbox[5] > img.shape[3]:
                    img = correct_bbox(img, bbox)
                else:
                    img = img[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

                art_patches.append(img)

            art_patches = np.stack(art_patches, axis=0)
            art_patches_.append(art_patches)

            print('ART done!')



            print('DEL patches...')

            del_patches = []

            for k, bbox_ven in zip(tqdm(self.df_vols_ven['dp_lesion']), bboxes_ven):

                if k.startswith('bat'):
                    point = '__'.join(k.split('__')[:-1]) + '_DEL.npy'
                    if point in os.listdir(self.config.path_to_data):
                        delphase.append(k)
                        img = np.load(join(self.config.path_to_data, point), allow_pickle=True)
                        bbox_ven = [bbox_ven[0], bbox_ven[1],
                                    bbox_ven[2] - 15, bbox_ven[3] + 15,
                                    bbox_ven[4] - 15, bbox_ven[5] + 15]
                        if bbox_ven[0] < 0 or bbox_ven[1] > img.shape[1] or bbox_ven[2] < 0 or bbox_ven[3] > img.shape[
                            2] or bbox_ven[4] < 0 or \
                                bbox_ven[5] > img.shape[3]:
                            img = correct_bbox(img, bbox_ven)
                        else:
                            img = img[:, bbox_ven[0]:bbox_ven[1], bbox_ven[2]:bbox_ven[3], bbox_ven[4]:bbox_ven[5]]
                    else:
                        img = np.zeros((1, 24, 126, 126))

                else:
                    point = '__'.join(k.split('__')[:-2]) + '__DEL.npy'
                    if point in os.listdir(self.config.path_to_data):
                        img = np.load(join(self.config.path_to_data, point), allow_pickle=True)
                        if k.replace('VEN', 'DEL') in list(bbs_del.keys()):
                            delphase.append(k)
                            bbox = bbs_del[k.replace('VEN', 'DEL')]
                            bbox = [bbox[0], bbox[1],
                                    bbox[2] - 15, bbox[3] + 15,
                                    bbox[4] - 15, bbox[5] + 15]
                            if bbox[0] < 0 or bbox[1] > img.shape[1] or bbox[2] < 0 or bbox[3] > img.shape[2] or bbox[
                                4] < 0 or \
                                    bbox[5] > img.shape[3]:
                                img = correct_bbox(img, bbox)
                            else:
                                img = img[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
                        else:
                            img = np.zeros((1, 24, 126, 126))
                    else:
                        img = np.zeros((1, 24, 126, 126))

                del_patches.append(img)

            del_patches = np.stack(del_patches, axis=0)
            del_patches_.append(del_patches)

            print('DEL done!')


            print('Patches extracted!')

            #pre_patches = np.stack(pre_patches, axis=0)
            #pre_patches_.append(pre_patches)


            self.volumes = list(self.df_vols_ven['dp_lesion'])
            self.z_pos = list(self.df_vols_ven[self.config.metadata])
            self.classes = list(self.df_vols_ven[self.config.label_name])

            if self.config.label_name == 'multilabel':
                self.has_hcc = np.array([ast.literal_eval(x) for x in self.df_vols_ven[
                    self.config.label_name]])
            else:
                self.has_hcc = list(self.df_vols_ven[self.config.label_name])
                self.center = list(self.df_vols_ven['center'])

            volumes.append(self.volumes)
            has_hcc.append(self.has_hcc)
            z_pos.append(self.z_pos)
            classes.append(self.classes)
            center.append(self.center)


        kept_cases = np.array(self.df_ven[
                                  self.df_ven['dp_lesion'].isin(
                                      set(delphase))].index)


        ven_patches = np.concatenate(ven_patches_)[kept_cases]
        #pre_patches = np.concatenate(pre_patches_)[kept_cases]
        seg_patches = np.concatenate(seg_patches_)[kept_cases]
        #liverseg_patches = np.concatenate(liverseg_patches_)[kept_cases]

        if self.config.label_name == 'npw' or self.config.label_name == 'ec' or self.config.label_name == 'has_hcc' or self.config.label_name == 'multilabel':
            art_patches = np.concatenate(art_patches_)[kept_cases]
            del_patches = np.concatenate(del_patches_)[kept_cases]
            self.slices = np.concatenate([art_patches, ven_patches, del_patches, seg_patches], axis=1)
        if self.config.label_name == 'aphe':
            art_patches = np.concatenate(art_patches_)[kept_cases]
            self.slices = np.concatenate([art_patches, ven_patches, seg_patches], axis=1)

        self.volumes = np.array(list(itertools.chain(*volumes)))[kept_cases]
        self.z_pos = np.array(list(itertools.chain(*z_pos)))[kept_cases]
        self.z_pos = ( self.z_pos - np.min(self.z_pos) ) / ( np.max(self.z_pos) - np.min(self.z_pos) )
        self.classes = np.array(list(itertools.chain(*classes)))[kept_cases]
        self.center = np.array(list(itertools.chain(*center)))[kept_cases]
        self.center = (self.center == 'chic')   # transform to a mask

        if self.config.label_name == 'multilabel':
            has_hcc = np.concatenate(has_hcc)
            self.has_hcc = has_hcc[kept_cases]  # shape N,4
        else:
            self.has_hcc = np.array(list(itertools.chain(*has_hcc)))[kept_cases]

        print('label shape', self.has_hcc.shape)
        self.n_slices = self.slices.shape[0]

        print(self.slices.shape)

        print('Data loaded!')

        if self.config.label_name == 'multilabel':

            self.class_sample_count = dict(Counter(self.has_hcc[:,-1])) # self.has_hcc
            self.weight = {k: 1/v for k, v in self.class_sample_count.items()}
            self.samples_weight = np.array([self.weight[t] for t in self.has_hcc[:,-1]]) # self.has_hcc
            self.samples_weight = torch.from_numpy(self.samples_weight)

        else:

            self.class_sample_count = dict(Counter(self.has_hcc))  # self.has_hcc
            self.weight = {k: 1 / v for k, v in self.class_sample_count.items()}
            self.samples_weight = np.array([self.weight[t] for t in self.has_hcc])  # self.has_hcc
            self.samples_weight = torch.from_numpy(self.samples_weight)

        print(self.n_slices)
        print('Class repartition', self.class_sample_count)


    def collate_fn(self,
                   list_samples):

        list_x = torch.stack(
            [torch.as_tensor(x.astype('uint8').copy(), dtype=torch.float) for (x, y, z, m, c) in list_samples], dim=0)     # dimension finale: (batch_size, 1, 512, 512)
        if self.config.label_name == 'multilabel':
            list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y, z, m, c) in list_samples], dim=0)
        else:
            list_y = torch.stack([torch.as_tensor(int(y), dtype=torch.long) for (x, y, z, m, c) in list_samples], dim=0)
        list_m = torch.stack([torch.as_tensor(m) for (x, y, z, m, c) in list_samples], dim=0)
        list_z = []
        list_c = []
        for (x, y, z, m, c) in list_samples:
            list_z.append(z)
            list_c.append(c)

        return list_x, list_y, list_z, list_m, list_c

    def __getitem__(self, idx):

        data = self.slices[idx, :]  # array of shape (B, C, D, H, W)
        subject_id = self.volumes[idx]       # subject_id: provider__ID__visit.nii.gz
        label = self.has_hcc[idx]            # on retire le label : 0 (sain) ou 1 (HCC)
        z = self.z_pos[idx]
        center = self.center[idx]

        return data, label, subject_id, z, center    # on retourne les deux images augmentées, les labels (stade), l'ID

    def __len__(self):
        return self.slices.shape[0]


