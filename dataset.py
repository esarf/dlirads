
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


### DATASET.PY POUR ENTRAINEMENT ECR AVEC FULL IMAGES


def filtrate_lesion(df,threshold):

    art_index = [i for i in df['datapoint'].index if
                 df.loc[i, 'datapoint'].endswith('__ART.nii.gz')]
    df_vols = df.loc[art_index, :].reset_index()

    df_vols = df_vols.sort_values('diameter_2d', ascending=False).reset_index(drop=True)

    df_bis = df_vols[df_vols['diameter_2d'] > threshold]

    for idx in tqdm(df_vols[(df_vols['has_hcc'] == 1) & (df_vols['diameter_2d'] <= threshold)].index):

        df_bis = pd.concat([df_bis, df_vols.loc[[idx]]])

        for id_ in df_vols[df_vols['has_hcc'] == 0].index:

            if df_vols.loc[id_, 'dp_lesion'] in list(df_bis['dp_lesion']):
                pass

            else:
                if np.abs(df_vols.loc[id_, 'diameter_2d'] - df_vols.loc[idx, 'diameter_2d']) < 0.1:
                    df_bis = pd.concat([df_bis, df_vols.loc[[id_]]]).drop_duplicates()

                    break

    df_final = df.loc[df['dp_lesion'].isin(df_bis['dp_lesion'])].reset_index(drop=True)

    return df_final


class DatasetCL(Dataset):  # output size : (N,(512,512,2),1,1)

    def __init__(self, config, training=False, validation=False, grenoble=False, dimension=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert training != validation

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
                self.df = pd.read_csv(join(self.config.lght_dir, '1_subjects_label_hcc_calv_trainval_cv.csv'), delimiter=",").drop(['Unnamed: 0'], axis=1)
                #self.df = self.df[self.df['annotation_confidence'].isin(
                #    ['2_high', '3_follow_up', '4_pathology_proven'])].reset_index().drop(['index'], axis=1)
        else:
            if self.config.database == 'grenoble':
                self.df = pd.read_csv(join(self.config.path_to_data, '1_subjects_label_hcc_grenoble.csv'), delimiter=",").drop(['Unnamed: 0'], axis=1)
                dct = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
                self.df['label'] = [*map(dct.get, self.df['class'])]
            else:
                #self.df = pd.read_csv(join(self.config.path_to_data, '1_subjects_label_hcc_calv_trainval.csv'), delimiter=",").drop(['Unnamed: 0'], axis=1)
                #self.df = pd.read_csv(join(self.config.path_to_data, '1_subjects_label_hcc_deeptekcalv.csv'),
                #                      delimiter=",").drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
                self.df = pd.read_csv(join(self.config.path_to_data, '1_subjects_label_hcc_deeptekcalv_chic.csv'),
                                      delimiter=",").drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
                types = self.df[self.df['diameter_2d'].isna()]['lesion_type'].unique()
                mean_values = self.df[self.df['lesion_type'].isin(types)].groupby('lesion_type').mean(numeric_only=True).reindex(types)[
                    'diameter_2d']
                self.df['diameter_2d'] = self.df['diameter_2d'].fillna(self.df['lesion_type'].map(dict(mean_values)))

                #self.df = pd.read_csv(join(self.config.path_to_data, '1_subjects_label_hcc_calv.csv'),
                #                      delimiter=",").drop(['Unnamed: 0'], axis=1)
                #self.df = self.df[self.df['annotation_confidence'].isin(
                #    ['2_high', '3_follow_up', '4_pathology_proven'])].reset_index().drop(['index'], axis=1)

                # inclusion criteria: add until equal avg
                #print('Filtrating lesion sizes...')
                #self.df = filtrate_lesion(self.df,25)
                #print('Done!')
                # inclusion criteria: cirrhotic patients
                #self.df = self.df[self.df['cirrhosis'] > 0]
                # inclusion criteria: lesion size
                #self.df = self.df[self.df['diameter_2d'] > 25]

        if training:
            #self.df = self.df[self.df['annotation_confidence'].isin(['2_medium','3_high', # deeptek
            #                                                         '2_high', '3_follow_up', '4_pathology_proven'])] # calv
            self.labels = self.df[self.df[config.train_set] == 1].reset_index(drop=True)
            if self.config.database == 'grenoble':
                self.path = self.config.path_to_data
            else:
                self.path = join(self.config.path_to_data, 'train')
        elif validation:
            #self.df = self.df[self.df['annotator_name'] == 'HCC01']    # full validation set: both annotators 1 & 2
            self.labels = self.df[self.df[config.val_set] == 1].reset_index(drop=True)
            if self.config.database == 'grenoble':
                self.path = self.config.path_to_data
            else:
                self.path = join(self.config.path_to_data, 'validation')


        print('columns in the dataset',self.df.columns)


        print('Loading the data...')

        # you can calculate percentage of available memory
        print('available memory',psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)

        if self.config.exp == 'oneseg':

            ## add a column "is_registered" with a boolean

            self.labels = self.labels[self.labels['is_registered'] == 1]
            #self.labels = self.labels[(self.labels['base'] == 'calv') | ((self.labels['base'] == 'deeptek') & (self.labels['datapoint_art'].notnull()))]

            idx = [i for i in self.labels['datapoint'].index if
                   self.labels.loc[i, 'datapoint'].endswith('__VEN.nii.gz') or self.labels.loc[
                       i, 'datapoint'].startswith('bat')]
            df_ven = self.labels.loc[idx, :].reset_index(drop=True)

            idx = [i for i in self.labels['datapoint'].index if
                             self.labels.loc[i, 'datapoint'].endswith('__ART.nii.gz') or str(self.labels.loc[i, 'datapoint_art']).endswith('_ART.nii.gz')]
            df_art = self.labels.loc[idx, :].reset_index(drop=True)

            s = df_ven.shape[0]

            r = list(range(0, s, s // 8))
            if r[-1] != s:
                r = r + [s]
            ven_seg_ = []
            ven_slices_ = []
            art_slices_ = []
            volumes = []
            has_hcc = []
            z_pos = []
            classes = []

            art_arrays = []

            for i, k in enumerate(r[:-1]):

                # remplacer df_ par self.labels

                df_ven_ = df_ven.loc[r[i]:r[i + 1] - 1, :]
                df_art_ = df_art.loc[r[i]:r[i + 1] - 1, :]

                ven_index = [i for i in df_ven_['datapoint'].index if
                             df_ven_.loc[i, 'datapoint'].endswith('__VEN.nii.gz') or df_ven_.loc[i, 'datapoint'].startswith(
                                 'bat')]
                art_index = [i for i in df_art_['datapoint'].index if
                             df_art_.loc[i, 'datapoint'].endswith('__ART.nii.gz') or df_art_.loc[i, 'datapoint_art'].endswith('_ART.nii.gz')]

                self.df_vols_ven = df_ven_.loc[ven_index, :].reset_index(drop=True)
                self.df_vols_art = df_art_.loc[art_index, :].reset_index(drop=True)

                ven_volumes = [k.replace('.nii.gz', '.npy') for k in self.df_vols_ven['datapoint']]
                art_volumes = [str(k).replace('.nii.gz', '.npy') for k in self.df_vols_art['datapoint'] if str(k).endswith('__ART.nii.gz')] + [str(k).replace('.nii.gz', '.npy') for k in self.df_vols_art['datapoint_art'] if str(k).endswith('_ART.nii.gz')]

                slices_ven_idx = self.df_vols_ven['z_large']
                slices_art_idx = self.df_vols_art['z_large']

                ven_arrays = [np.load(join(self.config.path_to_data, k), allow_pickle=True)[:, :, i] for k, i in
                              zip(tqdm(ven_volumes), slices_ven_idx)]

                art_arrays = [np.load(join(self.config.path_to_data, k), allow_pickle=True)[:, :, i] for k, i in
                              zip(tqdm(art_volumes), slices_art_idx)]

                ven_seg = [
                    np.load(join(self.config.path_to_data, k.replace('.npy', '__lesions.npy')), allow_pickle=True)[:,
                    :, i] for k, i in
                    zip(tqdm(ven_volumes), slices_ven_idx)]

                ven_seg = np.dstack(ven_seg)
                ven_slices = np.dstack(ven_arrays)
                ven_seg_.append(ven_seg)
                ven_slices_.append(ven_slices)

                art_slices = np.dstack(art_arrays)
                art_slices_.append(art_slices)

                print('ven slices shape',ven_slices.shape)
                print('art slices shape',art_slices.shape)
                print('ven seg shape',ven_seg.shape)

                self.volumes = [x.replace('.nii.gz', '__' + str(y)) for x, y in
                                zip(list(self.df_vols_ven['datapoint']), list(self.df_vols_ven['lesion_id']))]
                self.has_hcc = list(self.df_vols_ven[self.config.label_name])
                self.z_pos = list(self.df_vols_ven[self.config.metadata])
                self.classes = list(self.df_vols_ven[self.config.label_name])

                volumes.append(self.volumes)
                has_hcc.append(self.has_hcc)
                z_pos.append(self.z_pos)
                classes.append(self.classes)

                print('Ceiling lesion segmentations to 1...')
                for i, x in enumerate(tqdm(self.df_vols_ven['datapoint'])):
                    seg = ven_seg[:, :, i]
                    seg = np.where(seg != self.df_vols_ven.loc[i, 'lesion_id'], 0, seg)
                    seg = np.where(seg > 0, 250, seg)
                    ven_seg[:, :, i] = seg
                print('Done!')

            ven_slices = np.dstack(ven_slices_)
            art_slices = np.dstack(art_slices_)
            ven_seg = np.dstack(ven_seg_)
            self.volumes = list(itertools.chain(*volumes))
            self.has_hcc = list(itertools.chain(*has_hcc))
            self.z_pos = list(itertools.chain(*z_pos))
            self.classes = list(itertools.chain(*classes))

            self.slices = np.stack([art_slices, ven_slices, ven_seg], axis=0)
            self.n_slices = self.slices.shape[-1]

            print(self.slices.shape)

        elif self.config.exp == 'biseg':

            ## add a column "is_registered" with a boolean

            self.labels = self.labels[self.labels['is_registered'] == 1]
            #self.labels = self.labels[(self.labels['base'] == 'calv') | ((self.labels['base'] == 'deeptek') & (self.labels['datapoint_art'].notnull()))]

            idx = [i for i in self.labels['datapoint'].index if
                   self.labels.loc[i, 'datapoint'].endswith('__VEN.nii.gz') or self.labels.loc[
                       i, 'datapoint'].startswith('bat')]
            df_ven = self.labels.loc[idx, :].reset_index(drop=True)

            idx = [i for i in self.labels['datapoint'].index if
                             self.labels.loc[i, 'datapoint'].endswith('__ART.nii.gz') or str(self.labels.loc[i, 'datapoint_art']).endswith('_ART.nii.gz')]
            df_art = self.labels.loc[idx, :].reset_index(drop=True)

            s = df_ven.shape[0]

            r = list(range(0, s, s // 8))
            if r[-1] != s:
                r = r + [s]
            ven_seg_ = []
            art_seg_ = []
            ven_slices_ = []
            art_slices_ = []
            volumes = []
            has_hcc = []
            z_pos = []
            classes = []

            art_arrays = []

            for i, k in enumerate(r[:-1]):

                # remplacer df_ par self.labels

                df_ven_ = df_ven.loc[r[i]:r[i + 1] - 1, :]
                df_art_ = df_art.loc[r[i]:r[i + 1] - 1, :]

                ven_index = [i for i in df_ven_['datapoint'].index if
                             df_ven_.loc[i, 'datapoint'].endswith('__VEN.nii.gz') or df_ven_.loc[i, 'datapoint'].startswith(
                                 'bat')]
                art_index = [i for i in df_art_['datapoint'].index if
                             df_art_.loc[i, 'datapoint'].endswith('__ART.nii.gz') or df_art_.loc[i, 'datapoint_art'].endswith('_ART.nii.gz')]

                self.df_vols_ven = df_ven_.loc[ven_index, :].reset_index(drop=True)
                self.df_vols_art = df_art_.loc[art_index, :].reset_index(drop=True)

                ven_volumes = [k.replace('.nii.gz', '.npy') for k in self.df_vols_ven['datapoint']]
                art_volumes = [str(k).replace('.nii.gz', '.npy') for k in self.df_vols_art['datapoint'] if str(k).endswith('__ART.nii.gz')] + [str(k).replace('.nii.gz', '.npy') for k in self.df_vols_art['datapoint_art'] if str(k).endswith('_ART.nii.gz')]

                slices_ven_idx = self.df_vols_ven['z_large']
                slices_art_idx = self.df_vols_art['z_large']

                ven_arrays = []
                art_arrays = []
                ven_seg = []
                art_seg = []

                for (k, i) in zip(tqdm(ven_volumes), slices_ven_idx):
                    data = np.load(join(self.config.path_to_data, k), allow_pickle=True)
                    z = min(data.shape[-1],i+1)
                    ven_arrays.append(data[:, :, [i-1,i,z-1]])
                    ven_seg.append(np.load(join(self.config.path_to_data, k.replace('.npy', '__lesions.npy')), allow_pickle=True)[:,
                    :, [i-1,i,z-1]])

                for (k, i) in zip(tqdm(art_volumes), slices_art_idx):
                    data = np.load(join(self.config.path_to_data, k), allow_pickle=True)
                    z = min(data.shape[-1],i+1)
                    art_arrays.append(data[:, :, [i-1,i,z-1]])

                artsegs = [str(k).replace('.nii.gz', '.npy') for k in self.df_vols_art['datapoint'] ]

                for (k, i) in zip(tqdm(artsegs), slices_art_idx):
                    data = np.load(join(self.config.path_to_data, k.replace('.npy', '__lesions.npy')), allow_pickle=True)
                    z = min(data.shape[-1], i + 1)
                    art_seg.append(data[:, :, [i-1,i,z-1]])

                ven_seg = np.stack(ven_seg,axis=-1)
                ven_slices = np.stack(ven_arrays,axis=-1)
                ven_seg_.append(ven_seg)
                ven_slices_.append(ven_slices)
                art_seg = np.stack(art_seg,axis=-1)
                art_slices = np.stack(art_arrays,axis=-1)
                art_seg_.append(art_seg)
                art_slices_.append(art_slices)

                self.volumes = [x.replace('.nii.gz', '__' + str(y)) for x, y in
                                zip(list(self.df_vols_ven['datapoint']), list(self.df_vols_ven['lesion_id']))]
                self.has_hcc = list(self.df_vols_ven[self.config.label_name])
                self.z_pos = list(self.df_vols_ven[self.config.metadata])
                self.classes = list(self.df_vols_ven[self.config.label_name])

                volumes.append(self.volumes)
                has_hcc.append(self.has_hcc)
                z_pos.append(self.z_pos)
                classes.append(self.classes)

                print('Ceiling portal lesion segmentations to 1...')
                for i, x in enumerate(tqdm(self.df_vols_ven['datapoint'])):
                    seg = ven_seg[:, :, :, i]
                    seg = np.where(seg != self.df_vols_ven.loc[i,'lesion_id'], 0, seg)
                    seg = np.where(seg > 0, 250, seg)
                    ven_seg[:, :, :, i] = seg
                print('Done!')

                print('Ceiling arterial lesion segmentations to 1...')
                for i, x in enumerate(tqdm(self.df_vols_art['datapoint'])):
                    seg = art_seg[:, :, :, i]
                    seg = np.where(seg != self.df_vols_art.loc[i, 'lesion_id'], 0, seg)
                    seg = np.where(seg > 0, 250, seg)
                    art_seg[:, :, :, i] = seg
                print('Done!')

            ven_slices = np.concatenate(ven_slices_,axis=-1)
            art_slices = np.concatenate(art_slices_,axis=-1)
            ven_seg = np.concatenate(ven_seg_,axis=-1)
            art_seg = np.concatenate(art_seg_,axis=-1)

            print('ven slices shape', ven_slices.shape)
            print('art slices shape', art_slices.shape)
            print('ven seg shape', ven_seg.shape)
            print('art seg shape', ven_seg.shape)

            self.volumes = list(itertools.chain(*volumes))
            self.has_hcc = list(itertools.chain(*has_hcc))
            self.z_pos = list(itertools.chain(*z_pos))
            self.classes = list(itertools.chain(*classes))

            self.slices = np.moveaxis(np.concatenate([art_slices, art_seg, ven_slices, ven_seg], axis=2),2,0)
            self.n_slices = self.slices.shape[-1]

            print(self.slices.shape)

        elif self.config.exp == 'venseg':

            idx = [i for i in self.labels['datapoint'].index if
             self.labels.loc[i, 'datapoint'].endswith('__VEN.nii.gz') or self.labels.loc[i, 'datapoint'].startswith('bat')]
            df = self.labels.loc[idx,:].reset_index(drop=True)
            s = df.shape[0]

            r = list(range(0, s, s // 8))
            if r[-1] != s:
                r = r + [s]
            ven_seg_ = []
            ven_slices_ = []
            volumes = []
            has_hcc = []
            z_pos = []
            classes = []

            for i, k in enumerate(r[:-1]):

                # remplacer df_ par self.labels

                df_ = df.loc[r[i]:r[i + 1] - 1, :]

                ven_index = [i for i in df_['datapoint'].index if
                             df_.loc[i, 'datapoint'].endswith('__VEN.nii.gz') or df_.loc[i, 'datapoint'].startswith('bat')]

                self.df_vols = df_.loc[ven_index, :].reset_index()
                ven_volumes = [k.replace('.nii.gz', '.npy') for k in self.df_vols['datapoint']]
                slices_ven_idx = self.df_vols['z_large']

                ven_arrays = [np.load(join(self.config.path_to_data, k), allow_pickle=True)[:, :, i] for k, i in
                                  zip(tqdm(ven_volumes), slices_ven_idx)]
                ven_seg = [
                    np.load(join(self.config.path_to_data, k.replace('.npy', '__lesions.npy')), allow_pickle=True)[:,
                    :, i] for k, i in
                    zip(tqdm(ven_volumes), slices_ven_idx)]

                ven_seg = np.dstack(ven_seg)
                ven_slices = np.dstack(ven_arrays)

                ven_seg_.append(ven_seg)
                ven_slices_.append(ven_slices)

                self.volumes = [x.replace('.nii.gz', '__' + str(y)) for x, y in
                                zip(list(self.df_vols['datapoint']), list(self.df_vols['lesion_id']))]
                self.has_hcc = list(self.df_vols[self.config.label_name])
                self.z_pos = list(self.df_vols[self.config.metadata])
                self.classes = list(self.df_vols[self.config.label_name])

                volumes.append(self.volumes)
                has_hcc.append(self.has_hcc)
                z_pos.append(self.z_pos)
                classes.append(self.classes)

                print('Ceiling lesion segmentations to 1...')
                for i,x in enumerate(tqdm(self.df_vols['datapoint'])):
                    seg = ven_seg[:,:,i]
                    seg = np.where(seg != self.df_vols.loc[i,'lesion_id'], 0, seg)
                    seg = np.where(seg > 0, 250, seg)
                    ven_seg[:,:,i] = seg
                print('Done!')

            ven_slices = np.dstack(ven_slices_)
            ven_seg = np.dstack(ven_seg_)
            self.volumes = list(itertools.chain(*volumes))
            self.has_hcc = list(itertools.chain(*has_hcc))
            self.z_pos = list(itertools.chain(*z_pos))
            self.classes = list(itertools.chain(*classes))

            self.slices = np.stack([ven_slices, ven_seg], axis=0)
            self.n_slices = self.slices.shape[-1]



        elif self.config.exp == 'artseg':

            self.labels = self.labels[self.labels['is_registered'] == 1]

            idx = [i for i in self.labels['datapoint'].index if
                   self.labels.loc[i, 'datapoint'].endswith('__ART.nii.gz') or str(
                       self.labels.loc[i, 'datapoint_art']).endswith('_ART.nii.gz')]
            df = self.labels.loc[idx, :].reset_index(drop=True)
            s = df.shape[0]
            r = list(range(0, s, s // 8))
            if r[-1] != s:
                r = r + [s]

            art_seg_ = []
            art_slices_ = []
            volumes = []
            has_hcc = []
            z_pos = []
            classes = []

            for i, k in enumerate(r[:-1]):

                # remplacer df_ par self.labels

                df_ = df.loc[r[i]:r[i + 1] - 1, :]
                art_index = [i for i in df_['datapoint'].index if
                             df_.loc[i, 'datapoint'].endswith('__ART.nii.gz') or df_.loc[
                                 i, 'datapoint_art'].endswith('_ART.nii.gz')]

                self.df_vols = df_.loc[art_index, :].reset_index()
                art_volumes = [str(k).replace('.nii.gz', '.npy') for k in self.df_vols['datapoint'] if str(k).endswith('__ART.nii.gz')] + [str(k).replace('.nii.gz', '.npy') for k in self.df_vols['datapoint_art'] if str(k).endswith('_ART.nii.gz')]
                slices_art_idx = self.df_vols['z_large']
                art_arrays = [np.load(join(self.config.path_to_data, k), allow_pickle=True)[:, :, i] for k, i in
                              zip(tqdm(art_volumes), slices_art_idx)]


                artsegs = [str(k).replace('.nii.gz', '.npy') for k in self.df_vols['datapoint']]
                art_seg = [
                    np.load(join(self.config.path_to_data, k.replace('.npy', '__lesions.npy')), allow_pickle=True)[:,
                    :, i] for k, i in
                    zip(tqdm(artsegs), slices_art_idx)]

                art_seg = np.dstack(art_seg)
                art_slices = np.dstack(art_arrays)
                art_seg_.append(art_seg)
                art_slices_.append(art_slices)

                self.volumes = [x.replace('.nii.gz', '__' + str(y)) for x, y in
                                zip(list(self.df_vols['datapoint']), list(self.df_vols['lesion_id']))]
                self.has_hcc = list(self.df_vols[self.config.label_name])
                self.z_pos = list(self.df_vols[self.config.metadata])
                self.classes = list(self.df_vols[self.config.label_name])

                volumes.append(self.volumes)
                has_hcc.append(self.has_hcc)
                z_pos.append(self.z_pos)
                classes.append(self.classes)


                print('Ceiling lesion segmentations to 1...')

                for i, x in enumerate(tqdm(self.df_vols['datapoint'])):
                    seg = art_seg[:, :, i]
                    seg = np.where(seg != self.df_vols.loc[i, 'lesion_id'], 0, seg)
                    seg = np.where(seg > 0, 250, seg)
                    art_seg[:, :, i] = seg

                print('Done!')

            art_slices = np.dstack(art_slices_)
            art_seg = np.dstack(art_seg_)

            self.volumes = list(itertools.chain(*volumes))
            self.has_hcc = list(itertools.chain(*has_hcc))
            self.z_pos = list(itertools.chain(*z_pos))
            self.classes = list(itertools.chain(*classes))

            self.slices = np.stack([art_slices, art_seg], axis=0)

            self.n_slices = self.slices.shape[-1]


        print('Data loaded!')

        #u = dict(Counter(self.volumes))
        #self.z_pos = list(itertools.chain(*[[x / i for x in list(range(0, i))] for i in u.values()]))

        #self.z_pos = list(self.df_vols['z_center'])

        if validation:
            # if validation and 2D, we shuffle the dataset once
            # otherwise for each batch you will have slices from the same patient
            # so exp(z_i,z_j) / sum_k(exp(z_i,z_k)) will be very low, so high loss value
            l = list(range(self.n_slices))
            self.shuffled_idx = random.sample(l, len(l))
            self.slices = self.slices[:, :, :, self.shuffled_idx]
            self.volumes = list(itemgetter(*self.shuffled_idx)(self.volumes))
            self.z_pos = list(itemgetter(*self.shuffled_idx)(self.z_pos))
            self.has_hcc = list(itemgetter(*self.shuffled_idx)(self.has_hcc))

        #uncomment the following line for custom sampler (weight per volume and not per slice)

        #self.classes = list(self.df_vols[self.config.label_name])
        #self.classes = [int(self.labels.loc[x, 'class']) for x in self.volumes]
        #self.class_sample_count = np.array(
        #    [len(np.where(self.classes == t)[0]) for t in np.unique(self.classes)])

        self.class_sample_count = dict(Counter(self.has_hcc))

        print(self.class_sample_count)

        self.weight = {k: 1/v for k, v in self.class_sample_count.items()}
        self.samples_weight = np.array([self.weight[t] for t in self.has_hcc])
        self.samples_weight = torch.from_numpy(self.samples_weight)

        #self.label_slice = [int(self.labels.loc[x, 'has_hcc']) for x in self.volumes]

        """self.indices = {}  # indices of all the slice indices for each volume (patient) in the dataset
        for z in self.volumes:
            self.indices[z] = [i for i, x in enumerate(self.volumes) if x == z]"""

        print(self.n_slices)
        print('Class repartition',self.class_sample_count)


    def collate_fn(self,
                   list_samples):

        list_x = torch.stack(
            [torch.as_tensor(x.astype('uint8').copy(), dtype=torch.float) for (x, y, z, m) in list_samples], dim=0)     # dimension finale: (batch_size, 1, 512, 512)
        if self.config.pretrained:
            list_x = torch.repeat_interleave(list_x, 3, dim=1)
        list_y = torch.stack([torch.as_tensor(int(y), dtype=torch.long) for (x, y, z, m) in list_samples], dim=0)
        list_m = torch.stack([torch.as_tensor(m) for (x, y, z, m) in list_samples], dim=0)
        list_z = []
        for (x, y, z, m) in list_samples:
            list_z.append(z)

        return list_x, list_y, list_z, list_m

    def __getitem__(self, idx):

        data = self.slices[:, :, :, idx]  # on retire une matrice de dimension (2,H,W)
        subject_id = self.volumes[idx]    # subject_id: provider__ID__visit.nii.gz
        label = self.has_hcc[idx]
        #print(self.labels[self.labels['datapoint'] == '__'.join(subject_id.split('__')[:4])+'.nii.gz'][self.config.label_name].unique())
        #label = self.labels[self.labels['datapoint'] == '__'.join(subject_id.split('__')[:4])+'.nii.gz'][self.config.label_name].unique().item()

        # on retire le label : 0 (sain) ou 1 (HCC)
        z = self.z_pos[idx]
        # mask = self.masks[:,:,idx][np.newaxis]
        # data = np.vstack((data, mask))                             # shape (2,H,W)

        return data, label, subject_id, z  # on retourne les deux images augmentées, les labels (stade), l'ID

    def __len__(self):
        # return math.floor(self.slices.shape[-1] / 2.) * 2
        return self.slices.shape[-1]


