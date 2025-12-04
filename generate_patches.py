from typing import Callable, Optional, Tuple, cast, Dict, List
import numpy as np
from scipy import interpolate
from sklearn.metrics import auc
import pandas as pd

import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
# from loguru import logger
import shutil
# import cal.ds.logger as lgr
# from cal.ds.Volume import Volume

import os
import json
from glob import glob
from os.path import join, isdir, isfile, dirname

### Visualization ###
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import re

# from gia_core.core.DataAnnotation import DataAnnotation
from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import recall_score, confusion_matrix

import numpy as np
import cv2




path_to_imgs = "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new"
df = pd.read_csv(join(path_to_imgs, '1_subjects_label_hcc_deeptekcalv_chic.csv'),
                 delimiter=",").drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
df = df[df['is_registered'] == 1]
# with registered calv : df = df[df['base'] == 'deeptek' | df['base] == calv & df['datapoint'].endswith('VEN.nii.gz')]


### STEP 1: RETRIEVING ALL THE BBOXES FOR EACH LESION

print('ART and VEN bboxes...')

bboxes = []
volumes = []
deleted_lesions = []

for dp in tqdm(df['datapoint'].unique()):   # calv ven and art, and deeptek only ven

    # Read in the image with OpenCV
    seg = np.load(join(path_to_imgs, dp.replace('.nii.gz', '__lesions.npy')))

    for lesion in list(df[df['datapoint'] == dp]['lesion_id']):

        if lesion in seg:

            segmentation = np.where(seg == lesion)

            z_min = int(np.min(segmentation[1]))
            z_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[2]))
            y_max = int(np.max(segmentation[2]))
            x_min = int(np.min(segmentation[3]))
            x_max = int(np.max(segmentation[3]))

            bbox = z_min, z_max, y_min, y_max, x_min, x_max

            bboxes.append(bbox)
            volumes.append(df[(df['datapoint'] == dp) & (df['lesion_id'] == lesion)]['dp_lesion'].item())

        else:
            deleted_lesions.append([dp,str(lesion)])

np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/bboxes.npy',np.array(bboxes))
np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/bboxes_volumes.npy',np.array(volumes))
np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/deleted_lesions.npy',np.array(deleted_lesions))



#np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/bboxes_supplementary.npy',np.array(bboxes))
#np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/bboxes_volumes_supplementary.npy',np.array(volumes))
#np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/deleted_lesions_supplementary.npy',np.array(deleted_lesions))


"""



### DELAYED AND PRECONTRAST (calv only)

print('DELAYED bboxes...')


bboxes = {}
volumes = []
deleted_lesions = []

del_subjects = [x for x in os.listdir(path_to_imgs) if x.endswith('__DEL.npy')]

for dp in tqdm(del_subjects):

    # Read in the image with OpenCV
    seg = np.load(join(path_to_imgs, dp.replace('.npy', '__lesions.npy')))

    for lesion in list(df[df['datapoint'] == dp.replace('DEL.npy','VEN.nii.gz')]['lesion_id']):

        if lesion in seg:

            segmentation = np.where(seg == lesion)

            z_min = int(np.min(segmentation[1]))
            z_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[2]))
            y_max = int(np.max(segmentation[2]))
            x_min = int(np.min(segmentation[3]))
            x_max = int(np.max(segmentation[3]))

            bbox = z_min, z_max, y_min, y_max, x_min, x_max

            bboxes[dp.replace('.npy','__'+str(lesion))] = bbox
            volumes.append(dp)

        else:
            deleted_lesions.append([dp,str(lesion)])


np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/bboxes_del.npy',np.array(bboxes))
np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/bboxes_volumes_del.npy',np.array(volumes))
np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/deleted_lesions_del.npy',np.array(deleted_lesions))



print('PRECONTRAST bboxes ...')

bboxes = {}
volumes = []
deleted_lesions = []

pre_subjects = [x for x in os.listdir(path_to_imgs) if x.endswith('__PRE.npy')]

for dp in tqdm(pre_subjects):

    # Read in the image with OpenCV
    seg = np.load(join(path_to_imgs, dp.replace('.npy', '__lesions.npy')))

    for lesion in list(df[df['datapoint'] == dp.replace('PRE.npy','VEN.nii.gz')]['lesion_id']):

        if lesion in seg:

            segmentation = np.where(seg == lesion)

            z_min = int(np.min(segmentation[1]))
            z_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[2]))
            y_max = int(np.max(segmentation[2]))
            x_min = int(np.min(segmentation[3]))
            x_max = int(np.max(segmentation[3]))

            bbox = z_min, z_max, y_min, y_max, x_min, x_max

            bboxes[dp.replace('.npy','__'+str(lesion))] = bbox
            volumes.append(dp)

        else:
            deleted_lesions.append([dp,str(lesion)])


np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/bboxes_pre.npy',np.array(bboxes))
np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/bboxes_volumes_pre.npy',np.array(volumes))
np.save('/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new/deleted_lesions_pre.npy',np.array(deleted_lesions))


### STEP 2: GIVING THE "OPTIMAL" BOUNDING BOX LESION (decided somewhere else)

#x, y, z =


"""