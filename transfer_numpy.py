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

path_to_nib = "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/3_calv_ct_hcc__volume"
path_to_numpy = "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy"

del_subjects = [x for x in os.listdir(path_to_nib) if x.endswith('DEL.nii.gz')]

for k in tqdm([x.replace('.nii.gz', '__liver.nii.gz') for x in del_subjects]):
    mat = np.asanyarray(nib.load(join(path_to_nib, k)).dataobj)
    np.save(join(path_to_numpy, k.replace('.nii.gz', '.npy')), mat)

for k in tqdm([x.replace('.nii.gz', '__lesions.nii.gz') for x in del_subjects]):
    mat = np.asanyarray(nib.load(join(path_to_nib, k)).dataobj)
    np.save(join(path_to_numpy, k.replace('.nii.gz', '.npy')), mat)

for k in tqdm(del_subjects):
    mat = np.asanyarray(nib.load(join(path_to_nib, k)).dataobj)
    np.save(join(path_to_numpy, k.replace('.nii.gz', '.npy')), mat)


del_subjects = [x for x in os.listdir(path_to_nib) if x.endswith('PRE.nii.gz')]

for k in tqdm([x.replace('.nii.gz', '__liver.nii.gz') for x in del_subjects]):
    mat = np.asanyarray(nib.load(join(path_to_nib, k)).dataobj)
    np.save(join(path_to_numpy, k.replace('.nii.gz', '.npy')), mat)

for k in tqdm([x.replace('.nii.gz', '__lesions.nii.gz') for x in del_subjects]):
    mat = np.asanyarray(nib.load(join(path_to_nib, k)).dataobj)
    np.save(join(path_to_numpy, k.replace('.nii.gz', '.npy')), mat)

for k in tqdm(del_subjects):
    mat = np.asanyarray(nib.load(join(path_to_nib, k)).dataobj)
    np.save(join(path_to_numpy, k.replace('.nii.gz', '.npy')), mat)