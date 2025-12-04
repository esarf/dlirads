#import torch.cuda


from dataset3d import DatasetCL
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, DistributedSampler
from losses import GeneralizedSupervisedNTXenLoss, NTXenLoss, SupConLoss, SupSimLoss, ySimLoss, DINOLoss, WeightedMultilabelLoss
from torch.nn import CrossEntropyLoss
import cProfile
import torch.distributed as dist
import itertools
import models.network as model_
import time
from sampler import CustomSampler, DistributedSamplerWrapper, CustomWeightedRandomSampler
import argparse
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from config import Config
from yAwareContrastiveLearning import yAwareCLModel
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, train_test_split
#from ContrastiveLearning3D import CLModel3D
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.trainer.states import RunningStage
from os.path import join
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
import warnings
import os
from dino import DINOModel
import idr_torch
import random

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning", "autoencoder"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly!")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--cross_val", dest="cross_val", action="store_true")
    parser.add_argument("--no-cross_val", dest="cross_val", action="store_false")
    parser.add_argument("--n_fold", type=int)
    parser.add_argument("--database", type=str)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--dir", type=str)
    parser.add_argument("--pretrained_path", default=None, type=str)
    parser.add_argument("--sigma", default=0.5, type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--label_name", default="label", type=str)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--kernel", default='rbf', type=str)
    parser.add_argument("--max_epochs", default=40, type=int)
    parser.add_argument("--grenoble", dest="grenoble", action="store_true")
    parser.add_argument("--model_type", default="ysimclr", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--rep_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--output_dim", default=64, type=int)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--segmask", dest="segmask", action="store_true")
    parser.add_argument("--exp", default="baseline", type=str)
    parser.add_argument("--input_dim", default=2, type=int)
    parser.add_argument("--metadata", default='diameter_2d', type=str)
    parser.add_argument("--hidden_dim_vit", default=512, type=int)
    parser.add_argument("--num_heads", default=2, type=int)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--augment", default='torch', type=str)
    parser.add_argument("--experiment", default=None, type=str)

    #parser.set_defaults(cross_val=True)

    args = parser.parse_args()

    print(args.mode)
    print(args.lr)
    print(args.weight_decay)
    print(args.cross_val)
    print(args.n_fold)
    print(args.database)
    print(args.encoder)
    print(args.pretrained_path)
    print(args.sigma)
    print(args.temperature)
    print(args.label_name)
    print(args.kernel)
    print(args.max_epochs)
    print(args.grenoble)
    print(args.batch_size)
    print(args.exp)
    print(args.metadata)
    print(args.model_type)
    print(args.augment)

    mode = args.mode
    lr = args.lr
    weight_decay = args.weight_decay
    cross_val = args.cross_val
    n_fold = args.n_fold
    database = args.database
    encoder = args.encoder
    dir = args.dir
    pretrained_path = args.pretrained_path
    sigma = args.sigma
    temperature = args.temperature
    label_name = args.label_name
    num_classes = args.num_classes
    kernel = args.kernel
    max_epochs = args.max_epochs
    grenoble = args.grenoble
    model_type = args.model_type
    batch_size = args.batch_size
    pretrained = args.pretrained
    rep_dim = args.rep_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    segmask = args.segmask
    exp = args.exp
    input_dim = args.input_dim
    metadata = args.metadata
    hidden_dim_vit = args.hidden_dim_vit
    num_heads = args.num_heads
    num_layers = args.num_layers
    patch_size = args.patch_size
    augment = args.augment
    experiment = args.experiment

    config = Config(mode=mode,
                    lr=lr,
                    weight_decay=weight_decay,
                    cross_val=cross_val,
                    n_fold=n_fold,
                    database=database,
                    encoder=encoder,
                    dir=dir,
                    pretrained_path=pretrained_path,
                    sigma=sigma,
                    temperature=temperature,
                    label_name=label_name,
                    num_classes=num_classes,
                    kernel=kernel,
                    max_epochs=max_epochs,
                    grenoble=grenoble,
                    model_type=model_type,
                    batch_size=batch_size,
                    rep_dim=rep_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    pretrained=pretrained,
                    segmask=segmask,
                    exp=exp,
                    input_dim=input_dim,
                    metadata=metadata,
                    hidden_dim_vit=hidden_dim_vit,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    patch_size=patch_size,
                    augment=augment,
                    experiment=experiment
                    )


    print('label name', config.label_name)


    if config.cross_val:

        if config.database == 'grenoble':
            df = pd.read_csv(join(config.path_to_data, '1_subjects_label_hcc_grenoble.csv'), delimiter=",").drop(
                ['Unnamed: 0'], axis=1)
            dct = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
            df['label'] = [*map(dct.get, df['class'])]
        else:
            _df = pd.read_csv(join(config.path_to_data, '1_subjects_label_hcc_deeptekcalv_chic.csv'), delimiter=",").drop(['Unnamed: 0'], axis=1)

            _df['test_set'] = 0
            idx_calv = np.array(_df[_df['base'] == 'calv'].index)
            #idx_calv = np.array(_df[_df['center'] == 'chicp'].index)
            _df.loc[idx_calv, 'train_set'] = 0
            _df.loc[idx_calv, 'val_set'] = 0
            _df.loc[idx_calv, 'test_set'] = 1

            #### FOR CALV TRAINING ONLY
            #dplesion_calv_ven = np.load(
            #    '/lustre/fswork/projects/rech/cwn/ufd78nr/emma/2022_contrastive/3_models/finetuning_hcc/supervised_chic_cv_artvendel_small_calvtest_new_dropout/_test_volumes0.npy',
            #allow_pickle=True)
            #dplesion_calv_art = [x.replace('VEN', 'ART') for x in dplesion_calv_ven]
            #calv = [*dplesion_calv_ven, *dplesion_calv_art]
            #df = _df[_df['dp_lesion'].isin(calv)].reset_index(drop=True)
            ##### END CALV TRAINING

            vols = ['bat-chic-1_sub-CH10121_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10345_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10355_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10355_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10417_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10177_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10326_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10178_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10030_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10467_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10114_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20133_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10316_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10316_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10170_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10170_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10311_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10134_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20001_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10432_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10071_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10122_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20047_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20068_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20012_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20093_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10224_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10284_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10284_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20029_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10238_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10238_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10444_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10261_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10261_ses-00002_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10328_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10328_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20102_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10169_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10369_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10369_ses-00002_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10182_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10487_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10487_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10247_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10193_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10193_ses-00002_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10436_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10488_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10283_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10283_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10283_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10416_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10116_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10202_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10266_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10266_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10050_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10258_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10216_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10356_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10356_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10429_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10372_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10372_ses-00002_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10411_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10032_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10427_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10427_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10257_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10257_ses-00002_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20017_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20017_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20017_ses-00001_pro-CT-Portal__3',
             'bat-chic-2_sub-CH20125_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10110_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10060_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10083_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20099_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10314_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10112_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10401_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10401_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10401_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10387_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10127_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10127_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10442_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10013_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10268_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10023_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10023_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10041_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10041_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10101_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10101_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10230_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10230_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10108_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10379_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10379_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10128_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10317_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10452_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20089_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10087_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20141_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10338_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10338_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10027_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10063_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10063_ses-00002_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10063_ses-00002_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10149_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10457_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10018_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10021_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10290_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10484_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10484_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10385_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10366_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10201_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10201_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10020_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10020_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10111_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10478_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10089_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10300_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10142_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10354_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10277_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10277_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10176_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10341_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10485_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10327_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10327_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20028_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20085_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20085_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20085_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10073_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10073_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10282_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10489_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10148_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10148_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20130_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20065_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10399_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10362_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10251_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10251_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10394_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10410_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10410_ses-00002_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10458_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20075_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10461_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10453_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10453_ses-00002_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10154_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10094_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10094_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10472_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10472_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10472_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10439_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10439_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10194_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10364_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10364_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10364_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10164_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10315_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10315_ses-00002_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10099_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10208_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10208_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10223_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10223_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10009_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10070_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10390_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10351_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10186_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10186_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10253_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10253_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10253_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10167_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10190_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10190_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10162_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10307_ses-00002_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20091_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10483_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10329_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10126_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10289_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10476_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10273_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10273_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10273_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10305_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10481_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10209_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10209_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10082_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10241_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10034_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10061_ses-00002_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20114_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10373_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10373_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10173_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10240_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10378_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10378_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10285_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10382_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10382_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10152_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10357_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20020_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10229_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10155_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10155_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10255_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10468_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20004_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10049_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10419_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20042_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10214_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10048_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10119_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10294_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10294_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10138_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10138_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10138_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10339_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10059_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10469_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10469_ses-00002_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10301_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10301_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10363_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10235_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10421_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10296_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10296_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10296_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10054_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20061_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20061_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20061_ses-00001_pro-CT-Portal__3',
             'bat-chic-1_sub-CH10181_ses-00001_pro-CT-Portal__1',
             'bat-chic-2_sub-CH20026_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10445_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10344_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10344_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20110_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20055_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10090_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10459_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10221_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10007_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10011_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10011_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10144_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10144_ses-00001_pro-CT-Portal__2',
             'bat-chic-2_sub-CH20128_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10361_ses-00002_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10430_ses-00001_pro-CT-Portal__1',
             'bat-chic-1_sub-CH10430_ses-00001_pro-CT-Portal__2',
             'bat-chic-1_sub-CH10145_ses-00001_pro-CT-Portal__1']

            df = _df[_df['center'] == 'chic'].reset_index(drop=True)
            df = df.dropna(subset=['npw', 'aphe', 'ec']).reset_index(drop=True)
            df = df[df['dp_lesion'].isin(vols)].reset_index(drop=True)


            print('df shape', df.shape)



        skf = StratifiedKFold(n_splits=config.n_fold)

        list_ven = [x for x in df['datapoint'] if x.endswith('__VEN.nii.gz') or x.startswith('bat')]
        df_ven = df[df['datapoint'].isin(list_ven)].reset_index(drop=True)
        order_ = df_ven['dp_lesion']

        ### FOR CALV TRAINING ONLY
        #list_art = [x.replace('VEN', 'ART') for x in order_]
        #idxs = [df[df['dp_lesion'] == x].index.item() for x in list_art]
        #df_art = df.loc[idxs, :].reset_index(drop=True)
        #### END CALV TRAINING ONLY

        batch_size_per_gpu = config.batch_size // idr_torch.size

        print('idr_torch size', idr_torch.size)
        print('idr_torch rank', idr_torch.rank)
        print('num gpus', int(os.environ['SLURM_GPUS_ON_NODE']))
        print('num cpus', idr_torch.cpus_per_task)

        from sklearn.model_selection import train_test_split


        for j, (train_index, val_index) in enumerate(skf.split(df_ven['patient_name'].unique(),
                                                               [np.max(
                                                                   np.array(df_ven[df_ven['patient_name'] == x][
                                                                    'has_hcc'])) for x in
                                                                df_ven['patient_name'].unique()])):

            #if j == 0:


            """_, _, _, _, train_index, val_index = train_test_split(df_ven['patient_name'].unique(),
                                                                  [np.max(
                                                                      np.array(df_ven[df_ven['patient_name'] == x][
                                                                                   'has_hcc'])) for x in
                                                                      df_ven['patient_name'].unique()],
                                                                  list(range(len(df_ven['patient_name'].unique()))),
                                                                  stratify=[np.max(
                                                                      np.array(df_ven[df_ven['patient_name'] == x][
                                                                                   'has_hcc'])) for x in
                                                                      df_ven['patient_name'].unique()],
                                                                  test_size=0.3,
                                                                  random_state=10)#10"""

            print('Fold '+str(j)+' starting...')

            trainnames = df_ven['patient_name'].unique()[train_index]
            train_index = df_ven[df_ven['patient_name'].isin(trainnames)].index

            valnames = df_ven['patient_name'].unique()[val_index]
            val_index = df_ven[df_ven['patient_name'].isin(valnames)].index

            for name in trainnames:
                assert(name not in valnames)
            for name in valnames:
                assert(name not in trainnames)

            train_index_ = df_ven.loc[train_index, 'datapoint']
            val_index_ = df_ven.loc[val_index, 'datapoint']

            train = [1 if x in list(train_index_) else 0 for x in df_ven['datapoint']]
            val = [1 if x in list(val_index_) else 0 for x in df_ven['datapoint']]

            df_ven['train_set'] = train
            df_ven['val_set'] = val

            #df_art['train_set'] = train ### FOR CALV TRAINING ONLY
            #df_art['val_set'] = val     ### FOR CALV TRAINING ONLY

            test = _df[_df['test_set'] == 1]
            test['aphe'] = random.choices([0, 1], k=test.shape[0])
            test['npw'] = random.choices([0, 1], k=test.shape[0])
            test['ec'] = random.choices([0, 1], k=test.shape[0])

            chicp = _df[_df['center'] == 'chicp']
            chicp['aphe'] = random.choices([0, 1], k=chicp.shape[0])
            chicp['npw'] = random.choices([0, 1], k=chicp.shape[0])
            chicp['ec'] = random.choices([0, 1], k=chicp.shape[0])
            chicp['train_set'] = 0
            chicp['val_set'] = 0
            chicp['test_set'] = 0

            df_ = pd.concat([df_ven,
                             #df_art ### ONLY FOR CALV TRAINING
                             test.reset_index(drop=True),
                             chicp.reset_index(drop=True)
                             ]
                            ).reset_index(drop=True)
            df_['multilabel'] = pd.Series(np.vstack((df_['aphe'].to_numpy(),
                                           df_['ec'].to_numpy(),
                                           df_['npw'].to_numpy(),
                                           df_['has_hcc'].to_numpy())).T.tolist())

            ## re enregistrer le dataset en cross val
            if os.path.isdir(join(config.lght_dir, config.dir)) == False:
                os.mkdir(join(config.lght_dir, config.dir))
            df_.to_csv(join(config.lght_dir, config.dir,
                            '1_subjects_label_hcc_chic_cv.csv'))
            df_.to_csv(join(config.lght_dir, config.dir,
                            '1_subjects_label_hcc_chic_cv_'+str(j)+'.csv'))


            dataset_train = DatasetCL(config, training=True, grenoble=config.grenoble, dimension=config.dimension)
            dataset_val = DatasetCL(config, validation=True, grenoble=config.grenoble, dimension=config.dimension)
            dataset_test = DatasetCL(config, test=True, grenoble=config.grenoble, dimension=config.dimension)
            #dataset_chicp = DatasetCL(config, chicp=True, grenoble=config.grenoble, dimension=config.dimension)


            sampler = WeightedRandomSampler(dataset_train.samples_weight.type('torch.DoubleTensor'),
                                            len(dataset_train))

            print('Weights assigned to each class', dataset_train.weight)

            loader_train = DataLoader(dataset_train,
                                      batch_size=batch_size_per_gpu,
                                      sampler=DistributedSamplerWrapper(sampler,
                                                                        num_replicas=idr_torch.size,
                                                                        rank=idr_torch.rank,
                                                                        shuffle=True),
                                      #sampler=sampler,
                                      collate_fn=dataset_train.collate_fn,
                                      pin_memory=config.pin_mem,
                                      num_workers=8,
                                      drop_last=False)

            loader_val = DataLoader(dataset_val,
                                    batch_size=batch_size_per_gpu,
                                    sampler=DistributedSampler(dataset_val,
                                                               num_replicas=idr_torch.size,
                                                               rank=idr_torch.rank,
                                                               shuffle=False),
                                    #sampler=SequentialSampler(dataset_val),
                                    collate_fn=dataset_val.collate_fn,
                                    pin_memory=config.pin_mem,
                                    num_workers=8,
                                    drop_last=False)

            print('Ready to download the model!')
            print('Pretrained on ImageNet?', config.pretrained)
            net = model_.network(mode="classifier",
                                 net=config.encoder,
                                 pretrained=config.pretrained,
                                 n_layer=config.n_layer,
                                 num_classes=config.num_classes,
                                 rep_dim=config.rep_dim,
                                 hidden_dim=config.hidden_dim,
                                 input_dim=input_dim)
            print('Network downloaded!')

            if config.mode == "pretraining":
                loss = SupConLoss(config=config, temperature=config.temperature, return_logits=True)
            elif config.mode == 'finetuning':
                if config.label_name == 'multilabel':   # if multilabel problem, BCE on vector of shape C
                    #loss = WeightedMultilabelLoss(weights=torch.tensor([1/6, 1/6, 1/6, 1/2]).to(device='cuda'))
                    loss = torch.nn.BCEWithLogitsLoss()
                else:
                    loss = CrossEntropyLoss()           # if single label problem, cross entropy on output of shape 2 bc usually works better than bce

            """
            lr_logger = LearningRateMonitor(logging_interval='epoch')
            # Folder hack
            tb_logger = TensorBoardLogger(save_dir=config.lght_dir,
                                          name=config.dir,
                                          version=config.dir)"""

            checkpoint_callback = ModelCheckpoint(#monitor="vl_epoch",
                                                  #save_top_k=1,
                                                  #every_n_epochs=1,
                                                  save_last=True,
                                                  #mode="min",
                                                  dirpath=join(config.lght_dir,
                                                               config.dir,
                                                               config.dir),
                                                  #filename='best'
                                                  )

            trainer = pl.Trainer(default_root_dir=config.lght_dir,
                                 max_epochs=config.max_epochs,
                                 callbacks=[checkpoint_callback],
                                 val_check_interval=config.val_rate,
                                 # amp_backend="native",
                                 # precision=16,
                                 reload_dataloaders_every_n_epochs=0,
                                 num_sanity_val_steps=-1,
                                 num_nodes=int(os.environ['SLURM_NNODES']),
                                 devices=int(os.environ['SLURM_GPUS_ON_NODE']),
                                 strategy='ddp_find_unused_parameters_true',
                                 # profiler="simple",
                                 accelerator="gpu",
                                 # replace_sampler_ddp=False
                                 )

            model = yAwareCLModel(net, loss, config,
                                  dataset_train,
                                  dataset_val,
                                  dataset_test,
                                  dataset_test,
                                  config.mode)

            print(model)

            if config.pretrained:

                for param in model.net.layer1.parameters():
                    param.requires_grad = False
                for param in model.net.layer2.parameters():
                    param.requires_grad = False
                for param in model.net.layer3.parameters():
                    param.requires_grad = False
                #for param in model.net.layer4[0].parameters():
                #    param.requires_grad = False


            # we check the number of trainable parameters
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            model_params = sum([np.prod(p.size()) for p in model_parameters])
            print('Number of trainable parameters in the whole model: ' + str(model_params))
            config.cv_fold = j
            trainer.fit(model, loader_train, loader_val)


            print('Saving outputs...')
            np.save(join(config.lght_dir, config.dir, 'metrics'+str(j)+'.npy'), model.metrics,
                    allow_pickle=True)
            ####################################
            """
            np.save(join(config.lght_dir, config.dir, '_training_inputs'+str(j)+'.npy'),
                    np.concatenate(np.asanyarray(model.training_inputs, dtype="object")),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_validation_inputs'+str(j)+'.npy'),
                    np.concatenate(np.asanyarray(model.validation_inputs, dtype="object")),
                    allow_pickle=True)
            if j == 0:
                np.save(join(config.lght_dir, config.dir, '_test_inputs' + str(j) + '.npy'),
                        np.concatenate(np.asanyarray(model.test_inputs, dtype="object")),
                        allow_pickle=True)
                np.save(join(config.lght_dir, config.dir, '_chicp_inputs' + str(j) + '.npy'),
                        np.concatenate(np.asanyarray(model.chicp_inputs, dtype="object")),
                        allow_pickle=True)
            """
            ####################################
            np.save(join(config.lght_dir, config.dir, '_training_rep'+str(j)+'.npy'),
                    np.concatenate(np.asanyarray(model.training_representations, dtype="object")),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_validation_rep'+str(j)+'.npy'),
                    np.concatenate(np.asanyarray(model.validation_representations, dtype="object")),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_test_rep' + str(j) + '.npy'),
                    np.concatenate(np.asanyarray(model.test_representations, dtype="object")),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_chicp_rep' + str(j) + '.npy'),
                    np.concatenate(np.asanyarray(model.chicp_representations, dtype="object")),
                    allow_pickle=True)
            ####################################
            np.save(join(config.lght_dir, config.dir, '_training_volumes'+str(j)+'.npy'),
                    np.array(model.training_volumes),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_validation_volumes'+str(j)+'.npy'),
                    np.array(model.validation_volumes),
                    allow_pickle=True)

            np.save(join(config.lght_dir, config.dir, '_test_volumes' + str(j) + '.npy'),
                    np.array(model.test_volumes),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_chicp_volumes' + str(j) + '.npy'),
                    np.array(model.chicp_volumes),
                    allow_pickle=True)
            ####################################
            np.save(join(config.lght_dir, config.dir, '_training_labels'+str(j)+'.npy'),
                    np.array(model.training_labels),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_validation_labels'+str(j)+'.npy'),
                    np.array(model.validation_labels),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_test_labels' + str(j) + '.npy'),
                    np.array(model.test_labels),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_chicp_labels' + str(j) + '.npy'),
                    np.array(model.chicp_labels),
                    allow_pickle=True)
            ####################################
            if config.mode == 'finetuning':
                np.save(join(config.lght_dir, config.dir, '_training_probas'+str(j)+'.npy'),
                        np.concatenate(np.asanyarray(model.training_probas, dtype="object")),
                        allow_pickle=True)
                np.save(join(config.lght_dir, config.dir, '_validation_probas'+str(j)+'.npy'),
                        np.concatenate(np.asanyarray(model.validation_probas, dtype="object")),
                        allow_pickle=True)
                np.save(join(config.lght_dir, config.dir, '_test_probas' + str(j) + '.npy'),
                        np.concatenate(np.asanyarray(model.test_probas, dtype="object")),
                        allow_pickle=True)
                np.save(join(config.lght_dir, config.dir, '_chicp_probas' + str(j) + '.npy'),
                        np.concatenate(np.asanyarray(model.chicp_probas, dtype="object")),
                        allow_pickle=True)
                ####################################
                np.save(join(config.lght_dir, config.dir, '_training_preds'+str(j)+'.npy'),
                        np.concatenate(np.asanyarray(model.training_preds, dtype="object")),
                        allow_pickle=True)
                np.save(join(config.lght_dir, config.dir, '_validation_preds'+str(j)+'.npy'),
                        np.concatenate(np.asanyarray(model.validation_preds, dtype="object")),
                        allow_pickle=True)
                np.save(join(config.lght_dir, config.dir, '_test_preds' + str(j) + '.npy'),
                        np.concatenate(np.asanyarray(model.test_preds, dtype="object")),
                        allow_pickle=True)
                np.save(join(config.lght_dir, config.dir, '_chicp_preds' + str(j) + '.npy'),
                        np.concatenate(np.asanyarray(model.chicp_preds, dtype="object")),
                        allow_pickle=True)
            print('Outputs saved...')

    else:


        batch_size_per_gpu = config.batch_size // idr_torch.size

        print('idr_torch size',idr_torch.size)
        print('idr_torch rank', idr_torch.rank)
        print('num gpus', int(os.environ['SLURM_GPUS_ON_NODE']))
        print('num cpus', idr_torch.cpus_per_task)

        dataset_train = DatasetCL(config, training=True, grenoble=config.grenoble, dimension=config.dimension)
        dataset_val = DatasetCL(config, validation=True, grenoble=config.grenoble, dimension=config.dimension)
        dataset_test = DatasetCL(config, test=True, grenoble=config.grenoble, dimension=config.dimension)

        """
        indices = {}  # indices of each volume (patient) in the dataset
        for z in dataset_train.volumes:
            indices[z] = [i for i, x in enumerate(dataset_train.volumes) if x == z]

        sampler = CustomSampler(dataset_train, config.batch_size, indices,
                                weights=dataset_train.samples_weight.numpy(),
                                k=10000)"""

        #sampler = CustomWeightedRandomSampler(dataset=dataset_train,
        #                                      num_samples=len(dataset_train))

        sampler = WeightedRandomSampler(dataset_train.samples_weight.type('torch.DoubleTensor'),
                                        len(dataset_train))

        print('Weights assigned to each class', dataset_train.weight)

        loader_train = DataLoader(dataset_train,
                                  batch_size=batch_size_per_gpu,
                                  sampler=DistributedSamplerWrapper(sampler,
                                                                    num_replicas=idr_torch.size,
                                                                    rank=idr_torch.rank,
                                                                    shuffle=True),
                                  # sampler=sampler,
                                  collate_fn=dataset_train.collate_fn,
                                  pin_memory=config.pin_mem,
                                  num_workers=8,
                                  drop_last=False)

        loader_val = DataLoader(dataset_val,
                                batch_size=batch_size_per_gpu,
                                sampler=DistributedSampler(dataset_val,
                                                           num_replicas=idr_torch.size,
                                                           rank=idr_torch.rank,
                                                           shuffle=False),
                                # sampler=SequentialSampler(dataset_val),
                                collate_fn=dataset_val.collate_fn,
                                pin_memory=config.pin_mem,
                                num_workers=8,
                                drop_last=False)


        if config.mode == "pretraining":

            net = model_.network(mode="encoder",
                                 net=config.encoder,
                                 pretrained=config.pretrained,
                                 n_layer=config.n_layer,
                                 rep_dim=config.rep_dim,
                                 hidden_dim=config.hidden_dim,
                                 output_dim=config.output_dim,
                                 dimension=config.dimension,
                                 segmask=config.segmask,
                                 input_dim=config.input_dim)

            if config.model_type == "supervised":
                loss = SupConLoss(config=config, temperature=config.temperature, return_logits=True)
            elif config.model_type == "unsupervised":
                loss = NTXenLoss(temperature=config.temperature, return_logits=True)
            elif config.model_type == "supsim":
                loss_supcon = SupConLoss(config=config, temperature=config.temperature, return_logits=True)
                loss_simclr = NTXenLoss(temperature=config.temperature, return_logits=True)
                loss = SupSimLoss(loss_supcon, loss_simclr)
            elif config.model_type == "ysimclr":
                loss_supervised = CrossEntropyLoss()
                loss_simclr = NTXenLoss(temperature=config.temperature, return_logits=True)
                loss = ySimLoss(loss_supervised, loss_simclr)
            elif config.model_type == "z-supervised":
                loss = GeneralizedSupervisedNTXenLoss(config=config,
                                                      temperature=config.temperature,
                                                      kernel=config.kernel,
                                                      sigma=config.sigma,
                                                      return_logits=True)
            elif config.model_type == 'dino':
                loss = DINOLoss(config)

        elif config.mode == "finetuning":

            net = model_.network(mode="classifier",
                                 net=config.encoder,
                                 pretrained=config.pretrained,
                                 n_layer=config.n_layer,
                                 rep_dim=config.rep_dim,
                                 hidden_dim=config.hidden_dim,
                                 output_dim=config.output_dim,
                                 dimension=config.dimension,
                                 segmask=config.segmask,
                                 input_dim=config.input_dim)

            loss = CrossEntropyLoss()

        lr_logger = LearningRateMonitor(logging_interval='epoch')

        # Folder hack
        tb_logger = TensorBoardLogger(save_dir=config.lght_dir,
                                      name=config.dir,
                                      version=config.dir)

        checkpoint_callback = ModelCheckpoint(monitor="vl_epoch",
                                              save_top_k=1,
                                              every_n_epochs=1,
                                              save_last=True,
                                              mode="min",
                                              dirpath=tb_logger.log_dir,
                                              filename='best')

        trainer = pl.Trainer(default_root_dir=config.lght_dir,
                             max_epochs=config.max_epochs,
                             callbacks=[checkpoint_callback, lr_logger],
                             val_check_interval=config.val_rate,
                             #amp_backend="native",
                             #precision=16,
                             reload_dataloaders_every_n_epochs=0,
                             num_sanity_val_steps=-1,
                             logger=tb_logger,
                             num_nodes=int(os.environ['SLURM_NNODES']),
                             devices=int(os.environ['SLURM_GPUS_ON_NODE']),
                             strategy='ddp_find_unused_parameters_true',
                             #profiler="simple",
                             accelerator="gpu",
                             #replace_sampler_ddp=False
                             )
                # The monitor argument name corresponds to the scalar value that you log
                # when using the self.log method within the LightningModule hooks.

        model = yAwareCLModel(net, loss, config, dataset_train, dataset_val, dataset_test, config.mode)
        if config.model_type == 'dino':
            model = DINOModel(net, loss, config, dataset_train, dataset_val, config.mode)

        # we check the number of trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_params = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of trainable parameters in the whole model: ' + str(model_params))

        start = time.time()
        trainer.fit(model, loader_train, loader_val)
        trainer.test(model, loader_test)

        print('Training time', time.time() - start)

        print('Saving outputs...')
        np.save(join(config.lght_dir, config.dir, 'metrics.npy'), model.metrics,
                allow_pickle=True)
        ####################################
        np.save(join(config.lght_dir, config.dir, '_training_inputs.npy'),
                np.concatenate(np.asanyarray(model.training_inputs, dtype="object")),
                allow_pickle=True)
        np.save(join(config.lght_dir, config.dir, '_validation_inputs.npy'),
                np.concatenate(np.asanyarray(model.validation_inputs, dtype="object")),
                allow_pickle=True)
        np.save(join(config.lght_dir, config.dir, '_test_inputs.npy'),
                np.concatenate(np.asanyarray(model.test_inputs, dtype="object")),
                allow_pickle=True)
        ####################################
        np.save(join(config.lght_dir, config.dir, '_training_rep.npy'),
                np.concatenate(np.asanyarray(model.training_representations, dtype="object")),
                allow_pickle=True)
        np.save(join(config.lght_dir, config.dir, '_validation_rep.npy'),
                np.concatenate(np.asanyarray(model.validation_representations, dtype="object")),
                allow_pickle=True)
        np.save(join(config.lght_dir, config.dir, '_test_rep.npy'),
                np.concatenate(np.asanyarray(model.test_representations, dtype="object")),
                allow_pickle=True)
        ####################################
        np.save(join(config.lght_dir, config.dir, '_training_volumes.npy'),
                np.array(model.training_volumes),
                allow_pickle=True)
        np.save(join(config.lght_dir, config.dir, '_validation_volumes.npy'),
                np.array(model.validation_volumes),
                allow_pickle=True)
        np.save(join(config.lght_dir, config.dir, '_test_volumes.npy'),
                np.array(model.test_volumes),
                allow_pickle=True)
        ####################################
        np.save(join(config.lght_dir, config.dir, '_training_labels.npy'),
                np.array(model.training_labels),
                allow_pickle=True)
        np.save(join(config.lght_dir, config.dir, '_validation_labels.npy'),
                np.array(model.validation_labels),
                allow_pickle=True)
        np.save(join(config.lght_dir, config.dir, '_test_labels.npy'),
                np.array(model.test_labels),
                allow_pickle=True)
        ####################################
        if config.mode == 'finetuning':
            np.save(join(config.lght_dir, config.dir, '_training_probas.npy'),
                    np.concatenate(np.asanyarray(model.training_probas, dtype="object")),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_validation_probas.npy'),
                    np.concatenate(np.asanyarray(model.validation_probas, dtype="object")),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_test_probas.npy'),
                    np.concatenate(np.asanyarray(model.test_probas, dtype="object")),
                    allow_pickle=True)
            ####################################
            np.save(join(config.lght_dir, config.dir, '_training_preds.npy'),
                    np.concatenate(np.asanyarray(model.training_preds, dtype="object")),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_validation_preds.npy'),
                    np.concatenate(np.asanyarray(model.validation_preds, dtype="object")),
                    allow_pickle=True)
            np.save(join(config.lght_dir, config.dir, '_test_preds.npy'),
                    np.concatenate(np.asanyarray(model.test_preds, dtype="object")),
                    allow_pickle=True)
        print('Outputs saved...')

        ### CALL inference
        #print('Inference...')
        #path = join(config.lght_dir, config.dir, config.dir)
        #inference(config,path,'best',dataset_train, dataset_full_val,dataset_test)
        #print('Inference done!')