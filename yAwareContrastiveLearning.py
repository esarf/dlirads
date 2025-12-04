import os

import sklearn.metrics
import torch
from torch.optim.lr_scheduler import ExponentialLR
from collections import OrderedDict
# from sam import SAM
from models.gradcam import grad_cam
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.nn.functional import softmax, sigmoid
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, Subset
from dataset import Dataset
from sklearn.decomposition import PCA
import time
import cv2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from ignite.contrib.metrics import ROC_AUC
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, balanced_accuracy_score, roc_curve, auc
import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression
from dataset import DatasetCL
import seaborn as sns
import pandas as pd
import itertools
from os.path import join
from os import listdir
from collections import Counter
from torch import Tensor
import models.network as model
import random
from sklearn import svm
from torch.cuda.amp import autocast
import kornia
from kornia import image_to_tensor, tensor_to_image
from augmentations import *
from torchvision.transforms import v2
from misalignment import augment_misalign
import kornia.augmentation as K
import scipy



class DataAugmentation(torch.nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, dimension=2) -> None:
        super().__init__()

        if dimension == 2:
            self.transforms = torch.nn.Sequential(
                #kornia.augmentation.RandomGaussianNoise(p=0.5, mean=0.0, std=0.1),          # to comment
                #kornia.augmentation.RandomGaussianBlur(p=0.5, kernel_size=(3, 3), sigma=(0.1, 2)),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                kornia.augmentation.RandomVerticalFlip(p=0.5),
                #kornia.augmentation.RandomErasing(p=0.5, scale=(0.05, 0.05), ratio=(1, 1)),   # to comment
                #kornia.augmentation.RandomResizedCrop(p=0.5, size=(512, 512), scale=(0.7, 0.7)),
                #kornia.augmentation.RandomAffine(p=0.5, degrees=0, translate=(0.2, 0.2)),
                kornia.augmentation.RandomRotation(p=0.5, degrees=30)
            )

        else:
            self.transforms = torch.nn.Sequential(
                # kornia.augmentation.RandomGaussianNoise(p=0.5, mean=0.0, std=0.1),          # to comment
                # kornia.augmentation.RandomGaussianBlur(p=0.5, kernel_size=(3, 3), sigma=(0.1, 2)),
                kornia.augmentation.RandomHorizontalFlip3D(p=0.5),
                kornia.augmentation.RandomVerticalFlip3D(p=0.5),
                # kornia.augmentation.RandomErasing(p=0.5, scale=(0.05, 0.05), ratio=(1, 1)),   # to comment
                #kornia.augmentation.RandomCrop3D(p=0.5, size=(16, 512, 512), pad_if_needed=True),
                kornia.augmentation.RandomAffine3D(p=0.5, degrees=0, translate=(0.2, 0.2, 0)),
                kornia.augmentation.RandomRotation3D(p=0.5, degrees=30)
            )
    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        return x_out


def augmentations(img, config):

    with torch.no_grad():

        grids = torch.stack(torch.meshgrid(torch.arange(-1, 2, 2),
                                           torch.arange(-1, 2, 2), indexing='xy')).repeat(config.batch_size, 1, 1, 1)
        grids = grids.type_as(img)

        # image
        img_ = RandomGaussianNoise_(p=0.5)(img, config)
        #img_ = RandomGaussianBlur_()(img_, config)
        img_ = RandomHorizontalFlip_(p=0.5)(img_, config)
        img_ = RandomVerticalFlip_(p=0.5)(img_, config)
        #img_ = RandomCutout_(p=0.5)(img_, config)

        # spatial
        grids_ = RandomResizedCrop_(p=0.5)(grids, config)
        grids_ = RandomRotate_(p=0.5)(grids_, config)
        grids_ = RandomTranslate_(p=0.5)(grids_, config)

        # interpolate
        grids_ = F.interpolate(grids_, size=512, mode='bilinear', align_corners=True)
        img_ = torch.nn.functional.grid_sample(img_, grids_.permute(0, 2, 3, 1), align_corners=True)

        return img_

def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]


def recenter_patch(img, c=15):
    """Function to recenter the patch
    The patch was augmented in x,y,z for data augmentation by +-30 voxels
    so we recenter around the lesion to obtain the right final shape"""
    return img[:,:,:,c:img.shape[3]-c,c:img.shape[4]-c]
    #return img


class AugmentMisalign(object):

    def __init__(self,proba_squeeze=0.5,proba_rota=0.5,proba_trans=0.5):

        self.proba_squeeze = proba_squeeze
        self.proba_rota = proba_rota
        self.proba_trans = proba_trans

    def __call__(self, img, seg):

        return augment_misalign(img, seg, data_size=(32,126,126),
            im_channels_2_misalign=[1, ], label_channels_2_misalign=[1, ],
            do_squeeze=True,
            sq_x=[1.0, 1.0], sq_y=[0.9, 1.1], sq_z=[1.0, 1.0],
            p_sq_per_sample=self.proba_squeeze,
            do_rotation=True,
            angle_x=(- 0 / 360. * 2 * np.pi, 0 / 360. * 2 * np.pi),
            angle_y=(- 0 / 360. * 2 * np.pi, 0 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            p_rot_per_sample=self.proba_rota,
            do_transl=True,
            tr_x=[-32, 32], tr_y=[-32, 32], tr_z=[-2, 2],
            p_transl_per_sample=self.proba_trans,
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_data=3, order_seg=0)




class yAwareCLModel(pl.LightningModule):

    def __init__(self, net, loss, config, data_train, data_val, data_test, data_chicp, scheduler=None):
        """
        Parameters
        ----------
        net: subclass of nn.Module
        loss: callable fn with args (y_pred, y_true)
        loader_train, loader_val: pytorch DataLoaders for training/validation
        config: Config object with hyperparameters
        scheduler (optional)
        """
        super().__init__()
        self.loss = loss
        self.net = net
        self.config = config
        self.data_train = data_train
        self.data_val = data_val
        self.mode = config.mode
        self.data_test = data_test
        self.data_chicp = data_chicp
        self.imgs_epoch = torch.zeros(len(self.data_val), self.config.rep_dim)

        # data augmentations
        #self.transforms = v2.Compose([
        #    v2.RandomHorizontalFlip(p=0.6),
        #    v2.RandomVerticalFlip(p=0.6),
        #    v2.RandomAffine(degrees=15),
        #    v2.RandomRotation(degrees=15),
        #    v2.RandomPerspective(0.3,p=0.6),
        #    #v2.GaussianBlur(kernel_size=(5,5),sigma=(0.1,0.5))
        #])
        self.transforms = torch.nn.Sequential(
            K.RandomVerticalFlip3D(p=0.1),
            K.RandomHorizontalFlip3D(p=0.1),
            K.RandomRotation3D(degrees=20, p=0.1),
            K.RandomAffine3D(degrees=0, p=0.1),
        )
        self.misalignment = AugmentMisalign()

        self.val_metric_model = self.config.val_metric_model
        self.nb_steps = len(data_val) // self.config.batch_size + 1
        self.val_loss_step = 0
        self.test_loss_step = 0
        self.cutoff = 0.5
        self.vols_train = []
        self.y_pred_train = []
        self.y_true_train = []
        self.y_pred_val = []
        self.y_true_val = []
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_representations = []
        self.training_inputs = []
        self.training_preds = []
        self.validation_representations = []
        self.validation_inputs = []
        self.validation_preds = []
        self.test_representations = []
        self.training_probas = []
        self.validation_probas = []
        self.test_probas = []
        self.test_preds = []
        self.test_inputs = []

        self.chicp_probas = []
        self.chicp_preds = []
        self.chicp_inputs = []
        self.chicp_representations = []

        self.metrics = {'train_loss':[],
                        'train_auc':[],
                        'val_loss':[],
                        'val_auc':[],
                        'test_loss':[],
                        'test_auc':[]}

        assert self.mode in ['finetuning', 'pretraining', 'test'], ('self.mode =', self.mode)

        if self.config.mode == 'finetuning' or self.config.model_type == "supervised" or self.config.model_type == "supsim" or self.config.model_type == "ysimclr":

            # TRAINING PART
            index_train = self.data_train.volumes                                     # liste des volumes dans toute la base d'entraînement
            self.df_train = pd.DataFrame({'epoch_'+str(i): [np.zeros(1) for x in index_train] for i in range(1000)},
                                         index=index_train)
            # VALIDATION PART
            index_val = self.data_val.volumes
            self.df_val = pd.DataFrame({'epoch_'+str(i): [np.zeros(1) for x in index_val] for i in range(1000)},
                                       index=index_val)
            # TEST PART
            index_test = self.data_test.volumes
            self.df_test = pd.DataFrame(
                {'epoch_' + str(i): [np.zeros(1) for x in index_test] for i in range(1000)},
                index=index_test)

            if self.config.label_name == 'multilabel':
                self.df_train['label'] = list(self.data_train.has_hcc[:, -1])
                self.df_val['label'] = list(self.data_val.has_hcc[:, -1])
                self.df_test['label'] = list(self.data_test.has_hcc[:, -1])
            else:
                self.df_train['label'] = list(self.data_train.has_hcc)
                self.df_val['label'] = list(self.data_val.has_hcc)
                self.df_test['label'] = list(self.data_test.has_hcc)


        if hasattr(config, 'pretrained_path') and config.pretrained_path is not None:
            self.load_model(config.pretrained_path)


    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # discard the version number
        items.pop("v_num", None)
        return items

    def forward(self, x, s=None, mode=None):
        y = self.net(x, s, mode)
        return y

    def training_step(self, batch, batch_idx):

        ## add center input

        inputs, labels, subjects_id, z, center = batch          # batch : N x (images augmentées (1 et 2), label (age), subject_id)
                                                        # inputs shape      (batch_size,1 ou 3,1,512,512)
                                                        # labels shape      (batch_size,1)
                                                        # subjects_id shape (batch_size,1)


        assert self.mode in ['finetuning', 'pretraining'], ('self.mode =', self.mode)

        if self.mode == "pretraining":

            if self.config.augment == 'misalign':

                aug_i, aug_j = recenter_patch(self.misalignment((inputs / 250.)[:,:2,:].cpu().numpy(),
                                                                torch.stack(((inputs / 250.)[:,2,:],(inputs / 250.)[:,2,:]),dim=1).cpu().numpy())), \
                               recenter_patch(self.misalignment((inputs / 250.)[:, :2, :].cpu().numpy(),
                                                                torch.stack(((inputs / 250.)[:,2,:],(inputs / 250.)[:,2,:]),dim=1).cpu().numpy()))

                ## Forward pass
                z_i = self(torch.tensor(aug_i).cuda().type(torch.cuda.FloatTensor))  # z_i : première image augmentée passée dans simCLR
                z_j = self(torch.tensor(aug_j).cuda().type(torch.cuda.FloatTensor))  # z_j : deuxième image augmentée passée dans simCLR

            else:

                aug_i, aug_j = recenter_patch(self.transforms(inputs / 250.)), \
                               recenter_patch(self.transforms(inputs / 250.))

                ## Forward pass
                z_i = self(aug_i)  # z_i : première image augmentée passée dans simCLR
                z_j = self(aug_j)  # z_j : deuxième image augmentée passée dans simCLR


            # aug_i : shape (batch_size, 1, 512, 512)
            # aug_j : shape (batch_size, 1, 512, 512)
            # z_i : shape (batch_size,d=128)
            # z_j : shape (batch_size,d=128)

            ## Compute the Loss
            if self.config.model_type == "supervised" or self.config.model_type == "supsim":
                loss, logits, target = self.loss(z_i, z_j, labels)    # loss : SupCON loss if labels or NTXent without labels
            elif self.config.model_type == "unsupervised":
                loss, logits, target = self.loss(z_i, z_j)
            elif self.config.model_type == "z-supervised":
                loss, logits, target = self.loss(z_i, z_j, labels, z)
            elif self.config.model_type == "ysimclr":
                aug = self.transforms(inputs / 250.)  # inputs shape: (batch_size,1,512,512)
                y = self(aug, mode="classifier")  # y shape (batch_size,C)
                loss = self.loss(z_i, z_j, y, labels)  # loss : cross-entropy loss


        elif self.mode == "finetuning":

            if self.config.augment == 'misalign':

                aug = recenter_patch(self.misalignment((inputs / 250.)[:, :2, :].cpu().numpy(),
                                     torch.stack(((inputs / 250.)[:, 2, :], (inputs / 250.)[:, 2, :]), dim=1).cpu().numpy()))

                ## Forward pass
                y = self(torch.tensor(aug).cuda().type(torch.cuda.FloatTensor))  # y shape (batch_size,C)

            else:
                aug = recenter_patch(self.transforms(inputs / 250.))

            ## Forward pass
            y, y_hcc = self(aug, z)  # y shape (batch_size,C)

            ## Output probabilities of class 1 on the batch
            if self.config.label_name == 'multilabel':
                labels_hcc = labels[:, -1]
                ## Compute the Loss
                # add center mask
                loss = self.loss(y
                                 ,
                                 labels
                                 )
                y_prob = sigmoid(y_hcc).detach().cpu().numpy()
                ### Fill the epoch-level metrics dataframe : we compute the average probability over the whole epoch
                ep = 'epoch_' + str(self.current_epoch)
                for i, subject in enumerate(subjects_id):
                    self.df_train.loc[subject, ep] = np.nan_to_num(y_prob[i])
                self.y_pred_train.append(y_prob)
                self.y_true_train.append(labels_hcc.cpu())


            else:
                ## Compute the Loss
                loss = self.loss(y
                                 ,
                                 labels
                                 )
                y_prob = softmax(y, dim=1).detach().cpu().numpy()
                ### Fill the epoch-level metrics dataframe : we compute the average probability over the whole epoch
                ep = 'epoch_' + str(self.current_epoch)
                for i, subject in enumerate(subjects_id):
                    self.df_train.loc[subject, ep] = np.nan_to_num(y_prob[i, 1])
                self.y_pred_train.append(y_prob[:,1])
                self.y_true_train.append(labels.cpu())

        # accumulate the sampled volumes through the epoch
        self.vols_train.append(subjects_id)
        self.training_step_outputs.append(loss)

        return loss


    def on_train_epoch_end(self):

        print('')
        print('---Training metrics---')

        if self.mode == 'finetuning':

            if int(os.environ['SLURM_GPUS_ON_NODE']) > 1:

                all_loss = self.all_gather(torch.as_tensor([x.item() for x in self.training_step_outputs]))
                all_preds = self.all_gather(torch.as_tensor(np.concatenate(self.y_pred_train)))
                all_true = self.all_gather(torch.as_tensor(np.concatenate(self.y_true_train)))
                y_true = torch.flatten(all_true).cpu().numpy()
                y_pred = torch.flatten(all_preds).detach().cpu().numpy()

                train_loss_epoch = np.mean(torch.flatten(all_loss).detach().cpu().numpy())


            else:

                train_loss_epoch = np.mean([x.item() for x in self.training_step_outputs])

                self.metrics['train_loss'].append(train_loss_epoch)

                y_true = np.concatenate(self.y_true_train)
                y_pred = np.concatenate(self.y_pred_train)

            fpr_train, tpr_train, thresholds = roc_curve(y_true, y_pred)
            self.cutoff = cutoff_youdens_j(fpr_train, tpr_train, thresholds)

            ## Balanced Accuracy per subject
            y_pred_youden = [1 if x > self.cutoff else 0 for x in y_pred]
            ## Confusion matrix per subject
            cm = confusion_matrix(y_true, y_pred_youden)
            print('train confusion matrix', cm)
            print('loss train epoch', train_loss_epoch)

            ### ROC-AUC score per subject for HCC or radio criteria, be it in single or multi-label
            train_auc = roc_auc_score(y_true, y_pred)
            self.metrics['train_auc'].append(train_auc)
            self.metrics['train_loss'].append(train_loss_epoch)
            ### END METRICS ###

            self.vols_train.clear()
            self.y_pred_train.clear()
            self.y_true_train.clear()

            print('train_auc', train_auc)
            print('---END training metrics---')
            print('')
            ### END LOGS ###

        self.training_step_outputs.clear()


    def validation_step(self, batch, batch_idx):

        #add center

        inputs, labels, subjects_id, z, center = batch          # batch : N x (images augmentée (1 et 2), label (age), subject_id)
                                                        # inputs shape      (batch_size, 1 ou 3, 1, 512,512)
                                                        # labels shape      (batch_size)
                                                        # subjects_id       (batch_size)

        assert self.mode in ['finetuning', 'pretraining'], ('self.mode =', self.mode)

        if self.mode == "pretraining":

            if self.config.augment == 'misalign':

                aug_i, aug_j = recenter_patch(self.misalignment((inputs / 250.)[:, :2, :].cpu().numpy(),
                                                                torch.stack(((inputs / 250.)[:, 2, :],
                                                                             (inputs / 250.)[:, 2, :]),
                                                                            dim=1).cpu().numpy())), \
                               recenter_patch(self.misalignment((inputs / 250.)[:, :2, :].cpu().numpy(),
                                                                torch.stack(((inputs / 250.)[:, 2, :],
                                                                             (inputs / 250.)[:, 2, :]),
                                                                            dim=1).cpu().numpy()))

                ## Forward pass
                z_i = self(torch.tensor(aug_i).cuda().type(
                    torch.cuda.FloatTensor))  # z_i : première image augmentée passée dans simCLR
                z_j = self(torch.tensor(aug_j).cuda().type(
                    torch.cuda.FloatTensor))  # z_j : deuxième image augmentée passée dans simCLR

            else:

                aug_i, aug_j = recenter_patch(self.transforms(inputs / 250.)), \
                               recenter_patch(self.transforms(inputs / 250.))

                ## Forward pass
                z_i = self(aug_i)  # z_i : première image augmentée passée dans simCLR
                z_j = self(aug_j)  # z_j : deuxième image augmentée passée dans simCLR        # aug_i : shape (batch_size, 1, 512, 512)
                # aug_j : shape (batch_size, 1, 512, 512)
                # z_i : shape (batch_size,d=128)
                # z_j : shape (batch_size,d=128)

            ## Compute the Loss
            if self.config.model_type == "supervised" or self.config.model_type == "supsim":
                val_loss_step, logits, target = self.loss(z_i, z_j, labels)    # loss : SupCON loss if labels or NTXent without labels
            elif self.config.model_type == "unsupervised":
                val_loss_step, logits, target = self.loss(z_i, z_j)
            elif self.config.model_type == "z-supervised":
                val_loss_step, logits, target = self.loss(z_i, z_j, labels, z)
            elif self.config.model_type == "ysimclr":
                aug = self.transforms(inputs / 250.)  # inputs shape: (batch_size,1,512,512)
                y = self(aug, mode="classifier")  # y shape (batch_size,C)
                val_loss_step = self.loss(z_i, z_j, y, labels)  # loss : cross-entropy loss

            self.val_loss_step += val_loss_step / self.nb_steps          # for the log of the avg
            # no separation with the sanity check because we run the forward on the whole validation set



        elif self.mode == "finetuning":

            ## Forward pass
            y, y_hcc = self(recenter_patch(inputs / 250.), z)  # y shape (batch_size,2)

            ## Output probabilities of class 1 on the batch
            if self.config.label_name == 'multilabel':
                labels_hcc = labels[:, -1]
                ## Compute the Loss
                val_loss_step = self.loss(y
                                          , labels
                                          )    # add mask
                y_prob = sigmoid(y_hcc).detach().cpu().numpy()
                ### Fill the epoch-level metrics dataframe : we compute the average probability over the whole epoch
                ep = 'epoch_' + str(self.current_epoch)
                for i, subject in enumerate(subjects_id):
                    self.df_val.loc[subject, ep] = np.nan_to_num(y_prob[i])
                self.y_pred_val.append(y_prob)
                self.y_true_val.append(labels_hcc.cpu())
            else:
                ## Compute the Loss
                val_loss_step = self.loss(y
                                 ,
                                 labels
                                 )

                y_prob = softmax(y, dim=1).detach().cpu().numpy()
                ### Fill the epoch-level metrics dataframe : we compute the average probability over the whole epoch
                ep = 'epoch_' + str(self.current_epoch)
                for i, subject in enumerate(subjects_id):
                    self.df_val.loc[subject, ep] = np.nan_to_num(y_prob[i, 1])
                self.y_pred_val.append(y_prob[:, 1])
                self.y_true_val.append(labels.cpu())

            self.val_loss_step += val_loss_step / self.nb_steps  # for the log of the avg

        self.validation_step_outputs.append(val_loss_step)

        return val_loss_step

    def on_validation_epoch_end(self):

        self.val_loss_step = 0

        print('')
        print('---Validation metrics---')

        if self.mode == 'pretraining':

            val_loss_epoch = np.mean([x.item() for x in self.validation_step_outputs])

            if self.current_epoch % 5 == 0:

                print('')
                print('---Logistic Regression metrics---')

                trainlogreg = []
                vallogreg = []
                testlogreg = []

                for i in range(0, self.data_train.slices.shape[0], 32):
                    end = np.min([i + 32, self.data_train.slices.shape[0]])
                    data = torch.as_tensor(self.data_train.slices[i:end,:])
                    test_img = recenter_patch((data / 250.))
                    encoded_img = self(test_img.cuda().type(torch.cuda.FloatTensor), mode="representation")
                    trainlogreg.append(np.asanyarray(encoded_img.cpu()))

                for i in range(0, self.data_val.slices.shape[0], 32):
                    end = np.min([i + 32, self.data_val.slices.shape[0]])
                    data = torch.as_tensor(self.data_val.slices[i:end,:])
                    test_img = recenter_patch((data / 250.))
                    encoded_img = self(test_img.cuda().type(torch.cuda.FloatTensor), mode="representation")
                    vallogreg.append(np.asanyarray(encoded_img.cpu()))

                for i in range(0, self.data_test.slices.shape[0], 32):
                    end = np.min([i + 32, self.data_test.slices.shape[0]])
                    data = torch.as_tensor(self.data_test.slices[i:end,:])
                    test_img = recenter_patch((data / 250.))
                    encoded_img = self(test_img.cuda().type(torch.cuda.FloatTensor), mode="representation")
                    testlogreg.append(np.asanyarray(encoded_img.cpu()))

                trainrep = np.concatenate(np.asanyarray(trainlogreg, dtype="object"))
                valrep = np.concatenate(np.asanyarray(vallogreg, dtype="object"))
                testrep = np.concatenate(np.asanyarray(testlogreg, dtype="object"))


                for c in [0.0001,0.01,0.1,0.5,1,10,100,1000]:
                    print('C=',c)
                    clf = LogisticRegression(C=c, max_iter=5000, penalty='l2', solver='lbfgs')  # C=15
                    clf.fit(trainrep, np.array(self.data_train.has_hcc))

                    y_proba_train = clf.predict_proba(trainrep)
                    y_proba_val = clf.predict_proba(valrep)
                    y_proba_test = clf.predict_proba(testrep)

                    auc_train = roc_auc_score(np.array(self.data_train.has_hcc),y_proba_train)
                    auc_val = roc_auc_score(np.array(self.data_val.has_hcc), y_proba_val)
                    auc_test = roc_auc_score(np.array(self.data_test.has_hcc), y_proba_test)
                    self.metrics['train_auc'].append(auc_train)
                    self.metrics['val_auc'].append(auc_val)
                    self.metrics['test_auc'].append(auc_test)
                    print('LogReg train auc',auc_train)
                    print('LogReg CHIC auc', auc_val)
                    print('LogReg internal test set auc', auc_test)
                    print('---END Logistic Regression metrics---')
                    print('')


            if self.current_epoch == self.config.max_epochs - 1:
                for i in range(0, self.data_train.slices.shape[0], 32):
                    end = np.min([i + 32, self.data_train.slices.shape[0]])
                    data = torch.as_tensor(self.data_train.slices[i:end,:])
                    test_img = recenter_patch((data / 250.))
                    encoded_img = self(test_img.cuda().type(torch.cuda.FloatTensor), mode="representation")
                    self.training_representations.append(np.asanyarray(encoded_img.cpu()))
                    self.training_inputs.append(np.asanyarray(test_img.cpu()))


                for i in range(0, self.data_val.slices.shape[0], 32):
                    end = np.min([i + 32, self.data_val.slices.shape[0]])
                    data = torch.as_tensor(self.data_val.slices[i:end,:])
                    test_img = recenter_patch((data / 250.))
                    encoded_img = self(test_img.cuda().type(torch.cuda.FloatTensor), mode="representation")
                    self.validation_representations.append(np.asanyarray(encoded_img.cpu()))
                    self.validation_inputs.append(np.asanyarray(test_img.cpu()))


        elif self.mode == 'finetuning':       # loss_P = 1/nb_slices_P sum_{s=1^S} Softmax(Net(slices_P_s))


            #### MULTI GPU MODIFS

            if int(os.environ['SLURM_GPUS_ON_NODE']) > 1:

                all_loss = self.all_gather(
                    torch.as_tensor([x.item() for x in self.validation_step_outputs]))
                all_preds = self.all_gather(torch.as_tensor(np.concatenate(self.y_pred_val)))
                all_true = self.all_gather(torch.as_tensor(np.concatenate(self.y_true_val)))

                y_true = torch.flatten(all_true).cpu().numpy()
                y_pred = torch.flatten(all_preds).detach().cpu().numpy()

                val_loss_epoch = np.mean(torch.flatten(all_loss).detach().cpu().numpy())


            else:

                val_loss_epoch = np.mean([x.item() for x in self.validation_step_outputs])
                y_true = torch.tensor(self.df_val['label'], dtype=torch.long)
                y_pred = torch.tensor(np.vstack(self.df_val['epoch_' + str(self.current_epoch)]))

            ### ROC-AUC score for HCC
            val_auc = roc_auc_score(y_true, y_pred)
            ## Balanced Accuracy per subject
            y_pred_youden = [1 if x > self.cutoff else 0 for x in y_pred]
            ## Confusion matrix per subject
            cm = confusion_matrix(y_true, y_pred_youden)
            print('val confusion matrix', cm)
            print('loss val epoch', val_loss_epoch)
            self.metrics['val_loss'].append(val_loss_epoch)
            ### END METRICS ###


            self.y_pred_val.clear()
            self.y_true_val.clear()


            probas_test = []
            probas_test_hcc = []
            for i in range(0, self.data_chicp.slices.shape[0], 32):
                end = np.min([i + 32, self.data_chicp.slices.shape[0]])
                data = torch.as_tensor(self.data_chicp.slices[i:end,:])
                #test_img = torch.as_tensor(np.asanyarray(recenter_patch((data / 250.)).cpu())[:, :, :, ::-1, ::-1].copy())
                test_img = recenter_patch((data / 250.))
                probas, probas_hcc = self(test_img.cuda().type(torch.cuda.FloatTensor),
                                          torch.as_tensor(self.data_chicp.z_pos[i:end]).cuda().type(torch.cuda.FloatTensor))
                probas_test_hcc.append(np.asanyarray(probas_hcc.cpu()))
                probas_test.append(np.asanyarray(probas.cpu()))

            ### ROC-AUC score
            probas_test = np.concatenate(probas_test)
            probas_test_hcc = np.concatenate(probas_test_hcc)

            y_true = self.data_chicp.has_hcc
            if self.config.label_name == 'multilabel':
                test_auc = roc_auc_score(y_true[:, -1], scipy.special.expit(probas_test_hcc))
                test_loss = self.loss(torch.tensor(probas_test).to(device='cuda'), torch.tensor(y_true).to(device='cuda'))
            else:
                test_auc = roc_auc_score(y_true, scipy.special.softmax(probas_test, axis=1)[:, 1])
                test_loss = self.loss(torch.tensor(probas_test),
                                      torch.tensor(y_true, dtype=torch.long))
            self.metrics['test_auc'].append(test_auc)
            self.metrics['test_loss'].append(test_loss)


            print('---Test metrics---')
            print('test auc', test_auc)
            print('---END test metrics---')


            if self.current_epoch == self.config.max_epochs - 1:

                print('CHIC train inference...')

                for i in range(0, self.data_train.slices.shape[0], 32):

                    end = np.min([i + 32, self.data_train.slices.shape[0]])
                    data = torch.as_tensor(self.data_train.slices[i:end,:])
                    test_img = recenter_patch((data / 250.))
                    encoded_img = self(test_img.type(torch.cuda.FloatTensor),
                                       torch.as_tensor(self.data_train.z_pos[i:end]).cuda().type(torch.cuda.FloatTensor),
                                       mode="representation")
                    probas, probas_hcc = self(test_img.cuda().type(torch.cuda.FloatTensor),
                                              torch.as_tensor(self.data_train.z_pos[i:end]).cuda().type(torch.cuda.FloatTensor))
                    self.training_representations.append(np.asanyarray(encoded_img.cpu()))
                    if self.config.label_name == 'multilabel':
                        self.training_probas.append(np.asanyarray(probas.cpu()))
                        y_pred_youden = [1 if x > self.cutoff else 0 for x in probas_hcc]
                    else:
                        self.training_probas.append(np.asanyarray(probas.cpu()))
                        y_pred_youden = [1 if x > self.cutoff else 0 for x in probas[:, 1]]
                    self.training_preds.append(np.asanyarray(y_pred_youden))
                    self.training_inputs.append(np.asanyarray(test_img.cpu()))


                print('CHIC val inference...')


                for i in range(0, self.data_val.slices.shape[0], 32):
                    end = np.min([i + 32, self.data_val.slices.shape[0]])
                    data = torch.as_tensor(self.data_val.slices[i:end,:])
                    test_img = recenter_patch((data / 250.))
                    encoded_img = self(test_img.type(torch.cuda.FloatTensor),
                                       torch.as_tensor(self.data_val.z_pos[i:end]).cuda().type(torch.cuda.FloatTensor),
                                       mode="representation")
                    probas, probas_hcc = self(test_img.cuda().type(torch.cuda.FloatTensor),
                                              torch.as_tensor(self.data_val.z_pos[i:end]).cuda().type(torch.cuda.FloatTensor))
                    self.validation_representations.append(np.asanyarray(encoded_img.cpu()))
                    if self.config.label_name == 'multilabel':
                        self.validation_probas.append(np.asanyarray(probas.cpu()))
                        y_pred_youden = [1 if x > self.cutoff else 0 for x in probas_hcc]
                    else:
                        self.validation_probas.append(np.asanyarray(probas.cpu()))
                        y_pred_youden = [1 if x > self.cutoff else 0 for x in probas[:, 1]]
                    self.validation_preds.append(np.asanyarray(y_pred_youden))
                    self.validation_inputs.append(np.asanyarray(test_img.cpu()))

                print('CALV test...')

                for i in range(0, self.data_test.slices.shape[0], 32):
                    end = np.min([i + 32, self.data_test.slices.shape[0]])
                    data = torch.as_tensor(self.data_test.slices[i:end, :])
                    #test_img = torch.as_tensor(np.asanyarray(recenter_patch((data / 250.)).cpu())[:, :, :, ::-1, ::-1].copy())
                    test_img = recenter_patch((data / 250.))
                    encoded_img = self(test_img.type(torch.cuda.FloatTensor),
                                       torch.as_tensor(self.data_test.z_pos[i:end]).cuda().type(torch.cuda.FloatTensor),
                                       mode="representation")
                    probas, probas_hcc = self(test_img.type(torch.cuda.FloatTensor),
                                              torch.as_tensor(self.data_test.z_pos[i:end]).cuda().type(torch.cuda.FloatTensor))
                    self.test_representations.append(np.asanyarray(encoded_img.cpu()))
                    if self.config.label_name == 'multilabel':
                        self.test_probas.append(np.asanyarray(probas.cpu()))
                        y_pred_youden = [1 if x > self.cutoff else 0 for x in probas_hcc]
                    else:
                        self.test_probas.append(np.asanyarray(probas.cpu()))
                        y_pred_youden = [1 if x > self.cutoff else 0 for x in probas[:, 1]]
                    self.test_preds.append(np.asanyarray(y_pred_youden))
                    self.test_inputs.append(np.asanyarray(test_img.cpu()))

                print('CHICP test...')

                for i in range(0, self.data_chicp.slices.shape[0], 32):
                    end = np.min([i + 32, self.data_chicp.slices.shape[0]])
                    data = torch.as_tensor(self.data_chicp.slices[i:end, :])
                    test_img = recenter_patch((data / 250.))
                    encoded_img = self(test_img.type(torch.cuda.FloatTensor),
                                       torch.as_tensor(self.data_chicp.z_pos[i:end]).cuda().type(torch.cuda.FloatTensor),
                                       mode="representation")
                    probas, probas_hcc = self(test_img.type(torch.cuda.FloatTensor),
                                              torch.as_tensor(self.data_chicp.z_pos[i:end]).cuda().type(torch.cuda.FloatTensor))
                    self.chicp_representations.append(np.asanyarray(encoded_img.cpu()))
                    if self.config.label_name == 'multilabel':
                        self.chicp_probas.append(np.asanyarray(probas.cpu()))
                        y_pred_youden = [1 if x > self.cutoff else 0 for x in probas_hcc]
                    else:
                        self.chicp_probas.append(np.asanyarray(probas.cpu()))
                        y_pred_youden = [1 if x > self.cutoff else 0 for x in probas[:, 1]]
                    self.chicp_preds.append(np.asanyarray(y_pred_youden))
                    self.chicp_inputs.append(np.asanyarray(test_img.cpu()))

                self.test_volumes = self.data_test.volumes
                self.test_labels = self.data_test.has_hcc
                self.chicp_volumes = self.data_chicp.volumes
                self.chicp_labels = self.data_chicp.has_hcc

            ### LOGS ###


            #self.log("val_auc", val_auc) # for monitoring
            print('---Validation metrics---')
            self.metrics['val_auc'].append(val_auc)
            print('val_auc', val_auc)
            print('---END validation metrics---')
            print('')

            ### END LOGS ###


        self.training_volumes = self.data_train.volumes
        self.training_labels = self.data_train.has_hcc
        self.validation_volumes = self.data_val.volumes
        self.validation_labels = self.data_val.has_hcc

        self.validation_step_outputs.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, T_max=self.config.max_epochs, eta_min=0, last_epoch=-1
        #)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                "monitor":'vl_epoch'}

    def load_model(self, checkpoint):

        if checkpoint is not None:

            checkpoint_ = torch.load(checkpoint)

            model_dict = self.state_dict()

            ### IF THIS IS BYOL
            """new_state_dict = OrderedDict()
            #print(checkpoint['state_dict'].items())
            for key, v in checkpoint_['state_dict'].items():
                name = key.replace("online_network.", "")  # remove `online_network.` and discard the target network
                new_state_dict[name] = v"""
            ### END BYOL

            # comment if BYOL
            new_state_dict = checkpoint_['state_dict']

            print(new_state_dict.keys())
            print(model_dict.keys())

            #if new_state_dict.keys() != model_dict.keys():
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(pretrained_dict, strict=False)
            print("Pretrained model loaded!")
            #else:
            #    self.load_state_dict(new_state_dict, strict=True)