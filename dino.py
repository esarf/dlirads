import sklearn.metrics
import torch
from torch.optim.lr_scheduler import ExponentialLR
from collections import OrderedDict
# from sam import SAM
from models.gradcam import grad_cam
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.nn.functional import softmax
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
from typing import Sequence, Union
from torch.nn import Module
import sklearn.metrics
from torch.autograd import Variable
import torch
from torch.optim.lr_scheduler import ExponentialLR
from collections import OrderedDict
# from sam import SAM
from pytorch_lightning.callbacks import Callback
from models.gradcam import grad_cam
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, Subset
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
from models.vit import DINOBackbone, DINOHead
import copy
from torchvision.transforms import v2




class DINOAugmentation(torch.nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, n_local_views) -> None:
        super().__init__()

        global_transforms_0 = v2.Compose([
            #v2.RandomHorizontalFlip(p=0.5),
            #v2.RandomVerticalFlip(p=0.5),
            #v2.RandomRotation(degrees=30),
            v2.RandomResizedCrop(size=(512, 512), scale=(0.7, 0.7))])

        global_transforms_1 = torch.nn.Sequential(
            v2.RandomResizedCrop(size=(512, 512), scale=(0.7, 0.7)),
        )

        local_transforms = torch.nn.Sequential(
            v2.RandomResizedCrop(size=(512, 512), scale=(0.4, 0.4)),
        )

        self.local_transforms = [local_transforms] * n_local_views
        self.transforms = [global_transforms_0, global_transforms_1]
        self.transforms.extend(self.local_transforms)

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = tuple([transform(x) for transform in self.transforms])  # Bx(2+n_local_views)xHxW
        return x_out

        return x_out


def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, mode='encoder'):
        """
        x: list of input image tensors
        """
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0) # [2, 10] for student, [2] for teacher
        start_idx = 0
        out = []
        for end_idx in idx_crops:
            out += [self.backbone(torch.cat(x[start_idx:end_idx]))]
            start_idx = end_idx
        return self.head(torch.cat(out),mode=mode)


class DINOEMAWeightUpdate(Callback):
    """Weight update rule from BYOL.
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[BYOLMAWeightUpdate()])
    """

    def __init__(self, initial_tau: float = 0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.student
        target_net = pl_module.teacher

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: LightningModule, trainer: Trainer) -> float:
        max_steps = 10000 * int(trainer.max_epochs)
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_net: Union[Module, Tensor], target_net: Union[Module, Tensor]) -> None:
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(
            online_net.named_parameters(),
            target_net.named_parameters(),
        ):
            target_p.data = self.current_tau * target_p.data + (1 - self.current_tau) * online_p.data


def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

class DINOModel(pl.LightningModule):

    def __init__(self, net, loss, config, data_train, data_val, scheduler=None):
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
        self.config = config

        student_backbone = DINOBackbone(patch_size=self.config.patch_size,
                                        embed_dim=self.config.rep_dim,
                                        hidden_dim=self.config.hidden_dim_vit,
                                        num_heads=self.config.num_heads,
                                        num_layers=self.config.num_layers)
        self.teacher_backbone = copy.deepcopy(student_backbone)

        student_head = DINOHead(embed_dim=self.config.rep_dim,
                                num_classes=self.config.num_classes,
                                hidden_dim_cl=self.config.hidden_dim,
                                output_dim=self.config.output_dim,
                                mode='encoder',
                                freeze_last_layer=False)
        teacher_head = DINOHead(embed_dim=self.config.rep_dim,
                                num_classes=self.config.num_classes,
                                hidden_dim_cl=self.config.hidden_dim,
                                output_dim=self.config.output_dim,
                                mode='encoder')

        self.student = MultiCropWrapper(student_backbone, student_head)
        self.teacher = MultiCropWrapper(self.teacher_backbone, teacher_head)
        # teacher is not trained
        for p in self.teacher.parameters(): p.requires_grad = False
        self.weight_callback = DINOEMAWeightUpdate()


        self.data_train = data_train
        self.data_val = data_val
        self.mode = config.mode
        self.imgs_epoch = torch.zeros(len(self.data_val),self.config.rep_dim)
        self.transforms = DINOAugmentation(n_local_views=3)
        self.val_metric_model = self.config.val_metric_model

        self.nb_steps = len(data_val) // self.config.batch_size + 1
        self.val_loss_step = 0
        self.cutoff = 0.5
        self.vols_train = []
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.training_representations = []
        self.validation_representations = []
        self.training_probas = []
        self.validation_probas = []

        assert self.mode in ['finetuning', 'pretraining', 'test'], ('self.mode =', self.mode)

        #if config.mode == 'pretraining':
        #    df_labels = pd.read_csv(join(config.path_to_data, '1_subjects_label.csv'), delimiter=",",
        #                            index_col='subject')

        if self.config.mode == 'finetuning' or self.config.model_type == "supervised" or self.config.model_type == "supsim" or self.config.model_type == "ysimclr":

            # TRAINING PART
            index_train = self.data_train.volumes                                     # liste des volumes dans toute la base d'entraînement
            self.df_train = pd.DataFrame({'epoch_'+str(i): [np.zeros(self.config.num_classes) for x in index_train] for i in range(500)},
                                         index=index_train)
                                                                                      # df_train: contient la moyenne des probas par patient sur toute l'epoch (moyenne par step)
            dic_train = dict(Counter(self.data_train.volumes))                        # a dictionary that maps the volumes to their occurrences inside the validation dataset
            self.df_train['nb_slices'] = [dic_train[x] for x in self.df_train.index]  # nombre de slices associé à chaque volume dans la base de données
            self.df_train['label'] = list(self.data_train.has_hcc)         # label associé à chaque volume dans la base de données

            # VALIDATION PART
            index_val = self.data_val.volumes
            self.df_val = pd.DataFrame({'epoch_'+str(i): [np.zeros(self.config.num_classes) for x in index_val] for i in range(500)},
                                       index=index_val)
                                                                                      # df_train: contient la moyenne des probas par patient sur toute l'epoch (moyenne par step)
            dic_val = dict(Counter(self.data_val.volumes))                            # a dictionary that maps the volumes to their occurrences inside the validation dataset
            self.df_val['nb_slices'] = [dic_val[x] for x in self.df_val.index]
            self.df_val['label'] = list(self.data_val.has_hcc)             # label associé à chaque volume dans la base de données


        """if self.config.mode == 'pretraining':

            self.dataset_train_grenoble = DatasetCopy(config, training=True, grenoble=True, dimension=config.dimension)
            self.dataset_val_grenoble = DatasetCopy(config, validation=True, grenoble=True, dimension=config.dimension)
        """

        if hasattr(config, 'pretrained_path') and config.pretrained_path is not None:
            self.load_model(config.pretrained_path)


    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # discard the version number
        items.pop("v_num", None)
        return items

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Add callback to perform exponential moving average weight update on target network."""
        self.weight_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx)

    def forward(self, x, mode=None):
        y = self.teacher_backbone(x)
        return y

    def shared_step(self, batch, batch_idx):

        inputs, labels, subjects_id, z = batch

        ## Augmentations with Kornia
        aug = self.transforms(inputs / 250.) #tuple of 2+n_local_views of tensors of shapes (8,1,512,512)

        ## Forward pass
        teacher_output = self.teacher(aug[:2])
        student_output = self.student(aug)

        ## Compute the loss
        loss = self.loss(student_output, teacher_output, self.current_epoch)

        ## Visualization on TensorBoard
        if batch_idx == 0:
            self.logger.experiment.add_image("Images/Global augmentation", torch.unsqueeze((aug)[0][batch_idx, 0, :],0).cpu(),
                                             self.current_epoch)
            self.logger.experiment.add_image("Images/Local augmentation", torch.unsqueeze((aug)[3][batch_idx, 0, :],0).cpu(),
                                             self.current_epoch)

        return loss

    def training_step(self, batch, batch_idx):

        inputs, labels, subjects_id, z = batch          # batch : N x (images augmentées (1 et 2), label (age), subject_id)
                                                        # inputs shape      (batch_size,1 ou 3,1,512,512)
                                                        # labels shape      (batch_size,1)
                                                        # subjects_id shape (batch_size,1)

        #print('batch hcc labels proportion',torch.sum(labels) / len(labels))

        assert self.mode in ['finetuning', 'pretraining'], ('self.mode =', self.mode)

        if self.mode == "pretraining":

            print('teacher output first line 10 elements in rep mode',self(inputs/250.)[0][:10])

            # Final loss
            loss = self.shared_step(batch, batch_idx)

        elif self.mode == "finetuning":

            aug = self.transforms(inputs / 250.)                        # inputs shape: (batch_size,1,512,512)

            ## Forward pass
            y = self(aug)  # y shape (batch_size,C)
            ## Compute the Loss
            loss = self.loss(y, labels)  # loss : cross-entropy loss

            ## Output probabilities of class 1 on the batch
            y_prob = softmax(y, dim=1).detach().cpu().numpy()   # shape (batch_size,C)
            ## Fill the epoch-level metrics dataframe : we compute the average probability over the whole epoch
            ep = 'epoch_' + str(self.current_epoch)
            for i, subject in enumerate(subjects_id):
                for k in range(self.config.num_classes):
                    #self.df_train.loc[subject, ep][k] += np.nan_to_num(y_prob[i, k] / self.df_train.loc[subject, 'nb_slices'])
                    self.df_train.loc[subject, ep][k] += np.nan_to_num(y_prob[i, k])

            if batch_idx == 0:
                if self.config.dimension == 3:
                    self.logger.experiment.add_image("Images/OriginalImage",
                                                     (inputs / 250.)[batch_idx, :, 0, :].cpu(),
                                                     self.current_epoch)
                    self.logger.experiment.add_image("Images/Augmentation", (aug)[batch_idx, :, 0, :].cpu(),
                                                     self.current_epoch)
                else:
                    self.logger.experiment.add_image("Images/OriginalImage",
                                                     torch.unsqueeze((inputs / 250.)[batch_idx, 0, :],0).cpu(),
                                                     self.current_epoch)
                    self.logger.experiment.add_image("Images/AugmentationChannel1",
                                                     torch.unsqueeze((aug)[batch_idx, 0, :],0).cpu(),
                                                     self.current_epoch)
                    self.logger.experiment.add_image("Images/AugmentationChannel2",
                                                     torch.unsqueeze((aug)[batch_idx, 1, :], 0).cpu(),
                                                     self.current_epoch)

            # take the first image for all gradcams
            if self.global_step == 0:
                idx_label_0 = np.argwhere(np.array(labels.cpu().numpy()) == 0)[0]
                idx_label_1 = np.argwhere(np.array(labels.cpu().numpy()) == 1)[0]
                self.img_gradcam_0 = inputs[idx_label_0,:] / 250.
                self.img_gradcam_1 = inputs[idx_label_1,:] / 250.

        # accumulate the sampled volumes through the epoch
        self.vols_train.append(subjects_id)

        ### LOGS ###
        self.logger.experiment.add_scalar("Training/training_loss_step", loss, self.global_step)
        self.logger.experiment.add_scalars("TrainVal/training_val_loss_step", {'train': loss}, self.global_step)
        ## we log the loss of the first batch as the training loss at iteration 0 (before optimization)
        if self.global_step == 0:
            self.logger.experiment.add_scalar("Training/training_loss_epoch", loss, 0)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'train': loss}, 0)
        ### END LOGS ###

        self.training_step_outputs.append(loss)

        return loss


    def on_train_epoch_end(self):
        #train_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean().item()
        train_loss_epoch = np.mean([x.item() for x in self.training_step_outputs])
        self.log('tl_epoch', train_loss_epoch)

        if self.mode == 'pretraining':
            self.logger.experiment.add_scalar("Training/training_loss_epoch", train_loss_epoch, self.current_epoch)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'train': train_loss_epoch}, self.current_epoch)

        elif self.mode == 'finetuning':

            vols_train = list(itertools.chain(*self.vols_train))
            dic_train = dict(Counter(vols_train))
            self.df_train['nb_slices'] = [dic_train[x] if x in list(dic_train.keys()) else 1 for x in
                                          self.df_train.index]
            self.df_train['epoch_' + str(self.current_epoch)] /= self.df_train['nb_slices']
            self.vols_train = []

            if self.config.multiclass:
                ### EPOCH-LEVEL METRICS ###
                y_true = torch.nn.functional.one_hot(torch.tensor(self.df_train['class'],
                                                                  dtype=torch.long),
                                                                  num_classes=self.config.num_classes)  # shape N_patients, C
                y_pred = torch.tensor(np.vstack(self.df_train['epoch_' + str(self.current_epoch)]))     # shape N_patients, C
                ## Training loss per subject
                loss_vector = torch.sum(torch.mul(torch.log(y_pred), y_true), dim=1)                    # shape N_patients, 1

            else:
                y_true = torch.tensor(self.df_train['label'], dtype=torch.long)
                y_pred = torch.tensor(np.vstack(self.df_train['epoch_' + str(self.current_epoch)])[:, 1])
                fpr_train, tpr_train, thresholds = roc_curve(y_true, y_pred)
                self.cutoff = cutoff_youdens_j(fpr_train, tpr_train, thresholds)
                #y_pred_label = [1 if x > cutoff else 0 for x in y_pred]

                ## Training loss per subject
                loss_vector = torch.mul(torch.log(y_pred), y_true) + torch.mul(torch.log(1 - y_pred), 1 - y_true)

            train_loss_sub = -torch.mean(torch.nan_to_num(loss_vector, neginf=0, posinf=0))             # shape 1
            ### ROC-AUC score per subject
            train_auc = roc_auc_score(y_true, y_pred)
            ### END METRICS ###

            ### LOGS ###
            self.logger.experiment.add_scalar("Training/training_AUC", train_auc, self.current_epoch)
            self.logger.experiment.add_scalar("Training/training_loss_epoch", train_loss_epoch, self.current_epoch)
            self.logger.experiment.add_scalar("Training/training_loss_subject", train_loss_sub, self.current_epoch)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'train': train_loss_epoch},
                                               self.current_epoch)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_subject", {'train': train_loss_sub},
                                               self.current_epoch)
            ### END LOGS ###

            if self.current_epoch > 0:
                self.eval()
                self.img_gradcam_0.requires_grad = True
                input_tensor = self.img_gradcam_0.squeeze(0)
                if self.config.encoder == 'baseline' or self.config.encoder == 'small':
                    heatmap_layer = self.net.features_conv
                else:
                    heatmap_layer = self.net.layer4[-1].conv2
                superimpose, hm = grad_cam(self.net, input_tensor, heatmap_layer, truelabel=0)
                print('Prediction label 0',F.softmax(self(input_tensor.unsqueeze(0).float()), dim=1))
                plt.imshow(np.transpose(superimpose,(1, 0, 2))[::-1,::-1])
                self.logger.experiment.add_figure("GradCam/img0",
                                                  plt.gcf(),
                                                  self.current_epoch)
                plt.imshow(np.transpose(hm, (1, 0))[::-1, ::-1])
                self.logger.experiment.add_figure("GradCam/heatmap0",
                                                  plt.gcf(),
                                                  self.current_epoch)

                self.img_gradcam_1.requires_grad = True
                input_tensor = self.img_gradcam_1.squeeze(0)
                if self.config.encoder == 'baseline' or self.config.encoder == 'small':
                    heatmap_layer = self.net.features_conv
                else:
                    heatmap_layer = self.net.layer4[-1].conv2
                superimpose, hm = grad_cam(self.net, input_tensor, heatmap_layer, truelabel=1)
                print('Prediction label 1', F.softmax(self(input_tensor.unsqueeze(0).float()), dim=1))
                plt.imshow(np.transpose(superimpose, (1, 0, 2))[::-1, ::-1])
                self.logger.experiment.add_figure("GradCam/img1",
                                                  plt.gcf(),
                                                  self.current_epoch)
                plt.imshow(np.transpose(hm, (1, 0))[::-1, ::-1])
                self.logger.experiment.add_figure("GradCam/heatmap1",
                                                  plt.gcf(),
                                                  self.current_epoch)



    def validation_step(self, batch, batch_idx):

        inputs, labels, subjects_id, z = batch          # batch : N x (images augmentée (1 et 2), label (age), subject_id)
                                                        # inputs shape      (batch_size, 1 ou 3, 1, 512,512)
                                                        # labels shape      (batch_size)
                                                        # subjects_id       (batch_size)


        assert self.mode in ['finetuning', 'pretraining'], ('self.mode =', self.mode)

        if self.mode == "pretraining":

            val_loss_step = self.shared_step(batch, batch_idx)
            self.val_loss_step += val_loss_step / self.nb_steps  # for the log of the avg
            # no separation with the sanity check because we run the forward on the whole validation set


        elif self.mode == "finetuning":

            ## Forward pass
            y = self(inputs / 250.)  # y shape (batch_size,2)
            ## Compute the Loss
            val_loss_step = self.loss(y, labels)  # loss : cross-entropy loss
            self.val_loss_step += val_loss_step / self.nb_steps  # for the log of the avg

            ## Output probabilities of class 1 on the batch
            y_prob = softmax(y, dim=1).detach().cpu().numpy()

            ## Fill the epoch-level metrics dataframe : we compute the average probability over the whole epoch
            ep = 'epoch_' + str(self.current_epoch)
            for i, subject in enumerate(subjects_id):
                for k in range(self.config.num_classes):
                    self.df_val.loc[subject, ep][k] += np.nan_to_num(
                        y_prob[i, k] / self.df_val.loc[subject, 'nb_slices'])


        ### LOGS ###
        self.logger.experiment.add_scalar("Validation/val_loss_step", self.val_loss_step, self.global_step+1)
        self.logger.experiment.add_scalars("TrainVal/training_val_loss_step", {'val': self.val_loss_step},
                                           self.global_step+1)
        # we log the first forward pass on 2 batches from the sanity check before any optimization
        if self.trainer.sanity_checking:
            self.logger.experiment.add_scalar("Validation/val_loss_epoch", self.val_loss_step, 0)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'val': self.val_loss_step}, 0)
        ### END LOGS ###

        self.validation_step_outputs.append(val_loss_step)

        return val_loss_step

    def on_validation_epoch_end(self):

        self.val_loss_step = 0
        val_loss_epoch = np.mean([x.item() for x in self.validation_step_outputs])
        self.log('vl_epoch',val_loss_epoch)

        print('val loss epoch', val_loss_epoch)

        if self.mode == 'pretraining':

            if self.trainer.sanity_checking:
                self.logger.experiment.add_scalar("Validation/val_loss_epoch", val_loss_epoch, 0)
                self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'val': val_loss_epoch}, 0)

            else:
                self.logger.experiment.add_scalar("Validation/val_loss_epoch", val_loss_epoch, self.current_epoch + 1)
                self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'val': val_loss_epoch},
                                               self.current_epoch + 1)

            if self.current_epoch == self.config.max_epochs - 1:

                for i in range(0, self.data_train.slices.shape[-1], 32):
                    end = np.min([i + 32, self.data_train.slices.shape[-1]])
                    data = torch.as_tensor(self.data_train.slices[:, :, :, i:end])
                    data = data.permute(3, 0, 1, 2)
                    test_img = (data / 250.)
                    encoded_img = self(test_img.cuda(), mode="representation")
                    self.training_representations.append(np.asanyarray(encoded_img.cpu()))

                for i in range(0, self.data_val.slices.shape[-1], 32):
                    end = np.min([i + 32, self.data_val.slices.shape[-1]])
                    data = torch.as_tensor(self.data_val.slices[:, :, :, i:end])
                    data = data.permute(3, 0, 1, 2)
                    test_img = (data / 250.)
                    encoded_img = self(test_img.cuda(), mode="representation")
                    self.validation_representations.append(np.asanyarray(encoded_img.cpu()))


        elif self.mode == 'finetuning':       # loss_P = 1/nb_slices_P sum_{s=1^S} Softmax(Net(slices_P_s))

            if self.config.cross_val:
                self.df_val.to_csv(join(self.config.lght_dir, 'df_val'+str(self.config.cv_fold)), sep=';', index='subject')
            else:
                self.df_val.to_csv(join(self.config.lght_dir, 'df_val'),
                                   sep=';', index='subject')

            if self.config.multiclass:
                ### EPOCH-LEVEL METRICS ###
                y_true = torch.nn.functional.one_hot(torch.tensor(self.df_val['class'],
                                                                  dtype=torch.long),
                                                                  num_classes=self.config.num_classes)   # shape N_patients, C
                y_pred = torch.tensor(
                    np.vstack(self.df_val['epoch_' + str(self.current_epoch)]))             # shape N_patients, C

                ## Training loss per subject
                loss_vector = torch.sum(torch.mul(torch.log(y_pred), y_true), dim=1)        # shape N_patients, 1
                val_loss_sub = -torch.mean(torch.nan_to_num(loss_vector, neginf=0, posinf=0))  # shape 1
                ### ROC-AUC score per subject
                val_auc = roc_auc_score(y_true, y_pred, multi_class='ovo')
                ## Balanced Accuracy per subject
                val_acc = balanced_accuracy_score(y_true.argmax(dim=1), np.rint(y_pred).argmax(axis=1))
                ## Confusion matrix per subject
                cm = confusion_matrix(y_true.argmax(dim=1), np.rint(y_pred).argmax(axis=1))
                ### END METRICS ###

            else:
                y_true = torch.tensor(self.df_val['label'], dtype=torch.long)
                y_pred = torch.tensor(np.vstack(self.df_val['epoch_' + str(self.current_epoch)])[:, 1])

                ## Training loss per subject
                loss_vector = torch.mul(torch.log(y_pred), y_true) + torch.mul(torch.log(1 - y_pred), 1 - y_true)
                val_loss_sub = -torch.mean(torch.nan_to_num(loss_vector, neginf=0, posinf=0))   # shape 1
                ### ROC-AUC score per subject
                val_auc = roc_auc_score(y_true, y_pred)
                ## Balanced Accuracy per subject
                y_pred_youden = [1 if x > self.cutoff else 0 for x in y_pred]
                val_acc = balanced_accuracy_score(y_true, y_pred_youden)
                ## Confusion matrix per subject
                cm = confusion_matrix(y_true, np.rint(y_pred))
                ### END METRICS ###

            if self.current_epoch == self.config.max_epochs - 1:

                for i in range(0, self.data_train.slices.shape[-1], 32):
                    end = np.min([i + 32, self.data_train.slices.shape[-1]])
                    data = torch.as_tensor(self.data_train.slices[:, :, :, i:end])
                    data = data.permute(3, 0, 1, 2)
                    test_img = (data / 250.)
                    encoded_img = self(test_img.cuda(), mode="representation")
                    probas = self(test_img.cuda())
                    self.training_representations.append(np.asanyarray(encoded_img.cpu()))
                    self.training_probas.append(np.asanyarray(probas.cpu()))

                for i in range(0, self.data_val.slices.shape[-1], 32):
                    end = np.min([i + 32, self.data_val.slices.shape[-1]])
                    data = torch.as_tensor(self.data_val.slices[:, :, :, i:end])
                    data = data.permute(3, 0, 1, 2)
                    test_img = (data / 250.)
                    encoded_img = self(test_img.cuda(), mode="representation")
                    probas = self(test_img.cuda())
                    self.validation_representations.append(np.asanyarray(encoded_img.cpu()))
                    self.validation_probas.append(np.asanyarray(probas.cpu()))

            ### LOGS ###

            self.log("val_auc", val_auc) # for monitoring

            self.logger.experiment.add_scalar("Validation/val_loss_epoch", val_loss_epoch, self.current_epoch)
            self.logger.experiment.add_scalar("Validation/val_loss_subject", val_loss_sub, self.current_epoch)
            self.logger.experiment.add_scalar("Validation/val_AUC", val_auc, self.current_epoch)
            self.logger.experiment.add_scalar("Validation/val_accuracy", val_acc, self.current_epoch)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_epoch", {'val': val_loss_epoch},
                                               self.current_epoch)
            self.logger.experiment.add_scalars("TrainVal/training_val_loss_subject", {'val': val_loss_sub},
                                               self.current_epoch)

            plt.figure(figsize=(12, 7))
            plt.yticks([0.5, 1.5], ['No HCC', 'HCC'])
            plt.xticks([0.5, 1.5], ['No HCC', 'HCC'])
            self.logger.experiment.add_figure("Metrics/ConfusionMatrix",
                                              sns.heatmap(cm, annot=True, fmt="d",
                                                          cmap="Blues").get_figure(),
                                              self.current_epoch+1)

            ### END LOGS ###

        self.training_volumes = self.data_train.volumes
        self.training_labels = self.data_train.has_hcc
        self.validation_volumes = self.data_val.volumes
        self.validation_labels = self.data_val.has_hcc


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.config.max_epochs, eta_min=0, last_epoch=-1
        )

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