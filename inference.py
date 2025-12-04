import sklearn.metrics
import torch
from torch.optim.lr_scheduler import ExponentialLR
from collections import OrderedDict
# from sam import SAM
import pytorch_lightning as pl
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, Subset
from dataset import Dataset
from sklearn.decomposition import PCA
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from ignite.contrib.metrics import ROC_AUC
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, balanced_accuracy_score
import numpy as np
import psutil
from dataset import Dataset
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




def inference(config,path,type,data_train,data_val,data_test):

    weights_path = join(path, type)

    # we load the model
    checkpoint = torch.load(weights_path)

    if model_type == "resnet":
        model = res3d.resnet18(mode=config.mode,
                                output_dim=config.output_dim,
                                num_classes=config.num_classes,
                                rep_dim=config.rep_dim,
                                hidden_dim=config.hidden_dim)
    elif model_type == 'small':
        model = SmallNet3D(num_classes=config.num_classes,
                             mode=config.mode,
                             rep_dim=config.rep_dim,
                             hidden_dim=config.hidden_dim,
                             output_dim=config.output_dim,
                             in_channels=config.input_dim)

    # create new OrderedDict that does not contain `net.`

    new_state_dict = OrderedDict()
    for key, v in checkpoint['state_dict'].items():
        name = key.replace("net.", "")  # remove `net.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    for i in range(0, data_train.slices.shape[0], 32):
        end = np.min([i + 32, data_train.slices.shape[0]])
        data = torch.as_tensor(data_train.slices[i:end, :])
        test_img = (data / 250.)
        encoded_img = model(test_img.cuda().type(torch.cuda.FloatTensor), mode="representation")
        trainlogreg.append(np.asanyarray(encoded_img.cpu()))

    for i in range(0, data_val.slices.shape[0], 32):
        end = np.min([i + 32, data_val.slices.shape[0]])
        data = torch.as_tensor(self.data_val.slices[i:end, :])
        test_img = (data / 250.)
        encoded_img = model(test_img.cuda().type(torch.cuda.FloatTensor), mode="representation")
        vallogreg.append(np.asanyarray(encoded_img.cpu()))

    for i in range(0, data_test.slices.shape[0], 32):
        end = np.min([i + 32, data_test.slices.shape[0]])
        data = torch.as_tensor(data_test.slices[i:end, :])
        test_img = (data / 250.)
        encoded_img = model(test_img.cuda().type(torch.cuda.FloatTensor), mode="representation")
        testlogreg.append(np.asanyarray(encoded_img.cpu()))

    print('Saving outputs...')
    np.save(join(config.lght_dir, config.dir, 'metrics.npy'), model.metrics,
            allow_pickle=True)
    np.save(join(config.lght_dir, config.dir, '_training_rep.npy'),
            np.concatenate(np.asanyarray(model.training_representations, dtype="object")),
            allow_pickle=True)
    np.save(join(config.lght_dir, config.dir, '_validation_rep.npy'),
            np.concatenate(np.asanyarray(model.validation_representations, dtype="object")),
            allow_pickle=True)
    np.save(join(config.lght_dir, config.dir, '_test_rep.npy'),
            np.concatenate(np.asanyarray(model.test_representations, dtype="object")),
            allow_pickle=True)
    np.save(join(config.lght_dir, config.dir, '_training_volumes.npy'),
            np.array(model.training_volumes),
            allow_pickle=True)
    np.save(join(config.lght_dir, config.dir, '_validation_volumes.npy'),
            np.array(model.validation_volumes),
            allow_pickle=True)
    np.save(join(config.lght_dir, config.dir, '_test_volumes.npy'),
            np.array(model.test_volumes),
            allow_pickle=True)
    np.save(join(config.lght_dir, config.dir, '_training_labels.npy'),
            np.array(model.training_labels),
            allow_pickle=True)
    np.save(join(config.lght_dir, config.dir, '_validation_labels.npy'),
            np.array(model.validation_labels),
            allow_pickle=True)
    np.save(join(config.lght_dir, config.dir, '_test_labels.npy'),
            np.array(model.test_labels),
            allow_pickle=True)
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
    print('Outputs saved...')