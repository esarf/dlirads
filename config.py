from os.path import join
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from dataset import Dataset
from sklearn import svm
import os


class Config:

    def __init__(self,
                 mode: str = 'finetuning',
                 database: str = 'grenoble',
                 rep_dim: int = 256,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_classes: int = 2,
                 encoder: str = 'baseline',
                 lr: float = 1e-5,
                 weight_decay: float = 1e-5,
                 label_name: str = 'has_hcc',
                 n_fold: int = 4,
                 cross_val: bool = False,
                 dir: str = 'calv_baseline',
                 pretrained_path: str = None,
                 sigma: float = 0.85,
                 temperature: float = 0.1,
                 kernel: str = 'rbf',
                 max_epochs: int = 40,
                 grenoble: bool = False,
                 model_type: str = 'supervised',
                 batch_size: int = 64,
                 pretrained: bool = False,
                 segmask: bool = True,
                 exp: str = 'baseline',
                 input_dim: int = 2,
                 metadata: str = 'diameter_2d',
                 hidden_dim_vit=512,
                 num_heads=2,
                 num_layers=4,
                 patch_size=16,
                 augment='torch',
                 experiment=None
                 ):

        assert mode in {"pretraining", "finetuning", "test"}, "Unknown mode: %i"%mode

        self.mode = mode
        self.input_size = (1, 512, 512)
        self.nb_cpu = 1
        # print('number of cpu',int(os.cpu_count()))
        self.cuda = True
        self.scheduler = 0.987
        self.database = database
        #self.path_to_data = "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/3_calv_ct_hcc__volume__numpy"    FOR TRAINING LIKE ECR
        self.path_to_data = "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/5_deeptekcalv_ct_hcc__volume__numpy__preprocessed_new"
        #self.path_to_data = "/datadrive/emma/2022_contrastive/2_datasets/3_calv_ct_hcc__volume__numpy"

        # Freeze the weights of the feature extractor or not
        self.freeze_weights = False
        # encoder type
        self.encoder = encoder
        self.pretrained = pretrained
        self.segmask = segmask
        self.n_layer = 18
        self.rep_dim = rep_dim              # representation space for downstream tasks
        self.hidden_dim = hidden_dim        # space between the representation space and the "loss space"
        self.output_dim = output_dim        # space where the loss is computed
        self.num_classes = num_classes
        self.slices_epoch = 5000
        self.exp = exp
        # data augmentation on CPU or GPU
        self.augmentation = "GPU"
        self.augment = augment
        # validation metric model
        self.val_metric_model = None
        self.cross_val = cross_val
        self.n_fold = n_fold
        self.val_rate = 1.0
        self.dimension = 2
        self.model_type = model_type
        self.dir = dir
        self.grenoble = grenoble
        self.dimension = 2
        self.input_dim = input_dim
        self.metadata = metadata

        # Optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        # Checkpoint
        self.pretrained_path = pretrained_path

        # Vision Transformers parameters
        self.hidden_dim_vit = hidden_dim_vit
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.experiment = experiment

        if self.mode == "pretraining":

            self.label_name = label_name   # CLASS FOR Y AWARE
            self.lght_dir = "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/3_models/pretraining_hcc"
            #self.lght_dir = "/datadrive/emma/2022_contrastive/3_models/pretraining_hcc"

            # saving weights
            self.ckpt = True
            self.ckpt_name = "pretraining"

            self.train_set = 'train_set'
            self.val_set = 'val_set'
            self.test_set = 'test_set'
            self.pin_mem = True

            # Hyperparameters for our y-Aware InfoNCE Loss
            self.kernel = kernel
            self.sigma = sigma # depends on the meta-data at hand
            self.temperature = temperature


        elif self.mode == "finetuning":
            self.train_set = 'train_set'
            self.val_set = 'val_set'
            self.test_set = 'test_set'

            # Tensorboard and saving csv
            self.lght_dir = "/gpfswork/rech/cwn/ufd78nr/emma/2022_contrastive/3_models/finetuning_hcc"

            # Saving weights
            self.ckpt = True
            self.ckpt_name = "classif_calv"

            self.pin_mem = True

            self.label_name = label_name

            if self.pretrained:
                self.input_dim = 3
            else:
                self.input_dim = input_dim