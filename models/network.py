import models.densenet2d as dense
import models.resnet as res
#import models.densenet3d as dense3d
import models.resnet3d as res3d
import models.resnet_monai as resmonai
from models.baseline_network import BaselineNet, SmallNet, SmallNet3D, BaselineNet3D
#import autoencoder as ae
from models.vit import VisionTransformer

def network(mode="encoder",
            net="densenet",
            pretrained=False,
            n_layer=18,
            output_dim=128,
            num_classes=2,
            rep_dim=512,
            hidden_dim=256,
            input_dim=2):

    if net == "densenet":
        output = dense.densenet121(pretrained=pretrained, mode=mode, output_dim=output_dim, num_classes=num_classes)
    elif net == "resnet":

        """
        output = res3d.resnet18(pretrained=pretrained,
                                mode=mode,
                                output_dim=output_dim,
                                num_classes=num_classes,
                                rep_dim=rep_dim,
                                hidden_dim=hidden_dim,
                                input_dim=input_dim)"""
        if pretrained:
            pretrained_ = '/lustre/fswork/projects/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/resnet_18_23dataset.pth'
            #pretrained_ = '/lustre/fswork/projects/rech/cwn/ufd78nr/emma/2022_contrastive/2_datasets/resnet_50_23dataset.pth'
        else:
            pretrained_ = False
        output = resmonai.resnet18(pretrained=pretrained_,
                                    mode=mode,
                                    num_classes=num_classes,
                                    n_input_channels=input_dim,
                                    spatial_dims=3)

    elif net == "baseline":
        output = BaselineNet3D(pretrained=pretrained,
                         num_classes=num_classes,
                         mode=mode,
                         rep_dim=rep_dim,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         in_channels=input_dim)

    elif net == "small":
        output = SmallNet3D(pretrained=pretrained,
                         num_classes=num_classes,
                         mode=mode,
                         rep_dim=rep_dim,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         in_channels=input_dim)

    elif net == "vit":
        output = VisionTransformer()

    elif net == "resnetautoencoder":
        output = ae.VAE(z_dim=rep_dim)

    return output
