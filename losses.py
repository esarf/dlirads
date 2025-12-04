# Third party import
import logging
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances, cosine_similarity, laplacian_kernel
import numpy as np
from einops import reduce


def apply_weighting(x):
    if x == 0: return 1
    elif x == 1: return 0.5
    else: return 0

vec_weights = np.vectorize(apply_weighting, otypes=[float])

def apply_weighting(x):
    return np.exp(-x)

vec = np.vectorize(apply_weighting, otypes=[float])


class WeightedMultilabelLoss(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabelLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        return (loss * self.weights).mean()



class SupConLoss(nn.Module):
    def __init__(self, config, temperature=0.1, return_logits=False):
        """
        config: configuration file containing the main info for training
        temperature: 'tau' parameter specific to InfoNCE loss
        """

        # sigma = prior over the label's range
        super().__init__()
        self.temperature = temperature
        self.config = config
        self.return_logits = return_logits

    def forward(self, z_i, z_j, labels):
        N = len(z_i)
        id_mat = torch.eye(2*N, device=z_i.device)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]
        z = torch.cat([z_i, z_j], dim=0)         # dim [2N, D]
        sim_mat = (z @ z.T) / self.temperature  # shape [2N, 2N]
        sim_mat = ( sim_mat * ( 1 - id_mat ) ) - id_mat * 1e8  # similarity matrix: the diag has to be removed
        labs = func.one_hot(torch.tensor(labels, device=z_i.device).long(),
                            num_classes=self.config.num_classes)
        L = torch.cat([labs, labs], dim=0).to(torch.float32)       # shape [2N, 2]: one-hot encoded vector of labels, repeated twice
        mask = L @ L.T                          # shape [2N, 2N]: mask where mask(i,j) = 1 if z_i(i) has same label as z_j(j) and 0 otherwise
        mask = mask * (1 - id_mat)              # puts 0 on the diagonal
        card_P_i = mask.sum(dim=1)

        log_sim = func.log_softmax(sim_mat, dim=1)
        positive_mat = (log_sim * mask) / card_P_i
        loss = -positive_mat.sum() / (2*N)

        correct_pairs = torch.arange(N, device=z_i.device).long()
        sim_zij = sim_mat[:N,N:]                # the upper right matrix contains z_i @ z_j.T

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss




class GeneralizedSupervisedNTXenLoss(nn.Module):
    def __init__(self, config, kernel='rbf', temperature=0.1, return_logits=False, sigma=1.0):
        """
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma' which corresponds to gamma=1/(2*sigma**2)
        :param temperature:
        :param return_logits:
        """

        # sigma = prior over the label's range
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        self.config = config
        if self.kernel == 'rbf':
            self.kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        if self.kernel == 'laplacian':
            self.kernel = lambda y1, y2: laplacian_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        elif self.kernel == 'discrete':
            self.kernel = lambda y1, y2: vec_weights(euclidean_distances(y1,y2))
        elif self.kernel == 'distance':
            self.kernel = lambda y1, y2: vec(euclidean_distances(y1,y2))
        else:
            assert hasattr(self.kernel, '__call__'), 'kernel must be a callable'
        self.temperature = temperature
        self.return_logits = return_logits
        self.INF = 1e8

    def forward(self, z_i, z_j, labels, z_pos):
        N = len(z_i)
        num_classes = self.config.num_classes
        assert N == len(labels), "Unexpected labels length: %i"%len(labels)
        #print(z_i[0])
        #print(z_j[0])
        z_i = func.normalize(z_i, p=2, dim=-1)     # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)     # dim [N, D]
        sim_zii = (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)


        # positional weights (second mask array with continuous numbers)
        all_labels = torch.tensor(z_pos).view(N, -1).repeat(2, 1).detach().cpu().numpy()  # [2N, *]
        weights = self.kernel(all_labels, all_labels)                                     # [2N, 2N]
        weights = torch.from_numpy(weights * (1 - np.eye(2*N))).to(z_i.device)            # puts 0 on the diagonal
        final_weights = weights / weights.sum(dim=1) ## if dufumier loss: commment if wsp

        ## uncomment the following code for our wsp loss:

        # ground truth labels (first mask array with binary numbers)
        # labs = func.one_hot(torch.tensor(labels, device=z_i.device).long(), num_classes=num_classes)
        # L = torch.cat([labs, labs], dim=0).to(
        #    torch.float32)                  # shape [2N, 2]: one-hot encoded vector of labels, repeated twice
        # mask = L @ L.T                      # shape [2N, 2N]: mask where mask(i,j) = 1 if z_i(i) has same label as z_j(j) and 0 otherwise
        # mask = mask * (1 - torch.eye(2*N)).to(z_i.device)    # puts 0 on the diagonal

        # final array of mask, dot product between dirac and exponential + normalization
        #final_weights = weights * mask #) / mask.sum(dim=1) #just weights*mask normally
        #final_weights /= final_weights.sum(dim=1)

        # compute the loss
        sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1), torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0) # [2N, 2N]
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = -1./N * (log_sim_Z * final_weights).sum()

        correct_pairs = torch.arange(N, device=z_i.device).long()

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss




class NTXenLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations, arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        #print(z_i[0])
        #print(z_j[0])
        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii = (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)

        if self.return_logits:
            return loss_i + loss_j, sim_zij, correct_pairs

        return loss_i + loss_j

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)



class SupSimLoss(nn.Module):
    def __init__(self,
                 loss_supcon,
                 loss_simclr,
                 alpha=0.5):
        """
        weighted sum loss of supcon and simclr
        """

        # sigma = prior over the label's range
        super().__init__()
        self.loss_supcon = loss_supcon
        self.loss_simclr = loss_simclr
        self.alpha = alpha

    def forward(self, z_i, z_j, labels):
        loss_a, _, _ = self.loss_supcon(z_i, z_j, labels)
        loss_b, logits, target = self.loss_simclr(z_i,z_j)

        return self.alpha * loss_a + (1-self.alpha) * loss_b, logits, target



class ySimLoss(nn.Module):
    def __init__(self,
                 loss_supervised,
                 loss_simclr,
                 alpha=0.5):
        """
        weighted sum loss of cross entropy and simclr
        """

        # sigma = prior over the label's range
        super().__init__()
        self.loss_supervised = loss_supervised
        self.loss_simclr = loss_simclr
        self.alpha = alpha
        print(self.alpha)

    def forward(self, z_i, z_j, y, labels):
        loss_a = self.loss_supervised(y, labels)
        loss_b, _, _ = self.loss_simclr(z_i,z_j)

        return self.alpha * loss_a + (1-self.alpha) * loss_b


class SimSIAMLoss(nn.Module):
    """
    SimSIAM Loss
    """

    def __init__(self, ):
        super().__init__()

    def forward(self, a, b):

        b = b.detach()
        h = func.normalize(a, p=2, dim=-1) # dim [N, D]
        z = func.normalize(b, p=2, dim=-1) # dim [N, D]

        return -1 * ( h * z ).sum(-1).mean()



class BYOLLoss(nn.Module):
    """
    Bootstrap Your Own Latent
    Refer for instance to:
    Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec,
    Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires,
    Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot,
    Koray Kavukcuoglu, Rémi Munos, Michal Valko
    A Simple Framework for Contrastive Learning of Visual Representations, arXiv 2020
    """

    def __init__(self, ):
        super().__init__()

    def forward(self, a, b):

        h = func.normalize(a, p=2, dim=-1) # dim [N, D]
        z = func.normalize(b, p=2, dim=-1) # dim [N, D]

        return 2 - 2 * ( h * z ).sum(-1).mean()



class BarlowTwinsLoss(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.config = config

    def forward(self, a, b):
        # empirical cross-correlation matrix
        c = a.T @ b
        # sum the cross-correlation matrix between all gpus
        c.div_(self.config.batch_size)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # In order to match the code that was used to develop Barlow Twins,
        # the authors included an additional parameter, --scale-loss,
        # that multiplies the loss by a constant factor.
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.config.scale_loss)

        n, m = c.shape
        x = c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        off_diag = x.pow_(2).sum().mul(self.config.scale_loss)

        loss = on_diag + self.config.lambd * off_diag
        return loss



class DINOLoss(nn.Module):
    def __init__(self, config, ncrops=5,
                 warmup_teacher_temp=0.04,
                 final_teacher_temp=0.07,
                 warmup_teacher_temp_epochs=10,
                 nepochs=300,
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.out_dim = config.output_dim
        self.register_buffer("center", torch.zeros(1, self.out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, final_teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * final_teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_output: (B*ncrops, out_dim)
        teacher_output: (B*2, out_dim)
        """
        student_out = student_output/self.student_temp
        student_out = student_out.chunk(self.ncrops) # global views + local views

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = func.softmax((teacher_output-self.center)/temp, dim=-1)
        teacher_out = teacher_out.chunk(2) # global views

        total_loss = n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: # skip cases where student and teacher operate on the same view
                    continue
                loss = reduce(-q*func.log_softmax(student_out[v], dim=-1), 'b o -> b', 'sum')
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = reduce(teacher_output, 'b o -> 1 o', 'mean')

        self.center = self.center * self.center_momentum + \
                      batch_center * (1 - self.center_momentum)