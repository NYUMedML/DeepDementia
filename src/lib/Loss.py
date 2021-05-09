import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

def get_loss_criterion(loss_config,type,num_examp):
    if type == 'NCE':
        loss_criterion = NCECriterion(loss_config['NCE']['nLem'])
    elif type == 'CrossEntropyLoss':
        weight = loss_config['training_parameters']['weight']
        weight = [convert_to_float(i) for i in weight]
        loss_criterion = cross_entropy_loss(num_examp=num_examp, 
            num_classes=loss_config['model']['n_label'], 
            weight = weight)
    elif type == 'semi_loss':
        weight = loss_config['training_parameters']['weight']
        weight = [convert_to_float(i) for i in weight]
        loss_criterion = semi_loss(num_examp=num_examp, 
            num_classes=loss_config['model']['n_label'], 
            weight = weight)

    elif type == 'semi_loss_mse':
        weight = loss_config['training_parameters']['weight']
        weight = [convert_to_float(i) for i in weight]
        loss_criterion = semi_loss2(num_examp=num_examp, 
            num_classes=loss_config['model']['n_label'], 
            weight = weight)
    elif type == 'semi_loss_contrastive':
        weight = loss_config['training_parameters']['weight']
        weight = [convert_to_float(i) for i in weight]
        loss_criterion = semi_contrastive_loss(
            device = torch.device("cuda:0"), 
            batch_size = loss_config['data']['batch_size'], 
            temperature = 0.5, 
            use_cosine_similarity = True, 
            weight = weight)

    elif type == 'combined':
        loss_criterion = CombinedLoss()
    elif type == 'mse':
        loss_criterion = nn.MSELoss()
    elif type == 'LMCL':
        loss_criterion = LMCL_loss(num_classes=loss_config['model']['n_label'], feat_dim = loss_config['model']['nhid'])
    elif type == 'comp':
        loss_criterion = complementary_CE(ignore_index=-100)
    elif type == 'elr':
        loss_criterion = elr_loss(num_examp=num_examp, num_classes=loss_config['model']['n_label'])
    else:
        raise NotImplementedError
    return loss_criterion

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class complementary_CE(nn.Module):
    def __init__(self,ignore_index=-100,weight=None):
        super(complementary_CE, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self,pred_score,target_score):
        #log_softmax = ((-F.softmax(pred_score,dim=1)).exp_()+1).log_() 
        #return F.nll_loss(log_softmax,target_score,weight=self.weight,ignore_index=self.ignore_index) - F.softmax(pred_score, dim=1) * F.log_softmax(pred_score, dim=1).sum()
        return - F.softmax(pred_score, dim=1) * F.log_softmax(pred_score, dim=1).sum()

class LogitBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(LogitBinaryCrossEntropy, self).__init__()

    def forward(self, pred_score, target_score, weights=None):
        loss = F.binary_cross_entropy_with_logits(pred_score,
                                                  target_score,
                                                  size_average=True)
        loss = loss * target_score.size(1)
        return loss


def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)

    return torch.sum(res, dim=1, keepdim=True)


class weighted_softmax_loss(nn.Module):
    def __init__(self):
        super(weighted_softmax_loss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = loss * tar_sum
        loss = torch.sum(loss) / loss.size(0)
        return loss


class SoftmaxKlDivLoss(nn.Module):
    def __init__(self):
        super(SoftmaxKlDivLoss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        return loss


class wrong_loss(nn.Module):
    def __init__(self):
        super(wrong_loss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = F.kl_div(res, tar, size_average=True)
        loss *= target_score.size(1)
        return loss


class semi_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, alpha=0.3, weight = [1.0/561,1.0/835,1.0/522]):
        super(semi_loss, self).__init__()
        self.num_classes = num_classes
        self.weight = torch.FloatTensor(weight).cuda()#torch.FloatTensor([1.0/561,1.0/835,1.0/522]).cuda()#[1.0/392,1.0/530,1.0/338]).cuda()


    def forward(self, output, output_u_w, output_u, target):

        ce_loss = F.cross_entropy(output, target, ignore_index=-100, weight = self.weight)

        if output_u_w is not None:

            pseudo_label = torch.softmax(output_u_w.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(0.7).float()
            Lu = (F.cross_entropy(output_u, targets_u,
                                  reduction='none') * mask).mean()
        else:
            Lu = 0
        final_loss = ce_loss + Lu
        return  final_loss


class semi_loss2(nn.Module):
    def __init__(self, num_examp, num_classes=10, alpha=0.3, weight = [1.0/561,1.0/835,1.0/522]):
        super(semi_loss2, self).__init__()
        self.num_classes = num_classes
        self.weight = torch.FloatTensor(weight).cuda()#torch.FloatTensor([1.0/561,1.0/835,1.0/522]).cuda()#[1.0/392,1.0/530,1.0/338]).cuda()


    def forward(self, current, output, output_u_w, output_u, target):

        ce_loss = F.cross_entropy(output, target, ignore_index=-100, weight = self.weight)

        if output_u_w is not None:

            pseudo_label = torch.softmax(output_u_w.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            target_u = torch.zeros(len(targets_u), self.num_classes).cuda().scatter_(1, targets_u.view(-1,1), 1)
            Lu = F.mse_loss(target_u,torch.softmax(output_u,dim=-1) )
        else:
            Lu = 0
        final_loss = ce_loss + sigmoid_rampup(current, 10)*Lu
        return  final_loss


class cross_entropy_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, alpha=0.3, weight = [1.0/561,1.0/835,1.0/522]):
        super(cross_entropy_loss, self).__init__()
        self.num_classes = num_classes
        self.weight = torch.FloatTensor(weight).cuda()#torch.FloatTensor([1.0/561,1.0/835,1.0/522]).cuda()#[1.0/392,1.0/530,1.0/338]).cuda()


    def forward(self, output, target):

        ce_loss = F.cross_entropy(output, target, ignore_index=-100, weight = self.weight)
        final_loss = ce_loss 
        return  final_loss

class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, alpha=0.3):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.pred_hist = (torch.zeros(num_examp, self.num_classes)*1.0/self.num_classes).cuda()
        self.q = torch.ones(self.num_classes).cuda() / self.num_classes if self.USE_CUDA else torch.ones(self.num_classes) / self.num_classes
        self.alpha = alpha
        

    def forward(self, index, output, target):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        pred_ = y_pred.data.detach()
        self.pred_hist[index] = (1-self.alpha) * self.pred_hist[index] + self.alpha *  ((pred_)/(pred_).sum(dim=1,keepdim=True))
        self.q = self.pred_hist[index]
        ce_loss = F.cross_entropy(output, target, ignore_index=-100, weight = torch.FloatTensor([1.0/1223,1.0/2444,1.0/1687]).cuda())
        reg = ((1-(self.q * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss 
        return  final_loss


class semi_contrastive_loss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, weight):
        super(semi_contrastive_loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)

        self.weight = torch.FloatTensor(weight).cuda()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()#.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, current, output, zis, zjs, target):
        ce_loss = F.cross_entropy(output, target, ignore_index=-100, weight = self.weight, reduction="sum")


        if zis is not None:

            representations = torch.cat([zjs, zis], dim=0)

            similarity_matrix = self.similarity_function(representations, representations)

            # filter out the scores from the positive samples
            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)
            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

            negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

            logits = torch.cat((positives, negatives), dim=1)
            logits /= self.temperature

            labels = torch.zeros(2 * self.batch_size).cuda().long()
            loss = self.criterion(logits, labels)
        else:
            loss = 0

        return ce_loss + loss / 2
