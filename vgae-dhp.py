import os.path as osp
import argparse
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE, APPNP, GCNConv
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops)

EPS = 1e-15
MAX_LOGSTD = 10
MAX_TEMP = 2.5
MIN_TEMP = 0.5
COS_TEMP = 10.0
BETA = 1.0
MAX_BETA = 1.0
MIN_BETA = 0

beta = MIN_BETA 

decay_weight = np.log(MAX_TEMP/MIN_TEMP)
decay_step = 100.0
beta_decay_step = 100.0
patience = 50

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='SVGNAE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--training_rate', type=float, default=0.8) 
args = parser.parse_args()

dev = torch.device('cuda')
torch.cuda.empty_cache()


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(path, args.dataset, 'public')
if args.dataset in ['cs', 'physics']:
    dataset = Coauthor(path, args.dataset, 'public')
if args.dataset in ['computers', 'photo']:
    dataset = Amazon(path, args.dataset, 'public')

data = dataset[0]
data = T.NormalizeFeatures()(data)
cos = nn.CosineSimilarity()

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index,not_prop=0):
        if args.model == 'GNAE':
            x = self.linear1(x)
            x = F.normalize(x,p=2,dim=1)  * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x

        if args.model == 'VGNAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)

            x = self.linear2(x)
            x[:,:2] = F.normalize(x[:,:2],p=2,dim=1) * args.scaling_factor
            x[:,2:] = F.normalize(x[:,2:],p=2,dim=1) * args.scaling_factor
            x[:,2:] = self.propagate(x[:,2:], edge_index)
            return x, x_
        
        if args.model == 'SVGNAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)
            
            x = self.linear2(x)
            a = F.normalize(x[:,:2],p=2,dim=1) * args.scaling_factor/4
            b = F.normalize(x[:,2:],p=2,dim=1) * args.scaling_factor
            b = self.propagate(b, edge_index)
            c = torch.cat((a, b), dim=1)
            return c, x_

    
class StochasticWeight(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StochasticWeight, self).__init__()
        
        self.linear1 = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        x =  self.linear1(x)#, self.linear2(x)
        return torch.tanh(x)*2.5


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    
class SVGAE(torch.nn.Module):
    def __init__(self, encoder1, decoder):
        super().__init__()
        self.encoder1 = encoder1
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        SVGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder1)
        reset(self.decoder)
  
    def encode1(self, *args, **kwargs):
        """"""
        self.__mu1__, self.__logstd1__ = self.encoder1(*args, **kwargs)
        self.__logstd1__ = self.__logstd1__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu1__, self.__logstd1__)
        return z

    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
        
    
    def test(self, z1, temp, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z1.new_ones(pos_edge_index.size(1))
        neg_y = z1.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        
        pos_pred = self.decoder(z1, temp, pos_edge_index, sigmoid=True, training=False)
        neg_pred = self.decoder(z1, temp, neg_edge_index, sigmoid=True, training=False)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
  
    def recon_loss(self, z1, temp, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """
        decode_p, prob_edge = self.decoder(z1, temp, pos_edge_index, sigmoid=True, training=True)
        pos_loss = -torch.log(decode_p + EPS).mean()

        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z1.size(0))
        
        decode_n = self.decoder(z1, temp, neg_edge_index, sigmoid=True, training=True, pos=False)
        neg_loss = -torch.log(1 -decode_n + EPS).mean() 

        return (pos_loss + neg_loss), prob_edge #+ a
    
    def kl_loss1(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu1__ if mu is None else mu
        logstd = self.__logstd1__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

class InnerProductDecoder2(torch.nn.Module):
    def __init__(self):
        super().__init__()        
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    
    def forward(self, z1,  temp, edge_index, sigmoid=True, training=True, pos=True):
        if training: 
            if pos: 
                z11 = z1.detach().clone()
                
                vf = (z11[edge_index[0],2:] * z11[edge_index[1],2:]).sum(dim=1) #+ bias
                
                la = torch.cat(  (torch.unsqueeze(vf, 1), torch.zeros(torch.unsqueeze(vf, 1).shape).to(dev)   ),1)

                la_ra = la
                a = F.gumbel_softmax((la_ra), tau=temp, hard=True)[:,:1]
                value_feature = (z1[edge_index[0],2:] * z1[edge_index[1],2:]).sum(dim=1)
                value_network = z1[edge_index[0],[0]]  + z1[edge_index[1],[0]]

                original_flag = torch.flatten(a)
                return original_flag*torch.sigmoid(value_feature) + (1-original_flag)*torch.sigmoid(value_network),  a if sigmoid else value
            
            else:
                z11 = z1.detach().clone()
                #bbias = bias.detach().clone()
                vf = (z11[edge_index[0],2:] * z11[edge_index[1],2:]).sum(dim=1)#+ bbias
                la = torch.cat(  (torch.unsqueeze(vf, 1), torch.zeros(torch.unsqueeze(vf, 1).shape).to(dev)   ),1)
                la_ra = la
                a = F.gumbel_softmax((la_ra), tau=temp, hard=True)[:,:1]
 
                value_feature = (z1[edge_index[0],2:] * z1[edge_index[1],2:]).sum(dim=1)
                value_network = z1[edge_index[0],[0]]  + z1[edge_index[1],[0]] 

                original_flag = torch.flatten(a)
                return original_flag*torch.sigmoid(value_feature) + (1-original_flag)*torch.sigmoid(value_network) if sigmoid else value
                    
        else:
            z11 = z1.detach().clone()
            vf = (z11[edge_index[0],2:] * z11[edge_index[1],2:]).sum(dim=1)
            la = torch.cat(  (torch.unsqueeze(vf, 1), torch.zeros(torch.unsqueeze(vf, 1).shape).to(dev)   ),1)
            la_ra = la
            a = F.softmax((la_ra), dim=1)[:,:1]
            value_feature = (z1[edge_index[0],2:] * z1[edge_index[1],2:]).sum(dim=1)
            value_network = z1[edge_index[0],[0]]  + z1[edge_index[1],[0]] 
            original_flag = torch.flatten(a)
            return original_flag*torch.sigmoid(value_feature) + (1-original_flag)*torch.sigmoid(value_network) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
    

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

channels = args.channels
train_rate = args.training_rate
val_ratio = (1-args.training_rate) / 3
test_ratio = (1-args.training_rate) / 3 * 2
data = train_test_split_edges(data)
all_link = data.train_pos_edge_index
in_channels_w = dataset.num_features 
out_channels_w = 1   

N = int(data.x.size()[0])
if args.model == 'GNAE':   
    model = GAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)
if args.model == 'VGNAE':
    model = VGAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)
if args.model == 'SVGNAE':
    model = SVGAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index), InnerProductDecoder2()).to(dev)

data.train_mask = data.val_mask = data.test_mask = data.y = None
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

network_input = torch.eye(x.shape[0]).to(dev)

l1 = train_pos_edge_index
l2 = train_pos_edge_index

def train(epoch):
    temp = np.maximum(MAX_TEMP*np.exp(-(epoch-1)/decay_step*decay_weight), MIN_TEMP)
    
    global l1
    global l2 
    
    beta = BETA
    model.train()
    optimizer.zero_grad()
    z1 = model.encode1(x, l1)

    loss,prob_edge = model.recon_loss(z1, temp, train_pos_edge_index)

    if args.model in ['VGAE']:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    if args.model in ['SVGNAE']:
        loss = loss + (1.0 / data.num_nodes) * (model.kl_loss1())*beta
    if args.model in ['SGNAE']:
        loss = loss
    loss.backward()
    optimizer.step()
    return loss, [l1,l2]

def test(pos_edge_index, neg_edge_index, selected_list, plot_his=0):
    model.eval()
    with torch.no_grad():
        z1 = model.encode1(x, selected_list[0])    
    return model.test(z1, 1.0, pos_edge_index, neg_edge_index)


early_stopping = EarlyStopping(patience = patience, verbose = True)


for epoch in range(1,args.epochs + 1):
    loss, selected_list  = train(epoch)
    #loss  = train(epoch)
    loss = float(loss)
    
    
    ans = np.zeros( (x.shape[0],2))
    
    with torch.no_grad():
        val_pos, val_neg = data.val_pos_edge_index, data.val_neg_edge_index
        auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index, selected_list)
        if epoch>150:
            early_stopping(-auc, model)
            if early_stopping.early_stop:
                break
        
        
model.load_state_dict(torch.load('checkpoint.pt'))


val_pos, val_neg = data.val_pos_edge_index, data.val_neg_edge_index
auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index,selected_list)


test_pos, test_neg = data.test_pos_edge_index, data.test_neg_edge_index
auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index,selected_list)

print('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f} AP: {:.4f}'.format(epoch, loss, auc, ap))