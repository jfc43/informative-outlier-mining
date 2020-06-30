import torch
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans

import numpy as np
from sklearn.decomposition import PCA


class Metric(nn.Module):
    '''
        Abstract class that defines the concept of a metric. It is needed
        to define mixture models with different metrics.
        In the paper we use the PCAMetric
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, y, dim=None):
        pass

    def __add__(self, other):
        return SumMetric(self, other)

    def __rmul__(self, scalar):
        return ScaleMetric(scalar, self)


class SumMetric(Metric):
    def __init__(self, metric1, metric2):
        super().__init__()
        self.metric1 = metric1
        self.metric2 = metric2

    def forward(self, x, y, dim=None):
        return self.metric1(x, y, dim=dim) + self.metric2(x, y, dim=dim)


class ScaleMetric(Metric):
    def __init__(self, metric1, factor):
        super().__init__()
        self.metric1 = metric1
        self.factor = factor

    def forward(self, x, y, dim=None):
        return self.factor * self.metric1(x, y, dim=dim)


class LpMetric(Metric):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
        self.norm_const = 0.

    def forward(self, x, y, dim=None):
        return (x-y).norm(p=self.p, dim=dim)


class PerceptualMetric(Metric):
    def __init__(self, model, p=2, latent_dim=122880, indices=None):
        super().__init__()
        self.model = model
        self.p = p
        self.norm_const = 0.

        self.latent_dim = latent_dim
        reduced_latent_dim = int(0.01*latent_dim)

        if indices is None:
            self.indices = sorted(np.random.choice(latent_dim, size=reduced_latent_dim, replace=False))
        else:
            self.indices = indices

    def forward(self, x, y, dim=None):
        return (self.model(x)[:,self.indices][None,:,:]
                -self.model(y)[:,self.indices][:,None,:]).norm(p=self.p, dim=dim)


class PerceptualPCA(Metric):
    def __init__(self, model, pca, indices=None):
        super().__init__()
        self.model = model

        self.pca = pca

        if indices is None:
            self.indices = sorted(np.random.choice(latent_dim, size=reduced_latent_dim, replace=False))
        else:
            self.indices = indices


    def forward(self, x, y, dim=None):
        return self.pca(self.model(x)[:,self.indices][None,:,:],
                        self.model(y)[:,self.indices][:,None,:], dim=dim)


class PCAMetric(Metric):
    def __init__(self, X, p=2, min_sv_factor=100., covar=None):
        super().__init__()
        self.p = p

        if covar is None:
            X = np.array(X)
            pca = PCA()
            pca.fit(X)

            self.comp_vecs = nn.Parameter(torch.tensor(pca.components_), requires_grad=False)
            self.singular_values = torch.tensor(pca.singular_values_)
        else:
            singular_values, comp_vecs = np.linalg.eig(covar)

            self.comp_vecs = nn.Parameter(torch.tensor(comp_vecs, dtype=torch.float), requires_grad=False)
            self.singular_values = torch.tensor(singular_values, dtype=torch.float)

        self.min_sv = self.singular_values[0] / min_sv_factor
        self.singular_values[self.singular_values<self.min_sv] = self.min_sv
        self.singular_values = nn.Parameter(self.singular_values, requires_grad=False)
        self.singular_values_sqrt = nn.Parameter(self.singular_values.sqrt(), requires_grad=False)

        self.norm_const = self.singular_values.log().sum()

    def forward(self, x, y, dim=None):
        rotated_dist = torch.einsum("ijk,lk->ijl", (x-y, self.comp_vecs))
        rescaled_dist = rotated_dist / self.singular_values_sqrt[None,None,:]
        return rescaled_dist.norm(dim=2, p=self.p)


class MyPCA():
    '''
        A helper class that is used for adversarial attacks in a PCAMetric
    '''
    def __init__(self, comp_vecs, singular_values, shape):
        self.comp_vecs = comp_vecs
        self.comp_vecs_inverse = self.comp_vecs.inverse()
        self.singular_values = singular_values
        self.singular_values_sqrt = singular_values.sqrt()
        self.shape = tuple(shape)
        self.D = torch.tensor(shape).prod().item()

    def inv_trans(self, x):
        x = ( (x * self.singular_values_sqrt[None,:] ) @ self.comp_vecs_inverse )
        return x.view(tuple([x.shape[0]]) + self.shape)

    def trans(self, x):
        x = x.view(-1, self.D)
        return ( (x@self.comp_vecs) / self.singular_values_sqrt[None,:] )


class MixtureModel(nn.Module):

    def __init__(self, K, D, mu=None, logvar=None, alpha=None, metric=LpMetric()):
        """
        Initializes means, variances and weights randomly
        :param K: number of centroids
        :param D: number of features
        :param mu: centers of centroids (K,D)
        :param logvar: logarithm of the variances of the centroids (K)
        :param alpha: logarithm of the weights of the centroids (K)
        """
        super().__init__()

        self.D = D
        self.K = K
        self.metric = metric
        if mu is None:
            self.mu = nn.Parameter(torch.rand(K, D))
        else:
            self.mu = nn.Parameter(mu)

        if logvar is None:
            self.logvar = nn.Parameter(torch.rand(K))
        else:
            self.logvar = nn.Parameter(logvar)

        if alpha is None:
            self.alpha = nn.Parameter(torch.empty(K).fill_(1. / K).log())
        else:
            self.alpha = nn.Parameter(alpha)

        self.logvarbound = 0

    def forward(self, x):
        pass

    def calculate_bound(self, L):
        pass

    def get_posteriors(self, X):
        log_like = self.forward(X)
        log_post = log_like - torch.logsumexp(log_like, dim=0, keepdim=True)
        return log_post

    def EM_step(self, X):
        log_post = self.get_posteriors(X)

        log_Nk = torch.logsumexp(log_post, 1)

        self.mu.data = ((log_post[:,:,None] - log_Nk[:,None,None]).exp() * X[None,:,:]).sum(1)
        temp = log_post + (((X[None,:,:]-self.mu[:,None,:])**2).sum(dim=-1)/self.D).log()
        self.logvar.data = (- log_Nk
                            + torch.logsumexp(temp, dim=1, keepdim=False))
        self.alpha = ( log_Nk - torch.logsumexp(log_Nk, 0) ).clone().detach()

    def find_solution(self, X, initialize=True, iterate=True, use_kmeans=True, verbose=False):
        assert X.device==self.mu.device, 'Data stored on ' + str(X.device) + ' but model on ' + str(self.mu.device)

        with torch.no_grad():
            if initialize:
                m = X.size(0)

                if (use_kmeans):
                    kmeans = KMeans(n_clusters=self.K, random_state=0, max_iter=300).fit(X.cpu())
                    self.mu.data = torch.tensor(kmeans.cluster_centers_,
                                                dtype=torch.float,
                                                device=self.mu.device)
                else:
                    idxs = torch.from_numpy(np.random.choice(m, self.K, replace=False)).long()
                    self.mu.data = X[idxs]

                index = (X[:,None,:]-self.mu.clone().detach()[None,:,:]).norm(dim=2).min(dim=1)[1]
                for i in range(self.K):
                    assert (index==i).sum()>0, 'Empty cluster'
                    self.alpha.data[i] = ((index==i).float().sum() / (3*self.K)).log()
                    temp = (X[index==i,:] - self.mu.data[i,:]).norm(dim=1).mean()
                    if temp < 0.00001:
                        temp = torch.tensor(1.)
                    self.logvar.data[i] = temp.log() * 2

                self.alpha.data = self.alpha.data.exp()
                self.alpha.data /= self.alpha.data.sum()
                self.alpha.data = self.alpha.data.log()

                self.logvarbound = (X.var() / m).log()

            if iterate:
                for i in range(50):
                    mu_prev = self.mu.clone().detach()
                    logvar_prev = self.logvar.clone().detach()
                    alpha_prev = self.alpha.clone().detach()
                    self.EM_step(X)

                    self.logvar.data[self.logvar < self.logvarbound] = self.logvarbound

                    delta = torch.stack( ((mu_prev-self.mu).abs().max(),
                                (logvar_prev-self.logvar).abs().max(),
                                (alpha_prev-self.alpha).abs().max()) ).max()
                    if verbose:
                        print('Iteration: '+ str(i)+'\t delta: '+str(delta.item()))
                        print((mu_prev-self.mu).abs().max())
                        print((logvar_prev-self.logvar).abs().max())
                        print((alpha_prev-self.alpha).abs().max())
                    if delta<10e-6:
                        break


class GMM(MixtureModel):
    def __init__(self, K, D, mu=None, logvar=None, alpha=None, metric=LpMetric()):
        """
        Initializes means, variances and weights randomly
        :param K: number of centroids
        :param D: number of features
        """
        super().__init__(K, D, mu, logvar, alpha, metric)
        self.norm_const = .5 * torch.tensor(2*np.pi).log() * self.D + .5 * metric.norm_const
        self.norm_const = nn.Parameter(self.norm_const, requires_grad=False)

    def forward(self, X):
        """
        Compute the likelihood of each data point under each gaussians.
        :param X: design matrix (examples, features) (N,D)
        :return likelihoods: (K, examples) (K, N)
        """
        a = self.metric(X[None,:,:], self.mu[:,None,:], dim=2)**2
        b = self.logvar[:,None].exp()

        return (self.alpha[:,None] - .5*self.D*self.logvar[:,None]
                - .5*( a/b ) - self.norm_const)

    def calculate_bound(self, L):
        var = self.logvar[:,None].exp()
        bound = (self.alpha[:,None] - .5*self.D*self.logvar[:,None]
                - .5* ( L**2/(2*var) ) - self.norm_const )
        return torch.logsumexp(bound.squeeze(),dim=0)


class DoublyRobustModel(nn.Module):
    '''
        The CCU model 
        Both in- and out-mixture models have to be passed as arguments
    '''
    def __init__(self, base_model, mixture_model_in, mixture_model_out, loglam, dim=3072, classes=10):
        super().__init__()
        self.base_model = base_model

        self.dim = dim
        self.mm = mixture_model_in
        self.mm_out = mixture_model_out

        self.loglam = nn.Parameter(torch.tensor(loglam, dtype=torch.float), requires_grad=False)
        self.log_K = nn.Parameter(-torch.tensor(classes, dtype=torch.float).log(), requires_grad=False)


    def forward(self, x):
        batch_size = x.shape[0]

        likelihood_per_peak_in = self.mm(x.view(batch_size, self.dim))
        like_in = torch.logsumexp(likelihood_per_peak_in, dim=0)

        likelihood_per_peak_out = self.mm_out(x.view(batch_size, self.dim))
        like_out = torch.logsumexp(likelihood_per_peak_out, dim=0)

        x = self.base_model(x)
        x = F.log_softmax(x, dim=1)

        a1 = torch.stack( (x + like_in[:,None], 0*x + (self.loglam + self.log_K) + like_out[:,None] ), 0)
        b1 = torch.logsumexp(a1, 0).squeeze()

        a2 = torch.stack( (like_in , (self.loglam) + like_out), 0)
        b2 = torch.logsumexp(a2, 0).squeeze()[:,None]

        return b1-b2
