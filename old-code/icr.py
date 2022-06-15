import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import OPTICS, DBSCAN, Birch, AgglomerativeClustering
from scipy.special import expit
import ot
import math

from collections import UserDict

def knn(X, k, Y=None):
    if Y is None:
        Y = X
        k += 1
    nn=NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    return nn.kneighbors(Y)[0][:,k-1]

def knn2(X, k, Y=None, bs=32):
    if Y is None:
        Y = X
        k += 1
    X = X if type(X) == torch.Tensor else torch.Tensor(X)
    Y = Y if type(Y) == torch.Tensor else torch.Tensor(Y)
    N = Y.size(0)
    n_batches = int(math.ceil(N/bs))
    dists = torch.zeros(N)
    if torch.cuda.is_available():
        X = X.to('cuda')
        Y = Y.to('cuda')
        dists=dists.to('cuda')
    for i in range(n_batches):
        j_min = i*bs
        j_max = min(N, (i+1)*bs)
        all_dists_i = (Y[j_min:j_max].unsqueeze(1) - X.unsqueeze(0)).norm(dim=-1)
        topk_i = all_dists_i.topk(k, dim=1, largest=False)[0][:,k-1]
        dists[j_min:j_max] = topk_i
    
    return dists.cpu().numpy()
        
    

def kl_div(X, Y, k, remove_zeros=True):
    if remove_zeros:
        X = np.unique(X, axis=0)
        Y = np.unique(Y, axis=0)

    n = X.shape[0]
    m = Y.shape[0]
    d = X.shape[1]

    nu = knn2(X=Y, Y=X, k=k)
    eps = knn2(X=X, k=k)

    if remove_zeros:
        if len(np.nonzero(nu==0)[0]) > 0:
            X_valid=np.nonzero(nu)
            nu = nu[X_valid]
            eps = eps[X_valid]
            n = nu.shape[0]

    return d/n * np.log(nu/eps).sum() + np.log(m/(n-1))

def avg_nn_dist(X, Y, knn_fct=knn2):
    return knn_fct(X=Y, Y=X, k=1).sum() / X.shape[0]

def henderson_distance(vi, vj):
    return np.dot(expit(-vi), np.log(expit(-vj)))

def henderson_wasserstein(X, Y):
    costs = ot.dist(X, Y, lambda x,y:-1*henderson_distance(x,y))
    return ot.emd2([],[],costs)


class ICRDict(UserDict):
    @classmethod
    def from_dict(cls, vec_dict, seq_dict=None):
        data = {}
        for k,v in vec_dict.items():
            if seq_dict is not None:
                data[k] = ICRepresentation(k, v, seqs=seq_dict[k])
            else:
                data[k] = ICRepresentation(k, v)
        return cls(data)

    def __init__(self, data):
        super().__init__(data)

        #self._cluster_all()
        
    def cluster_all(self, *args, cluster_method=DBSCAN, apply_pca=True, n_components=2, **kwargs):
        for k in self.data.keys():
            self.data[k].cluster(*args, cluster_method=cluster_method, apply_pca=apply_pca, n_components=n_components, **kwargs)

    def pca(self, *keys, n_components=2):
        all_vecs = np.concatenate([self.data[k].vecs for k in keys], axis=0)
        pca = PCA(n_components=n_components).fit(all_vecs)
        return pca
    
    def plot(self, *keys, transform=None, apply_pca=True, use_clusters=True):
        assert not (transform is not None and apply_pca)
        if apply_pca:
            transform = self.pca(*keys).transform

        plt.figure()
        for k in keys:
            v = self.data[k]
            if v.clusters is None or not use_clusters:
                vecs = v.get_vecs(transform=transform)
                plt.scatter(vecs[:,0], vecs[:,1], label=k, s=1)
            else:
                for cluster in v.clusters.keys():
                    k_c = "%s_%s" % (k, cluster) if cluster != -1 else None
                    vecs = v.get_cluster_vecs(cluster, transform=transform)
                    plt.scatter(vecs[:,0], vecs[:,1], label=k_c, s=1)

        plt.legend()
        plt.show()

    def compare(self, k1, k2, dist_fct=avg_nn_dist, apply_pca=True, n_components=2, use_clusters=True, n_samples=-1, **kwargs):
        transform = self.pca(k1, k2).transform if apply_pca else None
        v1, v2 = self.data[k1], self.data[k2]
        if v1.clusters is None or not use_clusters:
            src_dict = {k1:v1.get_vecs(transform=transform, samples=n_samples)} 
        else: 
            src_dict = {"%s_%s" % (k1, k): v1.get_cluster(k, transform=transform) for k in v1.clusters.keys() if k != -1}
        if v2.clusters is None or not use_clusters:
            tgt_dict = {k2:v2.get_vecs(transform=transform, samples=n_samples)} 
        else: 
            tgt_dict = {"%s_%s" % (k2, k): v2.get_cluster(k, transform=transform) for k in v2.clusters.keys() if k != -1}
        dists={}
        for k_src, v_src in src_dict.items():
            for k_tgt, v_tgt in tgt_dict.items():
                dists["%s->%s"%(k_src, k_tgt)] = dist_fct(v_src, v_tgt, **kwargs)
                dists["%s->%s"%(k_tgt, k_src)] = dist_fct(v_tgt, v_src, **kwargs)
        return dists

    def score(self, k1, k2, dist_fct=avg_nn_dist, apply_pca=True, n_components=2, n_samples=-1, min_n=10, **kwargs):
        if k1 in self.data and k2 in self.data and self.data[k1].n > min_n and self.data[k2].n > min_n:
            transform = self.pca(k1, k2, n_components=n_components).transform if apply_pca else None
            v1, v2 = self.data[k1], self.data[k2]
            src_vecs = v1.get_vecs(transform=transform, samples=n_samples)
            tgt_vecs = v2.get_vecs(transform=transform, samples=n_samples)
            return -1 * dist_fct(src_vecs, tgt_vecs, **kwargs)
        else:
            return None


        
        
            
        




class ICRepresentation():
    def __init__(self, word, vecs, seqs=None):
        self.word = word
        self.vecs = vecs
        self.seqs=seqs
        self.n = vecs.shape[0]
        self.pca_vecs=None
        self.clusters=None

    def pca(self, n_components=2):
        self.pca_vecs = PCA(n_components=n_components).fit_transform(self.vecs)
    
    def cluster(self, *args, cluster_method=DBSCAN, apply_pca=True, n_components=2, **kwargs):
        self.clusters = {}
        X = PCA(n_components=n_components).fit_transform(self.vecs) if apply_pca else self.vecs
        labels = cluster_method(*args, **kwargs).fit_predict(X)
        cluster_labels = np.unique(labels)
        for label in cluster_labels:
            indices = (labels == label).nonzero()[0]
            self.clusters[label] = indices
    
    def get_cluster_vecs(self, key, transform=None):
        return self.get_vecs(indices=self.clusters[key], transform=transform)
    
    def get_cluster_seqs(self, key):
        return self.get_seqs(indices=self.clusters[key])
    
    def get_vecs(self, samples=-1, max_vecs=-1, indices=None, transform=None):
        if max_vecs > 0:
            if self.n > max_vecs:
                samples = max_vecs
        assert samples <= 0 or indices is None
        if indices is not None:
            vecs = self.vecs[indices]
        elif samples > 0:
            vecs = self.sample(samples)
        else:
            vecs = self.vecs
        return vecs if transform is None else transform(vecs)

    def get_seqs(self, indices=None):
        if indices is not None:
            return [self.seqs[i] for i in indices]
        else:
            return self.seqs

    def sample(self, k, replace=True):
        rng = np.random.default_rng()
        indices = rng.choice(self.n, k, replace=replace)
        return self.vecs[indices]