import time
import numpy as np
import pyro, pyro.distributions, pyro.optim, pyro.infer
from pyro.nn import PyroSample, PyroModule
from pyro.infer import autoguide
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import random, math, os, re, sys
import typing as ty
import dataclasses
import scipy.sparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def to_device(t): return torch.tensor(t).to(device)

@dataclasses.dataclass
class MOFAMatrices:
    Z: torch.Tensor
    Ws: dict[str, torch.Tensor]

def to_torch_sparse(matrix: scipy.sparse.csr_matrix, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    return torch.sparse_csr_tensor(matrix.indptr, matrix.indices, matrix.data, size=matrix.shape, device=device)

def get_mask(t: ty.Union[torch.Tensor, scipy.sparse.csr_matrix]) -> ty.Optional[torch.Tensor]:
    """Returns a mask with False where NaNs are presents. Return None, if there are no NaNs in the entire tensor"""
    if isinstance(t, scipy.sparse.csr_matrix):
        m = torch.isnan(to_torch_sparse(t))
        if torch.any(m.values()):
            return m.to_dense()
    else:
        m = torch.isnan(t)
        if torch.any(m):
            return m
    return None

#this does not work well for our batch sizes, just in case it comes in handy later:
# def torch_sparse_select_rows(tensor: torch.Tensor, ix: list[int] | torch.Tensor):
#     if not tensor.is_sparse_csr:
#         return tensor[ix]

#     ixt = torch.tensor(ix, device=tensor.device, dtype=torch.long)

#     crow = tensor.crow_indices()
#     rows_start = crow[ixt]
#     rows_end = crow[ixt+1]
#     rows_len = rows_end - rows_start
#     values = tensor.values()
#     new_values = torch.cat([
#         values[rows_start[i]:rows_end[i]] for i in range(len(ix))
#     ])
#     col_indices = tensor.col_indices()
#     new_cols = torch.cat([
#         col_indices[rows_start[i]:rows_end[i]] for i in range(len(ix))
#     ])

#     new_crow = torch.zeros(len(ix)+1, device=tensor.device, dtype=crow.dtype)
#     torch.cumsum(rows_len, dim=0, out=new_crow[1:])
#     return torch.sparse_csr_tensor(new_crow, new_cols, new_values, size=(len(ix), tensor.size(1)))
# print(rna_tensor.shape)
# print(rna.X.shape)
# print(rna.X[3].toarray())
# torch_sparse_select_rows(rna_tensor, [1]).to_dense()

def maybe_sparse_tensor(t):
    if isinstance(t, scipy.sparse.csr_matrix):
        return to_torch_sparse(t)
    if isinstance(t, torch.Tensor):
        return t
    return torch.tensor(t)

def replace_nans(x: ty.Union[torch.Tensor, scipy.sparse.csr_matrix], nan: float):
    if isinstance(x, scipy.sparse.csr_matrix):
        x.data = np.nan_to_num(x.data, nan=nan)
        return x
    else:
        return torch.nan_to_num(x, nan=nan)

class MOFA(PyroModule):
    def __init__(self, Ys: dict[str, torch.Tensor], K, batch_size=128, Guide: type[autoguide.AutoGuide] = autoguide.AutoNormal):
        """
        Args:
            Ys: Tensor for each modality (Samples x Features)
            K: Number of Latent Factors
        """
        super().__init__()
        pyro.clear_param_store()

        self.K = K  # number of factors 
        self.empirical_means = { m: torch.tensor(Y.mean(axis=0)) if isinstance(Y, scipy.sparse.csr_matrix) else Y.mean(dim=0)
                                 for m, Y in Ys.items() }
        self.empirical_stds = { m: 1 if isinstance(Y, scipy.sparse.csr_matrix) else torch.clamp(torch.std(Y, dim=0),1)
                                for m, Y in Ys.items() }

        self.obs_masks = { m: get_mask(Y)
                           for m, Y in Ys.items()}
        print(f"Observation masks: {self.obs_masks}")
        # a valid value for the NAs has to be defined even though these samples will be ignored later
        self.Ys = { m: replace_nans(Y, nan=0)
                    for m, Y in Ys.items()}  # data/observations
        
        # assert sample dim same in Ys
        num_samples = set(Y.shape[0] for Y in self.Ys.values())
        assert len(num_samples) == 1, f"Observation count does not match in all modalities: {{m: Y.shape[0] for m, Y in self.Ys.items()}}"
        self.num_samples = next(iter(num_samples))
        self.num_features = { m: Y.shape[1]
                              for m, Y in self.Ys.items() }
        
        self.batch_size = batch_size
        
        self.latent_factor_plate = pyro.plate("latent factors", self.K)
        self.Guide = Guide

    def get_sample_batch(self, m, indices):
        Y = self.Ys[m][indices]
        if isinstance(Y, scipy.sparse.csr_matrix):
            return to_torch_sparse(Y, device=device).to_dense()
        return Y.to(device)

    def model(self):
        """ Creates the model.
        
        The model needs to be created repeatedly (not sure why), in any case, it is important now, when using 
        `subsample_size` batch size to subsample the dataset differently in each train iteration
        """
        
        # needs to be shared, so returns the same indices in one train step
        sample_plate = pyro.plate("sample", self.num_samples, subsample_size=self.batch_size)
        # the plates get assigned a dim, depending on when in the plate hierarchy they are used. Unfortunately we want to use
        #   feature plates once outside and once inside other plate (sample resp. latent_factor plates, see below)
        #   we therefore need to create separate plates for each of those usages
        get_feature_plates = lambda dim: { m: pyro.plate(f"feature_{m}_{dim}", num_feats)
                                           for m, num_feats in self.num_features.items() }

        # W matrices for each modality
        Ws = {}
    
        # for each modality create W matrix and alpha vectors
        for m, feature_plate in get_feature_plates(-2).items():
            # the actual dimensions obtained by plates are read from right to left/inner to outer
            with self.latent_factor_plate:
                # Sample alphas (controls narrowness of weight distr for each factor) from a Gamma distribution
                # Gamma parametrization k, theta or eq. a, b; (where k=a and theta=1/b) 
                # (if k integer) Gamma = the sum of k independent exponentially distributed random variables, each of 
                # which has a mean of theta
                alpha = pyro.sample(f"alpha_{m}", pyro.distributions.Gamma(to_device(1.), 1.))
                
                with feature_plate:
                    # sample weight matrix with Normal prior distribution with alpha narrowness
                    Ws[m] = pyro.sample(f"W_{m}", pyro.distributions.Normal(to_device(0.), 1. / alpha))                
                
        # create Z matrix
        # (the actual dimensions are read from right to left/inner to outer)
        with self.latent_factor_plate, sample_plate:
            # sample factor matrix with Normal prior distribution
            Z = pyro.sample("Z", pyro.distributions.Normal(to_device(0.), 1.))
    
        # estimate for Y
        Y_hats = { m: torch.matmul(Z, W.t())
                   for m, W in Ws.items() }
        
        for m, feature_plate in get_feature_plates(-1).items():
            with feature_plate:
                # sample scale (tau) parameter for each feature-~~sample~~ pair with LogNormal prior (has to be positive)
                # add 0.001 to avoid NaNs when Normal(σ = 0)
                scale_tau = 0.001 + pyro.sample(f"scale_{m}", pyro.distributions.LogNormal(to_device(0.), 1.))
                
                with sample_plate as sub_indices:
                    # masking the NA values such that they are not considered in the distributions
                    obs_mask = self.obs_masks[m]
                    if obs_mask is None:
                        obs_mask = True
                    else:
                        obs_mask = obs_mask[sub_indices, :]

                    Y, Y_hat = self.get_sample_batch(m, sub_indices), Y_hats[m]

                    with pyro.poutine.mask(mask=obs_mask): #type: ignore
                        # # sample scale parameter for each feature-sample pair with LogNormal prior (has to be positive)
                        # scale = pyro.sample(f"scale_{m}", pyro.distributions.LogNormal(to_device(0., 1.)))
                        
                        # compare sampled estimation to the true observation Y
                        pyro.sample(f"obs_{m}", pyro.distributions.Normal(Y_hat, scale_tau), obs=Y)
                        # pyro.sample(f"obs_{m}", pyro.distributions.Normal(Y_hat, self.empirical_stds[m]), obs=Y)
                        # NB mean = r * (1-p) / p
                        #    vari = r * (1-p) / p^2
                        # r = torch.clamp(-Y_hat ** 2/(Y_hat - scale_tau), min=0.1)
                        # p = torch.clamp(torch.clamp(Y_hat, min=0.1)/scale_tau, 0.01, 0.99)
                        # print(r[:5, :5], p[:5, :5], Y[:5, :5])
                        # pyro.sample(f"obs_{m}", pyro.distributions.NegativeBinomial(r, p), obs=Y)

    def train(self, lr=0.002, num_iterations = 4000):
        # set training parameters
        optimizer = pyro.optim.Adam({"lr": lr})
        elbo = pyro.infer.Trace_ELBO()
        guide = self.Guide(self.model)
        t0 = time.time()
        # guide = autoguide.AutoDelta(self.model)
        
        # initialize stochastic variational inference
        svi = pyro.infer.SVI(
            model = self.model,
            guide = guide,
            optim = optimizer,
            loss = elbo
        )
        
        train_loss = []
        for j in range(num_iterations):
            # calculate the loss and take a gradient step
            # (loss should be already scaled down by the subsample_size)
            loss = svi.step()

            train_loss.append(loss/self.num_samples)
            if j % 200 == 0:
                print("[%02d:%02.1d iteration %05d] loss: %.4f" % (round((time.time() - t0)/60), round(time.time() - t0, ndigits=1) % 60, j + 1, loss / self.num_samples))
        
        # Obtain maximum a posteriori estimates for W and Z
        # map_estimates = guide(self.Y)  # not sure why needed Y?
        # "Note that Pyro enforces that model() and guide() have the same call signature, i.e. both callables should take the same arguments."
        with torch.no_grad():
            map_estimates = guide()
        
        return train_loss, map_estimates, guide
    
    def get_matrices(self, device:torch.device = torch.device("cpu"))  -> MOFAMatrices:
        guidename = "AutoNormal.locs" if self.Guide == autoguide.AutoNormal else "AutoDelta"
        return MOFAMatrices(
            Z=pyro.get_param_store().get_param(guidename+".Z").detach().to(device),
            Ws={ m: pyro.get_param_store().get_param(f'{guidename}.W_{m}').detach().to(device)
                 for m in self.Ys.keys() }
        )
    
    def save_h5(self, obs_index, modality_dset: dict[str, ty.Any], file: str, compression = 1):
        import h5py
        assert len(obs_index) == self.num_samples
        assert set(modality_dset.keys()) == set(self.Ys.keys())
        matrices = self.get_matrices()

        if os.path.exists(file):
            os.rename(file, file + ".bak")

        with h5py.File(file, "w") as f:
            opt = dict(compression="gzip", compression_opts=compression)
            for m, dset in modality_dset.items():
                f.create_dataset(f'features/{m}', data=np.array([*dset.var['feature_types'].index], dtype='S'), **opt)
                f.create_dataset(f'data/{m}/group1', data=dset.X.toarray(), **opt)
            f.create_dataset('samples/group1', data=np.array([*obs_index], dtype='S'), **opt)

            f.create_dataset('expectations/Z/group1', data=matrices.Z.T.numpy(), **opt)
            for m in self.Ys.keys():
                f.create_dataset(f'expectations/W/{m}', data=matrices.Ws[m].T.numpy(), **opt)
            f.create_dataset('model_options/likelihoods', data=np.array(['gaussian' for _ in modality_dset], dtype='S'), **opt)


