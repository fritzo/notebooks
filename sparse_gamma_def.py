from __future__ import absolute_import, division, print_function
from collections import defaultdict
import cloudpickle

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer.svi import SVI
from pyro.util import ng_ones, ng_zeros
from pyro.distributions.gamma import Gamma as NonRepGamma
import pyro.poutine as poutine

import sklearn
from sklearn.datasets import fetch_olivetti_faces
import time

data = fetch_olivetti_faces()
data = Variable(torch.Tensor(np.array(255*data.images.reshape(400, -1), dtype=np.int32)))
#data = Variable(torch.IntTensor(np.array(255*data.images.reshape(400, -1), dtype=np.int32)))

class SparseGammaDEF(object):
    def __init__(self):
        self.top_width = 100
        self.mid_width = 40
        self.bottom_width = 15
        self.image_size = 64 * 64
        self.alpha_z = Variable(torch.Tensor([0.1]))
        self.alpha_w = Variable(torch.Tensor([0.1]))
        self.beta_w = Variable(torch.Tensor([0.3]))

    def model(self, x):
        x_size = x.size(0)
        with poutine.scale(None, 1.0e-7):
            z_top = pyro.sample("z_top", dist.gamma, self.alpha_z.expand(x_size, self.top_width),
                                                     self.alpha_z.expand(x_size, self.top_width))
            w_top = pyro.sample("w_top", dist.gamma, self.alpha_w.expand(self.top_width * self.mid_width),
                                self.beta_w.expand(self.top_width * self.mid_width))
            w_top = w_top.view(self.top_width, self.mid_width)
            mean_mid = torch.mm(z_top, w_top)

            z_mid = pyro.sample("z_mid", dist.gamma, self.alpha_z.expand(x_size, self.mid_width),
                                self.alpha_z.expand(x_size, self.mid_width) / mean_mid)
            w_mid = pyro.sample("w_mid", dist.gamma, self.alpha_w.expand(self.mid_width * self.bottom_width),
                                self.beta_w.expand(self.mid_width * self.bottom_width))
            w_mid = w_mid.view(self.mid_width, self.bottom_width)
            mean_bottom = torch.mm(z_mid, w_mid)

            z_bottom = pyro.sample("z_bottom", dist.gamma, self.alpha_z.expand(x_size, self.bottom_width),
                                   self.alpha_z.expand(x_size, self.bottom_width) / mean_bottom)
            w_bottom = pyro.sample("w_bottom", dist.gamma,
                                   self.alpha_w.expand(self.bottom_width * self.image_size),
                                   self.beta_w.expand(self.bottom_width * self.image_size))
            w_bottom = w_bottom.view(self.bottom_width, self.image_size)
            mean_obs = torch.mm(z_bottom, w_bottom)

            with pyro.iarange('observe_data'):
                pyro.observe('obs', dist.poisson, x, mean_obs)

    def guide(self, x):
        x_size = x.size(0)

        def sample_zs(name, width):
            alpha_z_q = pyro.param("log_alpha_z_q_%s" % name, Variable(-2.0*torch.ones(x_size, width),
                                                                       requires_grad=True))
            beta_z_q = pyro.param("log_beta_z_q_%s" % name, Variable(-2.0*torch.ones(x_size, width),
                                                                     requires_grad=True))
            alpha_z_q, beta_z_q = torch.exp(alpha_z_q), torch.exp(beta_z_q)
            pyro.sample("z_%s" % name, dist.gamma, alpha_z_q, beta_z_q)

        with poutine.scale(None, 1.0e-7):
            sample_zs("top", self.top_width)
            sample_zs("mid", self.mid_width)
            sample_zs("bottom", self.bottom_width)

        def sample_ws(name, width):
            alpha_w_q = pyro.param("log_alpha_w_q_%s" % name, Variable(6.0*torch.ones(width),
                                                                       requires_grad=True))
            beta_w_q = pyro.param("log_beta_w_q_%s" % name, Variable(6.0*torch.ones(width),
                                                                     requires_grad=True))
            alpha_w_q, beta_w_q = torch.exp(alpha_w_q), torch.exp(beta_w_q)
            pyro.sample("w_%s" % name, dist.gamma, alpha_w_q, beta_w_q)

        with poutine.scale(None, 1.0e-7):
            sample_ws("top", self.top_width * self.mid_width)
            sample_ws("mid", self.mid_width * self.bottom_width)
            sample_ws("bottom", self.bottom_width * self.image_size)

    def do_inference(self):
        pyro.clear_param_store()
        pyro.util.set_rng_seed(0)
        t0 = time.time()

        adam = optim.Adam({"lr": 1.0e-2, "betas": (0.97, 0.999)})
        svi = SVI(self.model, self.guide, adam, loss="ELBO", trace_graph=False)
        losses = []

        for k in range(50000):
            loss = svi.step(data)
            losses.append(loss)
            if k % 20 == 0 and k > 100:
                t_k = time.time()
                print("[epoch %05d] mean elbo: %.5f     elapsed time: %.4f" % (k, -np.mean(losses[-100:]),
                      t_k - t0))

        return results

sgdef = SparseGammaDEF()
sgdef.do_inference()
