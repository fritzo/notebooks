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
#from pyro.distributions.gamma import Gamma as NonRepGamma
import pyro.poutine as poutine

import sklearn
from sklearn.datasets import fetch_olivetti_faces
import time

#data = fetch_olivetti_faces()
#data = Variable(torch.Tensor(np.array(255*data.images.reshape(400, -1), dtype=np.int32)))
#data = Variable(torch.IntTensor(np.array(255*data.images.reshape(400, -1), dtype=np.int32)))
data = Variable(torch.Tensor(np.loadtxt('faces_training.csv',delimiter=',')))

class SparseGammaDEF(object):
    def __init__(self):
        self.top_width = 100
        self.mid_width = 40
        self.bottom_width = 15
        self.image_size = 64 * 64
        self.alpha_z = Variable(torch.Tensor([0.1]))
        self.alpha_w = Variable(torch.Tensor([0.1]))
        self.beta_w = Variable(torch.Tensor([0.3]))
        self.softplus = nn.Softplus()
        self.use_softplus = False
        print("the total number of weight variational parameters is %d" % ( 2 * (self.top_width *
              self.mid_width + self.mid_width * self.bottom_width +
              self.bottom_width * self.image_size)))
        print("the total number of local variational parameters is %d" % ( 2 * 400 * (self.top_width +
              self.mid_width + self.bottom_width)))

    def model(self, x):
        x_size = x.size(0)
        with poutine.scale(None, 1.0e-7):
            z_top = pyro.sample("z_top", dist.Gamma( self.alpha_z.expand(x_size, self.top_width),
                                                     self.alpha_z.expand(x_size, self.top_width)))
            w_top = pyro.sample("w_top", dist.Gamma( self.alpha_w.expand(self.top_width * self.mid_width),
                                self.beta_w.expand(self.top_width * self.mid_width)))
            w_top = w_top.view(self.top_width, self.mid_width)
            mean_mid = torch.mm(z_top, w_top)

            z_mid = pyro.sample("z_mid", dist.Gamma(self.alpha_z.expand(x_size, self.mid_width),
                                self.alpha_z.expand(x_size, self.mid_width) / mean_mid))
            w_mid = pyro.sample("w_mid", dist.Gamma(
                                self.alpha_w.expand(self.mid_width * self.bottom_width),
                                self.beta_w.expand(self.mid_width * self.bottom_width)))
            w_mid = w_mid.view(self.mid_width, self.bottom_width)
            mean_bottom = torch.mm(z_mid, w_mid)

            z_bottom = pyro.sample("z_bottom", dist.Gamma(self.alpha_z.expand(x_size, self.bottom_width),
                                   self.alpha_z.expand(x_size, self.bottom_width) / mean_bottom))
            w_bottom = pyro.sample("w_bottom", dist.Gamma(
                                   self.alpha_w.expand(self.bottom_width * self.image_size),
                                   self.beta_w.expand(self.bottom_width * self.image_size)))
            w_bottom = w_bottom.view(self.bottom_width, self.image_size)
            mean_obs = torch.mm(z_bottom, w_bottom)

            with pyro.iarange('observe_data'):
                pyro.observe('obs', dist.poisson, x, mean_obs)

    def guide(self, x):
        x_size = x.size(0)

        def sample_zs(name, width, alpha_init=3.0, beta_init=5.0):
            alpha_z_q = pyro.param("log_alpha_z_q_%s" % name,
                                   Variable(alpha_init * torch.ones(x_size, width) + \
                                            0.1 * torch.randn(x_size, width), requires_grad=True))
            beta_z_q = pyro.param("log_beta_z_q_%s" % name,
                                  Variable(beta_init * torch.ones(x_size, width) + \
                                           0.1 * torch.randn(x_size, width), requires_grad=True))
            if self.use_softplus:
                alpha_z_q, beta_z_q = self.softplus(alpha_z_q), self.softplus(beta_z_q)
            else:
                alpha_z_q, beta_z_q = torch.exp(alpha_z_q), torch.exp(beta_z_q)
            z = pyro.sample("z_%s" % name, dist.Gamma( alpha_z_q, beta_z_q))
            #end = "\n" if name=="bottom" else ""
            #print("z_%s min: %e  " % (name, np.min(z.data.numpy())), end=end)

        with poutine.scale(None, 1.0e-7):
            #sample_zs("top", self.top_width, alpha_init=-2.0, beta_init=-2.0) # many iter of log
            #sample_zs("mid", self.mid_width, alpha_init=-2.0, beta_init=-1.0)
            #sample_zs("bottom", self.bottom_width, alpha_init=3.5, beta_init=5.0)
            sample_zs("top", self.top_width, alpha_init=-1.5, beta_init=2.5) # best log
            sample_zs("mid", self.mid_width, alpha_init=-1.0, beta_init=3.0)
            sample_zs("bottom", self.bottom_width, alpha_init=4.5, beta_init=3.5)

        def sample_ws(name, width, alpha_init=3.0, beta_init=5.0):
            alpha_w_q = pyro.param("log_alpha_w_q_%s" % name,
                                   Variable(alpha_init * torch.ones(width) + \
                                            0.1 * torch.randn(width), requires_grad=True))
            beta_w_q = pyro.param("log_beta_w_q_%s" % name,
                                  Variable(beta_init * torch.ones(width) + \
                                           0.1 * torch.randn(width), requires_grad=True))
            if self.use_softplus:
                alpha_w_q, beta_w_q = self.softplus(alpha_w_q), self.softplus(beta_w_q)
            else:
                alpha_w_q, beta_w_q = torch.exp(alpha_w_q), torch.exp(beta_w_q)
            w = pyro.sample("w_%s" % name, dist.Gamma(alpha_w_q, beta_w_q))
            #end = "\n" if name=="bottom" else ""
            #print("w_%s min: %e  " % (name, np.min(w.data.numpy())), end=end)

        with poutine.scale(None, 1.0e-7):
            #sample_ws("top", self.top_width * self.mid_width, alpha_init=-2.0, beta_init=2.5)
            #sample_ws("mid", self.mid_width * self.bottom_width, alpha_init=-1.0, beta_init=-1.0)
            #sample_ws("bottom", self.bottom_width * self.image_size, alpha_init=3.0, beta_init=5.0)
            sample_ws("top", self.top_width * self.mid_width, alpha_init=-1.0, beta_init=4.0)
            sample_ws("mid", self.mid_width * self.bottom_width, alpha_init=3.0, beta_init=2.0)
            sample_ws("bottom", self.bottom_width * self.image_size, alpha_init=4.0, beta_init=3.0)

    def get_w_stats(self, name):
        alpha_w_q = pyro.param("log_alpha_w_q_%s" % name)
        beta_w_q = pyro.param("log_beta_w_q_%s" % name)
        return np.min(alpha_w_q.data.numpy()), np.max(alpha_w_q.data.numpy()),\
               np.min(beta_w_q.data.numpy()),  np.max(beta_w_q.data.numpy())

    def get_z_stats(self, name):
        alpha_z_q = pyro.param("log_alpha_z_q_%s" % name)
        beta_z_q = pyro.param("log_beta_z_q_%s" % name)
        return np.min(alpha_z_q.data.numpy()), np.max(alpha_z_q.data.numpy()),\
               np.min(beta_z_q.data.numpy()),  np.max(beta_z_q.data.numpy())

    def do_inference(self):
        pyro.clear_param_store()
        pyro.util.set_rng_seed(0)
        t0 = time.time()

        adam = optim.Adam({"lr": 0.005, "betas": (0.95, 0.999)})
        svi = SVI(self.model, self.guide, adam, loss="ELBO", trace_graph=False,
                  analytic_kl=False)
        losses = []

        for k in range(100001):
            loss = svi.step(data)
            losses.append(loss)

            if k % 20 == 0 and k > 20:
                t_k = time.time()
                print("[epoch %05d] mean elbo: %.5f     elapsed time: %.4f" % (k, -np.mean(losses[-100:]),
                      t_k - t0))
                print("[W] %.2f %.2f %.2f %.2f   %.2f %.2f %.2f %.2f   %.2f %.2f %.2f %.2f" % (\
                      self.get_w_stats("top") + self.get_w_stats("mid") + self.get_w_stats("bottom") ))
                print("[Z] %.2f %.2f %.2f %.2f   %.2f %.2f %.2f %.2f   %.2f %.2f %.2f %.2f" % (\
                      self.get_z_stats("top") + self.get_z_stats("mid") + self.get_z_stats("bottom") ))

        return results

sgdef = SparseGammaDEF()
sgdef.do_inference()
