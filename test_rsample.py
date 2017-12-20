from __future__ import absolute_import, division, print_function

import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.distributions.transformed_distribution import TransformedDistribution
from pyro.infer.svi import SVI
from pyro.util import ng_ones, ng_zeros

def param_mse(name, target):
    return torch.sum(torch.pow(target - pyro.param(name), 2.0)).data.cpu().numpy()[0]


def param_abs_error(name, target):
    return torch.sum(torch.abs(target - pyro.param(name))).data.cpu().numpy()[0]

def noise(eps=0.2):
    return eps*torch.randn(1)

class PoissonGamma(object):
    def __init__(self):
        # poisson-gamma model
        # gamma prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([1.0]))
        # gamma prior hyperparameter
        self.beta0 = Variable(torch.Tensor([1.0]))
        self.data = []
        self.data.append(Variable(torch.Tensor([1.0])))
        self.data.append(Variable(torch.Tensor([2.0])))
        self.data.append(Variable(torch.Tensor([3.0])))
        self.n_data = len(self.data)
        sum_data = self.data[0] + self.data[1] + self.data[2]
        self.alpha_n = self.alpha0 + sum_data  # posterior alpha
        self.beta_n = self.beta0 + \
            Variable(torch.Tensor([self.n_data]))  # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def test(self):
        pyro.clear_param_store()
        print("*** poisson gamma ***")

        def model():
            lambda_latent = pyro.sample("lambda_latent", dist.gamma, self.alpha0, self.beta0)
            pyro.map_data("aaa",
                          self.data, lambda i, x: pyro.observe(
                              "obs_{}".format(i), dist.poisson, x, lambda_latent), batch_size=3)
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log",
                Variable(
                    self.log_alpha_n.data +
                    noise(),
                    requires_grad=True))
            beta_q_log = pyro.param(
                "beta_q_log",
                Variable(
                    self.log_beta_n.data -
                    noise(),
                    requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("lambda_latent", dist.gamma, alpha_q, beta_q)
            pyro.map_data("aaa", self.data, lambda i, x: None, batch_size=3)

        adam = optim.Adam({"lr": .001, "betas": (0.90, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(9001):
            svi.step()

            if k%250==0:
                alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
                beta_error = param_abs_error("beta_q_log", self.log_beta_n)
                print("errors [%04d]: %.4f %.4f" %(k, alpha_error, beta_error))

class ExponentialGamma(object):
    def __init__(self):
        # exponential-gamma model
        # gamma prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([1.0]))
        # gamma prior hyperparameter
        self.beta0 = Variable(torch.Tensor([1.0]))
        self.n_data = 2
        self.data = Variable(torch.Tensor([3.0, 2.0]))  # two observations
        self.alpha_n = self.alpha0 + \
            Variable(torch.Tensor([self.n_data]))  # posterior alpha
        self.beta_n = self.beta0 + torch.sum(self.data)  # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def test(self):
        pyro.clear_param_store()
        print("*** exponential gamma ***")

        def model():
            lambda_latent = pyro.sample("lambda_latent", dist.gamma, self.alpha0, self.beta0)
            pyro.observe("obs0", dist.exponential, self.data[0], lambda_latent)
            pyro.observe("obs1", dist.exponential, self.data[1], lambda_latent)
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log",
                Variable(self.log_alpha_n.data + noise(), requires_grad=True))
            beta_q_log = pyro.param(
                "beta_q_log",
                Variable(self.log_beta_n.data - noise(), requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("lambda_latent", dist.gamma, alpha_q, beta_q)

        adam = optim.Adam({"lr": .0008, "betas": (0.90, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(9001):
            svi.step()

            if k % 500==0:
                alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
                beta_error = param_abs_error("beta_q_log", self.log_beta_n)
                print("errors [%04d]: %.4f %.4f" %(k, alpha_error, beta_error))

class BernoulliBeta(object):
    def __init__(self):
        # bernoulli-beta model
        # beta prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([1.0]))
        self.beta0 = Variable(torch.Tensor([1.0]))  # beta prior hyperparameter
        self.data = []
        self.data.append(Variable(torch.Tensor([0.0])))
        self.data.append(Variable(torch.Tensor([1.0])))
        self.data.append(Variable(torch.Tensor([1.0])))
        self.data.append(Variable(torch.Tensor([1.0])))
        self.n_data = len(self.data)
        self.batch_size = None
        data_sum = self.data[0] + self.data[1] + self.data[2] + self.data[3]
        self.alpha_n = self.alpha0 + data_sum  # posterior alpha
        self.beta_n = self.beta0 - data_sum + \
            Variable(torch.Tensor([self.n_data]))
        # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def test(self):
        pyro.clear_param_store()
        print("*** bernoulli beta ***")

        def model():
            p_latent = pyro.sample("p_latent", dist.beta, self.alpha0, self.beta0)
            pyro.map_data("aaa",
                          self.data, lambda i, x: pyro.observe(
                              "obs_{}".format(i), dist.bernoulli, x, p_latent),
                          batch_size=self.batch_size)
            return p_latent

        def guide():
            alpha_q_log = pyro.param("alpha_q_log",
                                     Variable(self.log_alpha_n.data + noise(),
                                              requires_grad=True))
            beta_q_log = pyro.param("beta_q_log",
                                    Variable(self.log_beta_n.data - noise(), requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("p_latent", dist.beta, alpha_q, beta_q)
            pyro.map_data("aaa", self.data, lambda i, x: None, batch_size=self.batch_size)

        adam = optim.Adam({"lr": .002, "betas": (0.80, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(9001):
            svi.step()

            if k%500==0:
                alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
                beta_error = param_abs_error("beta_q_log", self.log_beta_n)
                print("errors [%04d]: %.4f %.4f" %(k, alpha_error, beta_error))

PoissonGamma().test()
#ExponentialGamma().test()
#BernoulliBeta().test()
