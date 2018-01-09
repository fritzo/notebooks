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
from pyro.distributions.beta import Beta as NonRepBeta
from pyro.distributions.dirichlet import Dirichlet as NonRepDirichlet

import sys


def param_mse(name, target):
    return torch.sum(torch.pow(target - pyro.param(name), 2.0)).data.cpu().numpy()[0]

def gamma_mean_error(name1, name2, target1, target2):
    target_mean = torch.exp(target1-target2)
    actual_mean = torch.exp(pyro.param(name1) - pyro.param(name2))
    return torch.abs(actual_mean - target_mean).data.cpu().numpy()[0]

def gamma_var_error(name1, name2, target1, target2):
    target_var = torch.exp(target1-2.0*target2)
    actual_var = torch.exp(pyro.param(name1) - 2.0*pyro.param(name2))
    return torch.abs(actual_var - target_var).data.cpu().numpy()[0]

def beta_mean_var_error(name1, name2, target1, target2):
    alphaT = torch.exp(target1)
    betaT = torch.exp(target2)
    alpha = pyro.param(name1)
    beta = pyro.param(name2)
    target_mean = alphaT/(alphaT+betaT)
    actual_mean = alpha/(alpha+beta)
    mean_error = torch.abs(actual_mean - target_mean).data.cpu().numpy()[0]
    target_var = alphaT*betaT/((alphaT+betaT)**2*(alphaT+betaT+1))
    actual_var = alpha*beta/((alpha+beta)**2*(alpha+beta+1))
    var_error = torch.abs(actual_var - target_var).data.cpu().numpy()[0]
    return mean_error, var_error

def param_abs_error(name, target):
    return torch.sum(torch.abs(target - pyro.param(name))).data.cpu().numpy()[0]

def noise(eps=0.6):
    return eps*torch.randn(1)

class PoissonGamma(object):
    def __init__(self, alpha0=1.0, n_data=50):
        # poisson-gamma model
        # gamma prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([alpha0]))
        # gamma prior hyperparameter
        self.beta0 = Variable(torch.Tensor([1.0]))
        self.n_data = n_data
        self.data = self.model().unsqueeze(1)
        self.alpha_n = self.alpha0 + self.data.sum()  # posterior alpha
        self.beta_n = self.beta0 + \
            Variable(torch.Tensor([self.n_data]))  # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def model(self, obs=None):
        lambda_latent = pyro.sample("lambda_latent", dist.gamma, self.alpha0, self.beta0)
        with pyro.iarange('observe_data'):
            if obs is None:
                return pyro.observe('obs', dist.poisson, obs, lambda_latent.expand(self.n_data))
            pyro.observe('obs', dist.poisson, obs, lambda_latent)

    def test(self, use_rep=False, lr=0.001, beta1=0.90, beta2=0.999,
             verbose=False, seed=1, noise_epsilon=0.6, report_frequencies=[5000, 10000, 15000]):
        pyro.clear_param_store()
        pyro.util.set_rng_seed(seed)
        if verbose:
            print("*** poisson gamma ***   [reparameterized = %s]" % use_rep)

        def guide(obs=None):
            alpha_q_log = pyro.param("alpha_q_log",
                Variable(self.log_alpha_n.data + noise(eps=noise_epsilon), requires_grad=True))
            beta_q_log = pyro.param("beta_q_log",
                Variable(self.log_beta_n.data - noise(eps=noise_epsilon), requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            if use_rep:
                pyro.sample("lambda_latent", dist.gamma, alpha_q, beta_q)
            else:
                pyro.sample("lambda_latent", NonRepGamma(alpha_q, beta_q))

        adam = optim.Adam({"lr": lr, "betas": (beta1, beta2)})
        svi = SVI(self.model, guide, adam, loss="ELBO", trace_graph=False)
        svi_eval = SVI(self.model, guide, adam, loss="ELBO", trace_graph=False, num_particles=500)
        results = []

        for k in range(report_frequencies[-1]):
            svi.step(self.data)

            if (k+1) in report_frequencies:
                alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
                beta_error = param_abs_error("beta_q_log", self.log_beta_n)
                mean_error = gamma_mean_error("alpha_q_log", "beta_q_log",
                                              self.log_alpha_n, self.log_beta_n)
                var_error = gamma_var_error("alpha_q_log", "beta_q_log",
                                              self.log_alpha_n, self.log_beta_n)
                if verbose:
                    print("[%04d]: %.4f    %.4f     %.4f      %.4f" % (k, alpha_error, beta_error,
                                                                       mean_error, var_error))

                elbo = -svi_eval.evaluate_loss(self.data)
                results.append((elbo, alpha_error, beta_error))

        return results

class ExponentialGamma(object):
    def __init__(self, use_rep=True):
        # exponential-gamma model
        # gamma prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([1.0]))
        # gamma prior hyperparameter
        self.beta0 = Variable(torch.Tensor([1.0]))
        self.data = Variable(torch.Tensor([[2.0],[3.0]]))
        self.n_data = self.data.size(0)
        self.alpha_n = self.alpha0 + Variable(torch.Tensor([self.n_data]))  # posterior alpha
        self.beta_n = self.beta0 + torch.sum(self.data)  # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)
        self.use_rep=use_rep

    def test(self):
        pyro.clear_param_store()
        pyro.util.set_rng_seed(5)
        print("*** exponential gamma ***   [reparameterized = %s]" % self.use_rep)
        print("        log_alpha log_beta   mean_error  var_error")

        def model():
            lambda_latent = pyro.sample("lambda_latent", dist.gamma, self.alpha0, self.beta0)
            with pyro.iarange('observe_data'):
                pyro.observe('obs', dist.exponential, self.data, lambda_latent)
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log", Variable(self.log_alpha_n.data + noise(), requires_grad=True))
            beta_q_log = pyro.param(
                "beta_q_log", Variable(self.log_beta_n.data - noise(), requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            if self.use_rep:
                pyro.sample("lambda_latent", dist.gamma, alpha_q, beta_q)
            else:
                pyro.sample("lambda_latent", NonRepGamma(alpha_q, beta_q))

        if self.use_rep:
            adam = optim.Adam({"lr": .0005, "betas": (0.95, 0.999)})
        else:
            adam = optim.Adam({"lr": .0005, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(15001):
            svi.step()

            if k % 500==0:
                alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
                beta_error = param_abs_error("beta_q_log", self.log_beta_n)
                mean_error = gamma_mean_error("alpha_q_log", "beta_q_log",
                                              self.log_alpha_n, self.log_beta_n)
                var_error = gamma_var_error("alpha_q_log", "beta_q_log",
                                              self.log_alpha_n, self.log_beta_n)
                print("[%04d]: %.4f    %.4f     %.4f      %.4f" % (k, alpha_error, beta_error,
                                                                   mean_error, var_error))

class BernoulliBeta(object):
    def __init__(self, use_rep=True):
        # bernoulli-beta model
        # beta prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([1.0]))
        self.beta0 = Variable(torch.Tensor([1.0]))  # beta prior hyperparameter
        self.data = Variable(torch.Tensor([[1.0],[0.0],[1.0],[1.0]]))
        self.n_data = self.data.size(0)
        self.alpha_n = self.alpha0 + self.data.sum()  # posterior alpha
        self.beta_n = self.beta0 - self.data.sum() + Variable(torch.Tensor([self.n_data]))
        # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)
        self.use_rep=use_rep

    def test(self):
        pyro.clear_param_store()
        pyro.util.set_rng_seed(1)
        print("*** bernoulli beta ***   [reparameterized = %s]" % self.use_rep)
        print("        log_alpha log_beta   mean_error  var_error")

        def model():
            p_latent = pyro.sample("p_latent", dist.beta, self.alpha0, self.beta0)
            with pyro.iarange('observe_data'):
                pyro.observe('obs', dist.bernoulli, self.data, p_latent)
            return p_latent

        def guide():
            alpha_q_log = pyro.param("alpha_q_log", Variable(self.log_alpha_n.data + noise(),
                                              requires_grad=True))
            beta_q_log = pyro.param("beta_q_log",
                                    Variable(self.log_beta_n.data - noise(), requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            if self.use_rep:
                pyro.sample("p_latent", dist.beta, alpha_q, beta_q)
            else:
                pyro.sample("p_latent", NonRepBeta(alpha_q, beta_q))

        if self.use_rep:
            adam = optim.Adam({"lr": .0005, "betas": (0.95, 0.999)})
        else:
            adam = optim.Adam({"lr": .0005, "betas": (0.95, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(15001):
            svi.step()

            if k%500==0:
                alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
                beta_error = param_abs_error("beta_q_log", self.log_beta_n)
                mean_error, var_error = beta_mean_var_error("alpha_q_log", "beta_q_log",
                                              self.log_alpha_n, self.log_beta_n)
                print("[%04d]: %.4f    %.4f     %.4f      %.4f" % (k, alpha_error, beta_error,
                                                                   mean_error, var_error))

class MultinomialDirichlet(object):
    def __init__(self, use_rep=True, N=8, noise=4.0):
        self.N = N
        No2 = int(N/2)
        self.alpha0 = Variable(torch.Tensor(2**np.arange(N) / (2**No2)))
        self.data = Variable(torch.Tensor(20*np.random.randint(0,N,N)))
        self.data[0]=0.0
        self.data[-1]=0.0
        #self.data = Variable(torch.Tensor(np.random.randint(0,N,N)).unsqueeze(1))
        self.n_data = self.data.size(0)
        self.alpha_n = self.alpha0 + self.data#.squeeze()
        self.log_alpha_n = torch.log(self.alpha_n)
        self.use_rep=use_rep
        self.noise=noise

    def test(self):
        pyro.clear_param_store()
        pyro.util.set_rng_seed(1)
        print("*** multinomial dirichlet ***   [reparameterized = %s]" % self.use_rep)
        print("alpha0: ", self.alpha0.data.numpy())
        print("alphap: ", np.exp(self.log_alpha_n.data.numpy()))

        def model():
            p_latent = pyro.sample("p_latent", dist.dirichlet, self.alpha0)
            with pyro.iarange('observe_data'):
                pyro.sample('obs', dist.multinomial, p_latent, 1, obs=self.data)

        def guide():
            alpha_q_log = pyro.param("alpha_q_log", Variable(self.log_alpha_n.data +
                                                             self.noise*torch.randn(self.N),
                                                             requires_grad=True))
            alpha_q = torch.exp(alpha_q_log)
            if self.use_rep:
                pyro.sample("p_latent", dist.dirichlet, alpha_q)
            else:
                pyro.sample("p_latent", NonRepDirichlet(alpha_q))

        if self.use_rep:
            adam = optim.Adam({"lr": .0003, "betas": (0.95, 0.999)})
        else:
            adam = optim.Adam({"lr": .0003, "betas": (0.95, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(25001):
            svi.step()

            if k%500==0:
                alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
                min_alpha = np.min(pyro.param("alpha_q_log").data.numpy())
                print("[%04d]: %.4f %.4f" % (k, alpha_error, min_alpha))


md = MultinomialDirichlet(use_rep=True, N=3, noise=0.5)
#md = MultinomialDirichlet(use_rep=False)
md.test()
sys.exit()

pg = PoissonGamma()
lrs = 2**np.arange(10)*0.0001
assert(len(sys.argv)==3)
alpha0=float(sys.argv[1])
tag=sys.argv[2]
print("alpha0: %.4f" % alpha0)
models = [(PoissonGamma(alpha0=alpha0, n_data=50), "poisson_gamma_%d_n50" % int(100*alpha0)),
          (PoissonGamma(alpha0=alpha0, n_data=500), "poisson_gamma_%d_n500" % int(100*alpha0))]
results = {}

for model, name in models:
    results[name] = {}
    for use_rep in [False, True]:
        repstring = 'Reparam' if use_rep else 'NonRep '
        results[name][repstring] = {}
        for lr in lrs:
            for beta1 in [0.95]:
                results[name][repstring][(lr, beta1)] = []
                print("doing %s %s %f" % (name, repstring, lr))
                for seed in range(10):
                    result = model.test(use_rep=use_rep, lr=lr, beta1=beta1, beta2=0.999,
                             verbose=False, seed=seed, noise_epsilon=1.0,
                             report_frequencies=[1,5000,10000,15000,20000])
                    results[name][repstring][(lr, beta1)].append(result)

print(results)
with open("pg_results.%s.pkl" % tag, "wb") as output_file:
    output_file.write(cloudpickle.dumps(results, protocol=2))

##with open(filename, "rb") as input_file:
##   state = cloudpickle.loads(input_file.read())
#pickle.load(open("pg_results.pkl", "rb" ) )
#print('rep_curve = ', rep_curve)
#print('nonrep_curve = ', nonrep_curve)

#PoissonGamma(use_rep=False).test()
#PoissonGamma(use_rep=True).test()
#ExponentialGamma(use_rep=False).test()
#ExponentialGamma(use_rep=True).test()
#BernoulliBeta(use_rep=False).test()
#BernoulliBeta(use_rep=True).test()
