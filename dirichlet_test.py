import numpy as np
import torch
from torch.autograd import Variable

# p(z) = Dir(z | [alpha, alpha, 1])
# L = Var(z_3)
# the derivative = dL/dalpha
# should be zero at alpha = alpha_crit where the loss attains its maximum

alpha_crit = 0.25 * (np.sqrt(5.0) - 1.0)
loss_max = 0.9016994375
print("alpha_crit", alpha_crit)

def compute_grad(alpha=alpha_crit, num_particles=100000):
    alpha = Variable(torch.Tensor([alpha]), requires_grad=True)
    alpha_vec = torch.cat([alpha, alpha, Variable(torch.ones(1))])
    d = torch.distributions.Dirichlet(alpha_vec.expand(num_particles, 3))
    z = d.rsample()
    mean_z3 = 1.0 / (2.0 * alpha + 1.0)
    delta = z[:, 2] - mean_z3
    loss = torch.pow(delta, 2.0).mean()
    alpha_grad = torch.autograd.grad(loss, [alpha])[0]
    return alpha_grad.data.numpy(), loss.data.numpy()

def true_grad(alpha=alpha_crit):
    num = 1.0 - 2.0 * alpha - 4.0 * alpha**2
    den1 = (1.0 + alpha)**2
    den2 = (1.0 + 2.0 * alpha)**3
    return num/(den1*den2)

for shift in [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10]:
    comp_grad, loss = compute_grad(alpha=alpha_crit + shift)
    print(("[alpha=alpha_c + %.2f]  computed_grad: %.5f  true_grad: %.5f   " +
          "loss: %.5f  loss_max: %5f") % (shift, comp_grad,
                                         true_grad(alpha=alpha_crit + shift), loss, loss_max))
