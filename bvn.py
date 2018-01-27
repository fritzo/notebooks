class BivariateNormal(Distribution):
    reparameterized = True

    def __init__(self, loc, scale_triu, batch_size=None):
        self.loc = loc
        self.scale_triu = scale_triu
        self.batch_size = 1 if batch_size is None else batch_size

    def batch_shape(self, x=None):
        loc = self.loc.expand(self.batch_size, *self.loc.size()).squeeze(0)
        if x is not None:
            if x.size()[-1] != loc.size()[-1]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.loc.size()[0], but got {} vs {}".format(
                                     x.size(-1), loc.size(-1)))
            try:
                loc = loc.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `loc` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(loc.size(), x.size(), str(e)))

        return loc.size()[:-1]

    def event_shape(self):
        return self.loc.size()[-1:]

    def sample(self):
        return self.loc + torch.mv(self.scale_triu.t(), Variable(torch.randn(self.loc.size())), )

    def batch_log_pdf(self, x):
        delta = x - self.loc
        z0 = delta[..., 0] / self.scale_triu[..., 0, 0]
        z1 = (delta[..., 1] - self.scale_triu[..., 0, 1] * z0) / self.scale_triu[..., 1, 1]
        z = torch.stack([z0, z1], dim=-1)
        mahalanobis_squared = (z ** 2).sum(-1)
        normalization_constant = self.scale_triu.diag().log().sum(-1) + np.log(2 * np.pi)
        return -(normalization_constant + 0.5 * mahalanobis_squared).unsqueeze(-1)

    def entropy(self):
        return self.scale_triu.diag().log().sum() + (1 + math.log(2 * math.pi))

def _BVN_backward_reptrick(white, scale_triu, grad_output):
    grad = (grad_output.unsqueeze(-1) * white.unsqueeze(-2)).squeeze(0)
    return grad_output, torch.triu(grad.t())

class _RepTrickSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_triu):
        ctx.save_for_backward(scale_triu)
        ctx.white = loc.new(loc.size()).normal_()
        return loc + torch.mm(ctx.white, scale_triu)

    @staticmethod
    def backward(ctx, grad_output):
        scale_triu, = ctx.saved_variables
        return _BVN_backward_reptrick(Variable(ctx.white), scale_triu, grad_output)

class _CanonicalSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_triu):
        ctx.save_for_backward(scale_triu)
        ctx.white = loc.new(loc.size()).normal_()
        ctx.z = torch.mm(ctx.white, scale_triu)
        return loc + ctx.z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        scale_triu, = ctx.saved_tensors
        return _BVN_backward_canonical(ctx.white, scale_triu, grad_output, ctx.z)


class BivariateNormalRepTrick(BivariateNormal):
    def sample(self):
        loc = self.loc.expand(self.batch_size, *self.loc.size())
        return _RepTrickSample.apply(loc, self.scale_triu)

class BivariateNormalCanonical(BivariateNormal):
    def sample(self):
        loc = self.loc.expand(self.batch_size, *self.loc.size())
        return _CanonicalSample.apply(loc, self.scale_triu)

def _BVN_backward_canonical(white, scale_triu, grad_output, z):
    g = grad_output
    epsilon = white
    epsilon_b = white.unsqueeze(-1).squeeze(0)
    g_a = g.unsqueeze(-2).squeeze(0)
    diff_R_ba = 0.5 * epsilon_b * g_a

    z_a = z.unsqueeze(-2).squeeze(0)
    R_invT_g_b = torch.trtrs(g.squeeze(0), scale_triu, transpose=True)[0]
    diff_R_ba += 0.5 * R_invT_g_b * z_a

    Sigma = torch.mm(scale_triu.t(), scale_triu)
    Sigma_inv = torch.inverse(Sigma)
    V, D, _ = torch.svd(Sigma_inv)
    D_outer = D.unsqueeze(-1) + D.unsqueeze(0)

    R_inv = torch.eye(2).type_as(g)
    R_inv = torch.trtrs(R_inv, scale_triu, transpose=False)[0]
    R_inv_ib = R_inv.t().unsqueeze(1).unsqueeze(-1)
    xi = 0.5 * R_inv_ib * Sigma_inv.unsqueeze(1)

    delta_ai = torch.eye(2).type_as(g).unsqueeze(2)
    Sigma_inv_R_inv_jb = torch.mm(Sigma_inv, R_inv).t().unsqueeze(1).unsqueeze(1)
    xi -= 0.5 * delta_ai * Sigma_inv_R_inv_jb

    Xi = xi + torch.transpose(xi.clone(), 2, 3)
    Xi_tilde = torch.matmul(V.t(), torch.matmul(Xi, V))
    S_tilde = Xi_tilde / D_outer
    S = torch.matmul(V, torch.matmul(S_tilde, V.t()))

    S_z = torch.matmul(S, z.t())
    S_z_g = torch.matmul(g, S_z).squeeze(-1).squeeze(-1)
    diff_R_ba += S_z_g

    return grad_output, torch.triu(diff_R_ba)
