import torch

class VP():
    def __init__(self, beta_min, beta_max, num_steps):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.num_steps = num_steps
        self.discrete_betas = torch.linspace(beta_min / num_steps, beta_max / num_steps, num_steps)
        self.alphas = 1. - self.discrete_betas

    def _beta_t(self, t):
        # TODO: compute beta(t)
        beta_t = self.beta_0 + t*(self.beta_1 - self.beta_0)
        return beta_t
    
    def _c_t(self, t):
        # TODO: compute c(t)
        c_t = -0.25 * (self.beta_1-self.beta_0) * (t**2) - 0.5 * self.beta_0 * t
        return c_t

    def marginal_proba(self, x, t):
        """ Compute the mean and standard deviation of the marginal prob p(x_t|x_0)
        """
        # TODO: compute mu and std (std is a scalar)
        c_t = self._c_t(t)
        mu_t = torch.exp(c_t[:,None,None,None]) * x
        std_t = torch.sqrt(1. - torch.exp(2.*c_t))
        return mu_t, std_t

    def drift(self, x, t):
        """ Compute the VP drift coefficient f(x, t) 
        """
        # TODO: compute drift coefficient -- make sure to give beta_t the appropriate shape
        drift = -0.5*self._beta_t(t)[:,None,None,None] * x
        return drift

    def diffusion(self, t):
        """ Compute the VP diffusion coefficient g(t)
        """
        # TODO: compute diffusion coefficient
        diffusion = torch.sqrt(torch.tensor(self._beta_t(t)))
        return diffusion
        