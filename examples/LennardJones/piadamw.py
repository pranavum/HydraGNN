from torch.optim import AdamW

class PhysicsInformedAdamW(AdamW):
    def __init__(self, params, lr=1e-3, mu=1e-15, **kwargs):
        super(PhysicsInformedAdamW, self).__init__(params, lr=lr, **kwargs)
        self.mu = mu
    def step(self, constraint, closure=None):
        loss = super(PhysicsInformedAdamW, self).step(closure)
        for group in self.param_groups:
            mu = self.mu
            for param in group['params']:
                if param.grad is None:
                    continue
                #constraint = self.compute_constraint(param)
                param.data.add_(-mu * constraint)
        return loss
    def compute_constraint(self, param):
        return None