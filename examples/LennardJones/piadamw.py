import torch
from torch.optim import AdamW, adamw

class PhysicsInformedAdamW(AdamW):
    def __init__(self, params, lr=1e-3, mu=1e-15, **kwargs):
        super(PhysicsInformedAdamW, self).__init__(params, lr=lr, **kwargs)
        self.mu = mu
    def _init_constraint_group(
        self,
        constraint,
        group,
        params_with_grad,
        grads,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            constraint_grad = torch.autograd.grad(constraint, inputs=p, grad_outputs=torch.ones_like(constraint), retain_graph=True)[0]
            if constraint_grad is None:
                continue
            params_with_grad.append(p)
            if constraint_grad.is_sparse:
                raise RuntimeError("PhysicsInformedAdamW does not support sparse gradients")
            grads.append(constraint_grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((1,), dtype=torch.float, device=p.device)
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0)
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if amsgrad:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            state_steps.append(state["step"])
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