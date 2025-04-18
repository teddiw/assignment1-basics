from collections.abc import Callable, Iterable
from typing import Optional, Tuple
import torch
import math
import numpy as np


class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 lr: float,
                 betas: Tuple[float],
                 weight_decay: float, 
                 eps: float = 1e-8
                 ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {'lr': lr, 'betas':betas, 'weight_decay': weight_decay, 'eps': eps}
        super().__init__(params, defaults)

    def step(self, scheduled_lr: int | None, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            if (scheduled_lr):
                lr = scheduled_lr
            else:
                lr = group['lr']
            betas = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p] 
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))
                t = state.get("t", 1)
                grad = p.grad.data
                state['m'] = betas[0]*m + (1-betas[0])*grad
                state['v'] = betas[1]*v + (1-betas[1])*grad**2

                lr_t = lr*math.sqrt(1-betas[1]**t)/(1-betas[0]**t)
                p.data -= lr_t*state['m']/(torch.sqrt(state['v'])+eps)
                p.data -= lr*weight_decay*p.data

                state["t"] = t + 1
        return loss
    

def learning_rate_scheduling(t: int,
                             a_max: float,
                             a_min: float,
                             T_w: int,
                             T_c: int,
                             ) -> float:
    if (t < T_w):
        return t*a_max/T_w
    elif (t <= T_c):
        return a_min + .5*(1 + math.cos((t - T_w)*math.pi / (T_c - T_w)))*(a_max - a_min)
    else:
        return a_min
    
def gradient_clipping(params: list[torch.Tensor],
                     max_norm: float,
                     eps: float = 1e-6,
                     ) -> None:
    temp_squared_sum = 0
    for p in params:
        if p.grad is None:
            continue
        temp_squared_sum += torch.sum(p.grad**2)
    
    l2_norm = math.sqrt(temp_squared_sum) 

    if (l2_norm >= max_norm):
        scale_factor = max_norm / (l2_norm + eps)
        for p in params:
            if p.grad is None:
                continue
            p.grad.data *= scale_factor
    return l2_norm
    


    
if __name__ == "__main__":
    weights = torch.nn.Parameter(5*torch.randn((10,10)))
    opt = AdamW([weights], 
                lr=1e3,
                betas = (.9, .999),
                weight_decay=0.01,
                eps=1e-8
                )

    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()