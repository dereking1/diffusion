import torch
from tqdm import tqdm

def Euler_Maruyama_sampler(score_model,
                           sde,
                           batch_size, 
                           num_steps=1000, 
                           device='cuda', 
                           eps=1e-3):
    # TODO: compute std at t=1 
    pass
    pass
    std = sde.diffusion(1)

    # TODO: sample a batch of x at t=1
    init_x = torch.randn_like(torch.zeros(batch_size, 1, 28, 28), device=device)

    # TODO: create a sequence of time_steps from 1 to very smoll
    time_steps = torch.linspace(1, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # TODO: the magic! 
    x = init_x
    with torch.no_grad():
        for i,time_step in enumerate(tqdm(time_steps)):  
            # TODO
            t = time_step * torch.ones(batch_size, device=device)
            f = sde.drift(x, t)
            g = sde.diffusion(t)
            g2 = torch.square(g)
            z = torch.randn_like(x, device=device)
            x_ = x - f*step_size + g2[:,None,None,None]*score_model(x, t)*step_size
            if i <= len(time_steps) - 2:
              x_ += g[:,None,None,None]*torch.sqrt(step_size)*z
            x = x_
    # Do not include any noise in the last sampling step.
    return x_

def predictor_corrector_sampler(score_model, 
                                sde,
                                batch_size,
                                num_steps=1000,
                                device='cuda',
                                snr=0.16,
                                num_corrector_steps=1,
                                eps=1e-3):
    # TODO: compute std at t=1 
    pass
    pass
    std = sde.diffusion(1)

    # TODO: sample a batch of x at t=1
    init_x = torch.randn_like(torch.zeros(batch_size, 1, 28, 28), device=device)

    # TODO: create a sequence of time_steps from 1 to very smoll
    time_steps = torch.linspace(1, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # TODO: the magic!
    x = init_x
    with torch.no_grad():
        for i,time_step in enumerate(tqdm(time_steps)):
            # TODO: setup
            t = time_step * torch.ones(batch_size, device=device)

            # Corrector step (Langevin MCMC - alorithm 5 in [Song21])
            for j in range(num_corrector_steps):
                z = torch.randn_like(x, device=device)
                g = score_model(x, t)
                z_norm = torch.norm(z.reshape(z.shape[0], -1), dim=-1).mean()
                g_norm = torch.norm(g.reshape(g.shape[0], -1), dim=-1).mean()
                e = 2 * sde.alphas[i] * torch.square(snr * z_norm/g_norm)
                x_ = x + e*g
                if i <= len(time_steps) - 2 or j <= num_corrector_steps - 2:
                    x_+= torch.sqrt(2*e)*z
            # Predictor step (Euler-Maruyama)
            x = x_

    # Do not include any noise in the last sampling step.
    return x_