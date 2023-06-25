import torch
from tableau import ButcherTableau


def explicit_rk_step(
    y0: torch.Tensor, # [b, ...]
    t0: torch.Tensor,
    f,
    dt,
    tableau: ButcherTableau,
    return_increments=False,
):
    y0_shape = y0.shape
    y0_dim = y0.dim()

    a = tableau.a
    b = tableau.b
    c = tableau.c
    s = tableau.s

    k = torch.empty((*y0_shape, s), dtype=y0.dtype, device=y0.device)
    k[..., 0] = f(t0 + c[0] * dt, y0)
    
    for i in range(1, s):
        as_current = a[i, :i]
        for _ in range(y0_dim):
            as_current = as_current.unsqueeze(0)
        increment_sum = torch.sum(
            as_current * k[..., :i],
            dim=-1,
        )    
        k[..., i] = f(t0 + c[i] * dt, y0 + dt * increment_sum)
    
    for _ in range(y0_dim):
        b = b.unsqueeze(0)
    return y0 + dt * torch.sum(b * k, dim=-1), k if return_increments else None


def rk_solve(y0, t0, t1, f, dt, tableau, return_all_states=False):
    t = t0
    y = torch.clone(y0)
    states = []
    times = []
    while t < t1:
        if return_all_states:
            times.append(t.detach().cpu().numpy())
            states.append(y.detach().cpu().numpy())
        y, _ = explicit_rk_step(y, t, f, dt, tableau)
        t = t + dt

    if return_all_states:
        times.append(t.detach().cpu().numpy())
        states.append(y.detach().cpu().numpy())
    return y, times, states


def rk_adaptive_embedded(y0, t0, t1, dt, f, tableau_low, tableau_high, return_all_states=False, atol=1e-6, rtol=1e-6):
    t = t0
    y = torch.clone(y0)
    states = []
    times = []
    atol = torch.tensor(atol, dtype=y0.dtype, device=y0.device)

    b_high = tableau_high.b
    for _ in range(y0.dim()):
        b_high = b_high.unsqueeze(0)


    if return_all_states:
        times.append(t.detach().cpu().numpy())
        states.append(y.detach().cpu().numpy())
    while t < t1:
        y_low, increments = explicit_rk_step(y, t, f, dt, tableau_low, return_increments=True)
        y_high = y + dt * torch.sum(b_high * increments, dim=-1)
        error_estimate = (y_low - y_high).flatten(1, -1).norm(dim=-1)
        y_norm = y.flatten(1, -1).norm(dim=-1)
        tolerance = torch.maximum(rtol * y_norm, atol)
        if torch.any(error_estimate > tolerance):
            # reject step
            dt = dt / 2.0
        else:
            # accept step
            t += dt
            y = y_high
            if return_all_states:       
                times.append(t.detach().cpu().numpy())
                states.append(y.detach().cpu().numpy())
            dt = 1.1 * dt
            if t + dt > t1:
                dt = t1 - t + 1e-10
    return y, times, states
            
        