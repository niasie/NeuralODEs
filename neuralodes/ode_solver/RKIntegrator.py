import torch
from .tableau import ButcherTableau

def rk_step(
    y0: torch.Tensor, # [b, ...]
    t0: torch.Tensor,
    dt,
    f,
    tableau: ButcherTableau,
    return_increments=False,
):
    if torch.norm(torch.triu(tableau.a)) < 1e-7:
        return explicit_rk_step(y0, t0, dt, f, tableau, return_increments)
    else:
        return implicit_rk_step(y0, t0, dt, f, tableau, return_increments)

def explicit_rk_step(
    y0: torch.Tensor, # [b, ...]
    t0: torch.Tensor,
    dt,
    f,
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
    dt = dt.to(y0.device)

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
    k = k.to(y0.device)
    b = b.to(y0.device)
    return y0 + dt * torch.sum(b * k, dim=-1), k if return_increments else None

def implicit_rk_step(
    y0: torch.Tensor,
    t0: torch.Tensor,
    dt,
    f,
    tableau: ButcherTableau,
    return_increments=False
):
    y0_shape = y0.shape
    y0_dim = y0.dim()

    a = tableau.a
    b = tableau.b
    c = tableau.c
    s = tableau.s

    k = torch.nn.Parameter(data=torch.zeros((*y0_shape, s), dtype=y0.dtype, device=y0.device))

    optim = torch.optim.LBFGS(
        [y0], 
        lr=1., 
        max_iter=200, 
        max_eval=2000, 
        tolerance_grad=1e-12, 
        tolerance_change=1e-12, 
        history_size=100, 
        line_search_fn='strong_wolfe'
    )

    def closure():
        optim.zero_grad()

        residual = torch.tensor(0.).to(y0.device, y0.dtype)

        for i in range(s):
            residual += torch.nn.functional.mse_loss(k[..., i], f(t0 + c[i] * dt, y0 + dt * torch.tensordot(k, a[:, i], dims=([-1], [0]))))
        
        return residual
    
    optim.step(closure)

    y1 = y0 + dt * torch.tensordot(k, b, dims=([-1], [0]))

    return y1, k.data if return_increments else None

def rk_solve(f, y0_, t0_, t1_, dt_, tableau, return_all_states=False):
    t = torch.clone(t0_)
    dt = torch.clone(dt_)
    y = torch.clone(y0_)
    t1 = torch.clone(t1_)
    y = torch.clone(y0_)
    integrate_forward = t1_ > t0_
    if not integrate_forward and dt > 0:
        dt = -dt

    states = []
    times = []
    while not reached_t1(t, t1, integrate_forward):
        if return_all_states:
            times.append(t.detach().cpu().numpy())
            states.append(y.detach().cpu().numpy())

        y, _ = rk_step(y, t, dt, f, tableau)
        
        t = t + dt

    if return_all_states:
        times.append(t.detach().cpu().numpy())
        states.append(y.detach().cpu().numpy())
    return y, times, states


def rk_adaptive(y0_, t0_, t1_, dt_, f, tableau_low, tableau_high, return_all_states=False, atol=1e-6, rtol=1e-6):
    t = torch.clone(t0_)
    dt = torch.clone(dt_)
    y = torch.clone(y0_)
    t1 = torch.clone(t1_)
    states = []
    times = []
    atol = torch.tensor(atol, dtype=y.dtype, device=y.device)

    integrate_forward = t1_ > t0_
    if not integrate_forward and dt > 0:
        dt = -dt

    if return_all_states:
        times.append(t.detach().cpu().numpy())
        states.append(y.detach().cpu().numpy())

    while not reached_t1(t, t1, integrate_forward):
        y_high, _ = rk_step(y, t, dt, f, tableau_high)
        with torch.no_grad():
            y_low, _ = rk_step(y, t, dt, f, tableau_low)
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
                if reached_t1(t + dt, t1, integrate_forward):
                    if integrate_forward:
                        dt = t1 - t + 1e-10
                    else:
                        dt = t1 - t - 1e-10
    return y, times, states


def rk_adaptive_embedded(y0_, t0_, t1_, dt_, f, tableau_low, tableau_high, return_all_states=False, atol=1e-6, rtol=1e-6):
    t = torch.clone(t0_)
    dt = torch.clone(dt_)
    y = torch.clone(y0_)
    t1 = torch.clone(t1_)
    states = []
    times = []
    atol = torch.tensor(atol, dtype=y.dtype, device=y.device)

    integrate_forward = t1_ > t0_
    if not integrate_forward and dt > 0:
        dt = -dt

    b_high = tableau_high.b
    for _ in range(y.dim()):
        b_high = b_high.unsqueeze(0)


    if return_all_states:
        times.append(t.detach().cpu().numpy())
        states.append(y.detach().cpu().numpy())

    while not reached_t1(t, t1, integrate_forward):
        y_low, increments = rk_step(y, t, dt, f, tableau_low, return_increments=True)
        y_high = y + dt * torch.sum(b_high * increments, dim=-1)
        with torch.no_grad():
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
                if reached_t1(t + dt, t1, integrate_forward):
                    if integrate_forward:
                        dt = t1 - t + 1e-10
                    else:
                        dt = t1 - t - 1e-10
    return y, times, states
            

def reached_t1(t, t1, is_forward):
    if is_forward:
        return t >= t1
    else:
        return t <= t1