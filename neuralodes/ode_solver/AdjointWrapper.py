import torch

class AdjointWrapper(torch.autograd.Function):
    """
    Wraps a black-box ODE integrator to be compatible with Pytorch Autograd
    by using the adjoint sensitivity method.
    """

    @staticmethod
    def forward(ctx, solver, f, z0, t0, t1, dt, *f_params):
        ctx.solver = solver
        ctx.f = f

        with torch.no_grad():
            z1, _, _ = solver(f, z0, t0, t1, dt)
            z1 = z1[-1]

            ctx.save_for_backward(z0, t0, t1, dt, z1, *f_params)
        
        return z1

    @staticmethod
    def backward(ctx, dL_dz1):
        with torch.no_grad():
            z0, t0, t1, dt, z1, *f_params = ctx.saved_tensors

            solver = ctx.solver
            f = ctx.f

            s0 = []
            s0.append(z1.clone().detach())
            s0.append(dL_dz1.clone().detach())

            for param in f_params:
                s0.append(torch.zeros_like(param))

            def augmented_dynamics(t, s):
                # Convention for s:
                # s[0] = z(t)
                # s[1] = a(t) := dL_dz(t)
                # s[2:] = *

                with torch.enable_grad():
                    t, s = torch.tensor(t), torch.tensor(s)
                    zt = s[0].clone().detach().requires_grad_()

                    aug = [None] * s.shape[0]
                    f_eval = f(t, zt)
                    
                    aug[0] = f_eval
                    aug[1], *aug[2:] = torch.autograd.grad(
                        f_eval, (zt, *f_params), -s[1], allow_unused=True, retain_graph=True)
            
                    for i, elem in enumerate(aug):
                        if elem is None:
                            aug[i] = torch.zeros_like(s[i])

                # return f(z(t), t), -a^T df_dz(t), -a^T df_dparam(t)
                return torch.tensor(aug)

            s1, _, _ = solver(augmented_dynamics, torch.tensor(s0), t1, t0, dt)
        
            z0_backwards = s1[0]
            dL_dz0 = s1[1].unsqueeze(0)
            dL_dparam = s1[2:].unsqueeze(0)

        return (None, None, dL_dz0, None, None, None, *dL_dparam)


def adjoint_wrapper(solver, f, f_params, y0, t0, t1, dt):
    result = AdjointWrapper.apply(
        solver, f, y0, t0, t1, dt, *f_params)
    return result
