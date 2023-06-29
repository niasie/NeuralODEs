import torch

class AdjointWrapper(torch.autograd.Function):
    """
    Wraps a black-box ODE integrator to be compatible with Pytorch Autograd
    by using the adjoint sensitivity method.
    """

    @staticmethod
    def forward(ctx, solver, f, z0, t0, t1, solver_kwargs, *f_params):
        ctx.solver = solver
        ctx.f = f
        ctx.solver_kwargs = solver_kwargs

        with torch.no_grad():
            z1 = solver(f, z0, t0, t1, solver_kwargs)[0, -1]

            ctx.save_for_backward(z0, t0, t1, z1, *f_params)
        
        return z1

    @staticmethod
    def backward(ctx, dL_dz1):
        with torch.no_grad():
            z0, t0, t1, z1, *f_params = ctx.saved_tensors

            solver = ctx.solver
            f = ctx.f
            solver_kwargs = ctx.solver_kwargs

            s0 = []
            s0.append(f(z1, t1))
            s0.append(dL_dz1.clone().detach())

            for param in f_params:
                s0.append(torch.zeros_like(param))

            def augmented_dynamics(t, s):
                # Convention for s:
                # s[0] = z(t)
                # s[1] = a(t) := dL_dz(t)
                # s[2:] = *

                with torch.enable_grad():
                    t, s = torch.tensor(t), torch.tensor(s).detach().requires_grad_()

                    aug = [None] * s.shape[0]
                    f_eval = f(t, s[0])
                    
                    aug[0] = f_eval
                    aug[1], *aug[2:] = torch.autograd.grad(
                        f_eval, (s[0], *f_params), -s[1], allow_unused=True, retain_graph=True)
            
                    for i, grad in enumerate(aug):
                        if grad is None:
                            aug[i] = torch.zeros_like(s[i])

                # return f(z(t), t), df_dz(t), df_dparam(t)
                return aug[0], aug[1], *aug[2:]

            s1 = solver(augmented_dynamics, s0, t1, t0, solver_kwargs)
        
            z0_backwards = s1[0, -1]
            dL_dz0 = s1[1, -1].unsqueeze(0)
            dL_dparam = s1[2:, -1].unsqueeze(0)

        return (None, None, dL_dz0, None, None, None, *dL_dparam)


def adjoint_wrapper(solver, f, f_params, y0, t0, t1, solver_kwargs={}):
    result = AdjointWrapper.apply(
        solver, f, y0, t0, t1, solver_kwargs, *f_params)
    return result
