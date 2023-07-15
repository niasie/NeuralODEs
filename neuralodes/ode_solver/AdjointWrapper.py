import torch
import numpy as np

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
            z1, times, states = solver(f, z0, t0, t1, dt)

            ctx.save_for_backward(z0, t0, t1, dt, z1, *f_params)
        
        return z1, times, states

    @staticmethod
    def backward(ctx, dL_dz1, dtimes_dz1, dstates_dz1):
        with torch.no_grad():
            z0, t0, t1, dt, z1, *f_params = ctx.saved_tensors

            solver = ctx.solver
            f = ctx.f

            s0 = []
            s0.append(z1.clone().detach())
            s0.append(dL_dz1.clone().detach())

            for param in f_params:
                s0.append(torch.zeros_like(param))

            def flatten_state(s):
                vectors = []
                shapes = []

                for tensor in s:
                    vectors.append(tensor.reshape(-1))
                    shapes.append(tensor.shape)

                return torch.cat(vectors), shapes

            def reconstruct_state(s_flat, shapes):
                s = []
                idx = 0

                for shape in shapes:
                    num_elements = np.prod(shape, dtype=np.int32)
                    tensor = s_flat[idx:idx+num_elements].reshape(*shape)

                    s.append(tensor)

                    idx += num_elements
                
                return s
            

            def augmented_dynamics(t, s_flat):
                # Convention for s:
                # s[0] = z(t)
                # s[1] = a(t) := dL_dz(t)
                # s[2:] = *

                s = reconstruct_state(s_flat, shapes)

                with torch.enable_grad():
                    t = t.clone().detach()
                    zt = s[0].clone().detach().requires_grad_()

                    aug = [None] * len(s)
                    f_eval = f(t, zt)
                    
                    aug[0] = f_eval
                    aug[1], *aug[2:] = torch.autograd.grad(
                        f_eval, (zt, *f_params), -s[1], allow_unused=True, retain_graph=True)
            
                    for i, elem in enumerate(aug):
                        if elem is None:
                            aug[i] = torch.zeros_like(s[i])

                # return f(z(t), t), -a^T df_dz(t), -a^T df_dparam(t)

                return flatten_state(aug)[0]

            s0_flat, shapes = flatten_state(s0)
            s1_flat, _, _ = solver(augmented_dynamics, s0_flat, t1, t0, dt)
            s1 = reconstruct_state(s1_flat, shapes)
        
            z0_backwards = s1[0]
            dL_dz0 = s1[1]
            dL_dparam = s1[2:]

        return (None, None, dL_dz0, None, None, None, *dL_dparam)