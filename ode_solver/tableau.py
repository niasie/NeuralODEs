import torch


class ButcherTableau():
    def __init__(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        self.a = a
        self.b = b
        self.c = c
        self.s = c.shape[0]


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ExplicitEuler(ButcherTableau):
    def __init__(self):
        super().__init__(
            torch.tensor([[0]], dtype=torch.float32, device=device),
            torch.tensor([1], dtype=torch.float32, device=device),
            torch.tensor([0], dtype=torch.float32, device=device),
        )


class ExplicitTrapezoidal(ButcherTableau):
    def __init__(self):
        super().__init__(
            torch.tensor([[0.0, 0.0],
                        [1.0, 0.0]], dtype=torch.float32, device=device),
            torch.tensor([0.0, 1.0], dtype=torch.float32, device=device),
            torch.tensor([0.5, 0.5], dtype=torch.float32, device=device),
        )


class ExplicitMidpoint(ButcherTableau):
    def __init__(self):
        super().__init__(
            torch.tensor([[0.0, 0.0],
                        [0.5, 0.0]], dtype=torch.float32, device=device),
            torch.tensor([0.0, 0.5], dtype=torch.float32, device=device),
            torch.tensor([0.0, 1.0], dtype=torch.float32, device=device),
        )
