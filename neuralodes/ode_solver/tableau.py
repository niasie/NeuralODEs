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
            torch.tensor([0.0, 1.0], dtype=torch.float32, device=device),
            torch.tensor([0.0, 0.5], dtype=torch.float32, device=device),
        )


class ClassicalRK4(ButcherTableau):
    def __init__(self):
        super().__init__(
            torch.tensor([[0.0, 0.0, 0.0, 0.0],
                          [0.5, 0.0, 0.0, 0.0],
                          [0.0, 0.5, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=device),
            torch.tensor([1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0], dtype=torch.float32, device=device),
            torch.tensor([0.0, 0.5, 0.5, 1.0], dtype=torch.float32, device=device),
        )


class Kuttas38Method(ButcherTableau):
    def __init__(self):
        super().__init__(
            torch.tensor([[0.0, 0.0, 0.0, 0.0],
                          [1.0 / 3.0, 0.0, 0.0, 0.0],
                          [-1.0 / 3.0, 1.0, 0.0, 0.0],
                          [1.0, -1.0, 1.0, 0.0]], dtype=torch.float32, device=device),
            torch.tensor([1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 6.0], dtype=torch.float32, device=device),
            torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=torch.float32, device=device),
        )


class Fehlberg5(ButcherTableau):
    def __init__(self):
        super().__init__(
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0],
                        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296 / 2197.0, 0.0, 0.0, 0.0],
                        [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0, 0.0],
                        [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0.0]],
                        dtype=torch.float32, device=device,
            ),
            torch.tensor([16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0], dtype=torch.float32, device=device),
            torch.tensor([0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0], dtype=torch.float32, device=device),
        )


class Fehlberg4(ButcherTableau):
    def __init__(self):
        super().__init__(
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0],
                        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296 / 2197.0, 0.0, 0.0, 0.0],
                        [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0, 0.0],
                        [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0.0]],
                        dtype=torch.float32, device=device,
            ),
            torch.tensor([25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0], dtype=torch.float32, device=device),
            torch.tensor([0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0], dtype=torch.float32, device=device),
        )