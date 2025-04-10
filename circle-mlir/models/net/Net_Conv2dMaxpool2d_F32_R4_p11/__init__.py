import torch


class net_Conv2dMaxpool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.Conv2d(3, 8, (3, 3), padding=(1, 1))
        self.op2 = torch.nn.MaxPool2d(2)
        self.op3 = torch.nn.Conv2d(8, 8, (3, 3), padding=(1, 1))
        self.op4 = torch.nn.MaxPool2d(2)

    def forward(self, input):
        r1 = self.op1(input)
        r2 = self.op2(r1)
        r3 = self.op3(r2)
        return self.op4(r3)


_model_ = net_Conv2dMaxpool2d()

_inputs_ = torch.randn(1, 3, 32, 32)
