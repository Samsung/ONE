import torch


class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(4, 16, 5, padding=(2, 2))
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 4, 3, padding=(1, 1))
        self.relu2 = torch.nn.ReLU()

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = torch.split(output, 2, dim=1)
        return output


model = TestModel()
torch.save(model, "model.pth")

input = torch.randn(1, 4, 16, 16)
torch.onnx.export(
    model,
    input,
    "onnx_conv2d_conv2d_split.onnx",
    verbose=True,
    input_names=["input"],
    output_names=["out1", "out2"])
