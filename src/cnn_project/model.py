import io
from torch.utils.benchmark import Timer

import torch
import torch.nn as nn

class EuroSATModel(nn.Module):

    def __init__(self, channels = (16,32,64,128)):
        super(EuroSATModel, self).__init__()

        c1, c2, c3, c4 = channels

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=c3, out_channels=c4, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(c4*4*4,10)
        )
    def forward(self,x):
        return self.model(x)


def serialized_state_dict_size_mib(model):
    """Return the torch.save state dictionary size in MiB (1 MiB = 1024**2 bytes).

    Includes stored tensors, metadata, and serialization overhead. Measures
    checkpoint storage, not a complete exported program or inference memory.
    """
    with io.BytesIO() as buffer:
        torch.save(model.state_dict(), buffer)
        return buffer.tell() / 1024**2


def model_tensor_size_mib(model):
    """Return registered parameter and buffer tensor data size in MiB.

    Includes frozen parameters and registered INT8 weight buffers in PT2E graphs.
    Excludes serialization/Python overhead, inputs, activations, workspace,
    gradients, and optimizer state; this is not peak inference or training memory.
    Counts logical tensor elements, not unique underlying storage: distinct
    tensors sharing storage can be counted twice. Packed weights outside registered
    parameters and buffers are not included.
    """
    parameter_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    buffer_bytes = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )
    return (parameter_bytes + buffer_bytes) / 1024**2

def measure_inference_time_ms(model, example_input, num_threads=1):
    """Return median forward-pass time in milliseconds per batch.

    Measures CPU execution, excluding data loading and preprocessing.
    Warm-up runs exclude initial compilation from the measurement.
    Pass a model already configured for inference and an input on CPU.
    """
    with torch.inference_mode():
        for _ in range(10):
            model(example_input)

        measurement = Timer(
            stmt="model(example_input)",
            globals={"model": model, "example_input": example_input},
            num_threads=num_threads,
        ).blocked_autorange(min_run_time=2.0)

    return measurement.median * 1000

# Test the output shape of the model

# model=EuroSATModel()
# output= model(torch.randn(1,3,64,64))
# print(output.shape)
