from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class BatchLinearLayer(nn.Module):
    def __init__(self, linears: List[nn.Linear]):
        super().__init__()
        for linear in linears:
            assert linear.in_features == linears[0].in_features
            assert linear.out_features == linears[0].out_features

        len(linears)
        linears[0].in_features
        linears[0].out_features

        self.register_buffer("weight", torch.stack([linear.weight.data for linear in linears]))
        self.register_buffer("bias", torch.stack([linear.bias.data for linear in linears]))

    def forward(self, x: Tensor) -> Tensor:
        output = torch.einsum("bni,noi->bno", x, self.weight)
        output = output + self.bias.unsqueeze(0)
        return output


class BatchBatchNorm1d(nn.Module):
    def __init__(self, batch_norms: List[nn.BatchNorm1d]):
        super().__init__()
        # Validate input batch norms
        for bn in batch_norms:
            assert (
                bn.num_features == batch_norms[0].num_features
            ), "All batch norms must have same number of features"
            assert bn.eps == batch_norms[0].eps, "All batch norms must have same eps value"

        len(batch_norms)
        batch_norms[0].num_features

        self.register_buffer("weight", torch.stack([bn.weight.data for bn in batch_norms]))
        self.register_buffer("bias", torch.stack([bn.bias.data for bn in batch_norms]))

        self.register_buffer(
            "running_mean", torch.stack([bn.running_mean.data for bn in batch_norms]).unsqueeze(0)
        )
        self.register_buffer(
            "running_var", torch.stack([bn.running_var.data for bn in batch_norms]).unsqueeze(0)
        )
        self.eps = batch_norms[0].eps

    def forward(self, x: Tensor) -> Tensor:
        out = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        out = out * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return out

    def cuda(self):
        # Note: This method might not be necessary as register_buffer handles
        # device movement automatically when calling .cuda() on the module
        return super().cuda()


class BatchFCN(nn.Module):
    def __init__(self, fcns: List[nn.Sequential]):
        super().__init__()
        batch_layers = []
        for i in range(len(fcns[0])):
            layer = fcns[0][i]
            layers = [fcn[i] for fcn in fcns]
            if isinstance(layer, nn.Linear):
                batch_layers.append(BatchLinearLayer(layers))
            elif isinstance(layer, nn.BatchNorm1d):
                # pass
                batch_layers.append(BatchBatchNorm1d(layers))
            elif isinstance(layer, nn.ReLU):
                batch_layers.append(nn.ReLU())
            else:
                raise ValueError(f"Layer {layer} not supported")
        self.batch_layers = nn.Sequential(*batch_layers)
        self.n_fcn = len(fcns)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            out = x.unsqueeze(1).expand(-1, self.n_fcn, -1)
        else:
            out = x
        for layer in self.batch_layers:
            out = layer(out)
        return out

    def cuda(self):
        for layer in self.batch_layers:
            layer.cuda()
        return super().cuda()


if __name__ == "__main__":

    real = True
    if real:
        from hifuku.core import SolutionLibrary
        from hifuku.domain import FetchConwayJailInsert
        from hifuku.script_utils import load_library

        lib: SolutionLibrary = load_library(FetchConwayJailInsert, "cuda", postfix="0.2")
        fncs = [pred.linears for pred in lib.predictors][1:]
        dummy_input = torch.ones(1, 250).cuda()

        # naive
        outputs = []
        for fcn in fncs:
            # remove batch norm
            # fcn_removed = nn.Sequential(*[layer for layer in fcn if not isinstance(layer, nn.BatchNorm1d)])
            fcn_removed = fcn
            output = fcn_removed(dummy_input)
            outputs.append(output)
        output = torch.stack(outputs, dim=1)
        print(output)

        fcn = BatchFCN(fncs).cuda()
        output2 = fcn(dummy_input)
        print(output2)
        print(output - output2)
    else:
        linears = [nn.Linear(100, 3) for _ in range(10)]
        batch_linear = BatchLinearLayer(linears)

        dummy_input = torch.randn(2, 100)

        # batch
        dummy_input_expanded = dummy_input.unsqueeze(1).expand(-1, 10, -1)
        output = batch_linear(dummy_input_expanded)
        print(output.shape)

        # naive
        outputs = []
        for linear in linears:
            output = linear(dummy_input)
            outputs.append(output)
        output = torch.stack(outputs, dim=1)
        # sum
        output = output.sum(dim=1)
        print(output)
