from torch import Tensor, exp, log, nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.functional import linear, hardtanh
from .nac import NacCell


# Source from https://github.com/bharathgs/NALU
class NaluCell(nn.Module):
    """Basic NALU unit implementation
    from https://arxiv.org/pdf/1808.00508.pdf
    """

    def __init__(self, in_shape, out_shape):
        """
        in_shape: input sample dimension
        out_shape: output sample dimension
        """
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.G = Parameter(Tensor(out_shape, in_shape))
        self.nac = NacCell(in_shape, out_shape)

        xavier_uniform_(self.G)
        self.eps = 1e-7
        self.register_parameter('bias', None)

    def forward(self, x):
        g = linear(x, self.G, self.bias).sigmoid()
        #g = hardtanh(linear(x, self.G, self.bias), 0., 1.)
        a = self.nac(x)
        ag = a * g

        log_in = self.nac(log(abs(x) + self.eps))
        #log_in = log_in.clamp(-32, 16)
        #log_in = hardtanh(log_in, -32., 16.)
        m = exp(log_in)
        md = m * (1 - g)

        return ag + md
