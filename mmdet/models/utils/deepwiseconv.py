from torch import nn


class DeepWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, ksize=3, padding=1, bais=True):
        super(DeepWiseConv, self).__init__()

        self.depthwiseConv = nn.Conv2d(in_channels=in_channel,
                                       out_channels=in_channel,
                                       groups=in_channel,
                                       kernel_size=ksize,
                                       padding=padding,
                                       bias=bais)
        self.pointwiseConv = nn.Conv2d(in_channels=in_channel,
                                       out_channels=out_channel,
                                       kernel_size=1,
                                       padding=0,
                                       bias=bais)

    def forward(self, x):
        out = self.depthwiseConv(x)
        out = self.pointwiseConv(out)
        return out
