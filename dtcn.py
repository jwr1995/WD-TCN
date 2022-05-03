
""" 
Lots of this code is adapted from the SpeechBrain Conv-TasNet implementation:
> https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/conv_tasnet.py

The MaskNet class is configured the same as the MaskNet class in the above link
2020 SpeechBrain
2022 William Ravenscroft
"""
import torch
import torch.nn as nn
import speechbrain as sb
import torch.nn.functional as F

from speechbrain.processing.signal_processing import overlap_and_add
from speechbrain.lobes.models.conv_tasnet import GlobalLayerNorm, ChannelwiseLayerNorm, Chomp1d, choose_norm
from speechbrain.nnet.CNN import Conv1d

EPS = 1e-8

class DynamicTemporalBlocksSequential(sb.nnet.containers.Sequential):
    """
    A wrapper for the temporal-block layer to replicate it

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    H : int
        The number of intermediate channels.
    P : int
        The kernel size in the convolutions.
    R : int
        The number of times to replicate the multilayer Temporal Blocks.
    X : int
        The number of layers of Temporal Blocks with different dilations.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example
    -------
    >>> x = torch.randn(14, 100, 10)
    >>> H, P, R, X = 10, 5, 2, 3
    >>> DynamicTemporalBlocks = DynamicTemporalBlocksSequential(
    ...     x.shape, H, P, R, X, 'gLN', False
    ... )
    >>> y = DynamicTemporalBlocks(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self, 
        input_shape, 
        H, 
        P, 
        R, 
        X, 
        norm_type, 
        causal,
        se_kernel_size=20,
        bias=True,
        pool="global",
        attention_type="se",
        num_heads=4
        ):
        super().__init__(input_shape=input_shape)
        for r in range(R):
            for x in range(X):
                dilation = 2 ** x
                self.append(
                    DynamicTemporalBlock,
                    out_channels=H,
                    kernel_size=P,
                    stride=1,
                    padding="same",
                    dilation=dilation,
                    norm_type=norm_type,
                    causal=causal,
                    layer_name=f"temporalblock_{r}_{x}",
                    se_kernel_size=se_kernel_size,
                    bias=bias,
                    pool=pool,
                    attention_type=attention_type,
                    num_heads=num_heads
                )


class MaskNet(nn.Module):
    """
    Arguments
    ---------
    N : int
        Number of filters in autoencoder.
    B : int
        Number of channels in bottleneck 1 × 1-conv block.
    H : int
        Number of channels in convolutional blocks.
    P : int
        Kernel size in convolutional blocks.
    X : int
        Number of convolutional blocks in each repeat.
    R : int
        Number of repeats.
    C : int
        Number of speakers.
    norm_type : str
        One of BN, gLN, cLN.
    causal : bool
        Causal or non-causal.
    mask_nonlinear : str
        Use which non-linear function to generate mask, in ['softmax', 'relu'].

    Example:
    ---------
    >>> N, B, H, P, X, R, C = 11, 12, 2, 5, 3, 1, 2
    >>> MaskNet = MaskNet(N, B, H, P, X, R, C)
    >>> mixture_w = torch.randn(10, 11, 100)
    >>> est_mask = MaskNet(mixture_w)
    >>> est_mask.shape
    torch.Size([2, 10, 11, 100])
    """

    def __init__(
        self,
        N,
        B,
        H,
        P,
        X,
        R,
        C,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
        se_kernel_size=20,
        bias=True,
        pool="global",
        attention_type="se",
        num_heads=4,
    ):
        super(MaskNet, self).__init__()

        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear

        # Components
        # [M, K, N] -> [M, K, N]
        self.layer_norm = ChannelwiseLayerNorm(N)

        # [M, K, N] -> [M, K, B]
        self.bottleneck_conv1x1 = sb.nnet.CNN.Conv1d(
            in_channels=N, out_channels=B, kernel_size=1, bias=False,
        )

        # [M, K, B] -> [M, K, B]
        in_shape = (None, None, B)
        self.temporal_conv_net = DynamicTemporalBlocksSequential(
            in_shape, 
            H, 
            P, 
            R, 
            X, 
            norm_type, 
            causal,
            se_kernel_size=20,
            bias=True,
            pool="global",
            attention_type=attention_type,
            num_heads=num_heads
        )

        # [M, K, B] -> [M, K, C*N]
        self.mask_conv1x1 = sb.nnet.CNN.Conv1d(
            in_channels=B, out_channels=C * N, kernel_size=1, bias=False
        )

    def forward(self, mixture_w):
        """Keep this API same with TasNet.

        Arguments
        ---------
        mixture_w : Tensor
            Tensor shape is [M, K, N], M is batch size.

        Returns
        -------
        est_mask : Tensor
            Tensor shape is [M, K, C, N].
        """

        mixture_w = mixture_w.permute(0, 2, 1)
        M, K, N = mixture_w.size()
        y = self.layer_norm(mixture_w)
        y = self.bottleneck_conv1x1(y)
        y = self.temporal_conv_net(y)
        score = self.mask_conv1x1(y)

        # score = self.network(mixture_w)  # [M, K, N] -> [M, K, C*N]
        score = score.contiguous().reshape(
            M, K, self.C, N
        )  # [M, K, C*N] -> [M, K, C, N]

        # [M, K, C, N] -> [C, M, N, K]
        score = score.permute(2, 0, 3, 1)

        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=2)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class DynamicTemporalBlock(torch.nn.Module):
    """The conv1d compound layers used in Masknet.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input.
    out_channels : int
        The number of intermediate channels.
    kernel_size : int
        The kernel size in the convolutions.
    stride : int
        Convolution stride in convolutional layers.
    padding : str
        The type of padding in the convolutional layers,
        (same, valid, causal). If "valid", no padding is performed.
    dilation : int
        Amount of dilation in convolutional layers.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example:
    ---------
    >>> x = torch.randn(14, 100, 10)
    >>> DynamicTemporalBlock = DynamicTemporalBlock(x.shape, 10, 11, 1, 'same', 1)
    >>> y = DynamicTemporalBlock(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding="same",
        norm_type="gLN",
        causal=False,
        se_kernel_size=20,
        bias=True,
        pool="global",
        attention_type="se",
        num_heads=4
    ):
        super().__init__()
        M, K, B = input_shape # batch x time x features

        self.layers = sb.nnet.containers.Sequential(input_shape=input_shape)

        # [M, K, B] -> [M, K, H]
        self.layers.append(
            sb.nnet.CNN.Conv1d,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            layer_name="conv",
        )
        self.layers.append(nn.PReLU(), layer_name="act")
        self.layers.append(
            choose_norm(norm_type, out_channels), layer_name="norm"
        )

        # [M, K, H] -> [M, K, B]
        self.layers.append(
            DynamicDepthwiseSeparableConv,
            out_channels=B,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm_type=norm_type,
            causal=causal,
            se_kernel_size=se_kernel_size,
            bias=bias,
            pool=pool,
            layer_name="DSconv",       
            attention_type=attention_type,
            num_heads=num_heads     
        )

    def forward(self, x):
        """
        Arguments
        ---------
        x : Tensor
            Tensor shape is [batch size, sequence length, input channels].

        Returns
        -------
        x : Tensor
            Tensor shape is [M, K, B].
        """
        residual = x
        x = self.layers(x)
        return x + residual

class CumAvgPool1d(nn.Module):

    def __init__(
        self
    ):
        super().__init__()

    def forward(x):
        y = torch.zeros(x.shape)
        y[:,:,0] = x[:,:,0]
        for i in range(1,y.shape[-1]):
            y[:,:,i] = torch.mean(x[:,:,:i+1], dim=-1)

        return y

class SqueezeExciteAttention(nn.Module):

    def __init__(
        self,
        kernel_size=20,
        input_d=512,
        excite_d=4,
        output_d=2,
        pool="global" # ["global","sequential","cumulative"]
    ):
        super().__init__()

        if pool=="global" or kernel_size == None:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        elif pool=="sequential":
            self.avg_pool = nn.AvgPool1d(
                kernel_size = kernel_size,
                stride=1,
                padding=kernel_size//2,
                ceil_mode=True,
            )
        else:
            self.avg_pool = CumAvgPool1d()

        self.linear_1 = nn.Linear(
            input_d,
            excite_d
        )

        self.linear_2 = nn.Linear(
            excite_d,
            output_d
        )
    
    def forward(self, x):
        
        x = self.avg_pool(x) # batch x features x length#
        
        x = F.relu(self.linear_1(x.moveaxis(-2,-1))) # batch x length x features
        
        x = F.relu(self.linear_2(x))
        
        return F.softmax(x,dim=-1) # batch x length x features

class DynamicDepthwiseConvolution(nn.Module):

    def __init__(
        self,
        kernel_size,
        input_shape=None,
        stride=1,
        dilation=1,
        attention_type="se", # or "la"
        se_kernel_size=20,
        padding="same",
        bias=True,
        pool="global",
        num_heads=4
    ):
        super().__init__()

        bz, time, chn = input_shape
        if attention_type=="se":
            self.se_block = SqueezeExciteAttention(
                kernel_size=se_kernel_size,
                input_d=chn,
                excite_d=4,
                output_d=2,
                pool=pool
            )
        else:
            self.attention = LinearSelfAttention(
                chn, 
                d_out=2,
                num_heads=num_heads, 
                causal=False,
            )
            self.se_block = None

        self.depthwise_dilated = Conv1d(
            chn,
            kernel_size,
            input_shape=input_shape,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=chn,
            bias=bias,
        )

        self.depthwise_narrow = Conv1d(
            chn,
            kernel_size,
            input_shape=input_shape,
            stride=stride,
            dilation=1,
            padding=padding,
            groups=chn,
            bias=bias,
        )
    
    def forward(self, x):
        # B x N x L
        if type(self.se_block) == SqueezeExciteAttention:
            attn = self.se_block(x.moveaxis(-1,-2))  # batch x length / 1 x feats
        else:
            attn = self.attention(x)

        dilated_attn = attn[:,:,0].unsqueeze(-1)
        narrow_attn = attn[:,:,1].unsqueeze(-1)

        dil_x = dilated_attn*self.depthwise_dilated(x)
        nar_x = narrow_attn*self.depthwise_narrow(x)
        
        return dil_x + nar_x



class DynamicDepthwiseSeparableConv(sb.nnet.containers.Sequential):
    """Building block for the Temporal Blocks of Masknet in ConvTasNet.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    out_channels : int
        Number of output channels.
    kernel_size : int
        The kernel size in the convolutions.
    stride : int
        Convolution stride in convolutional layers.
    padding : str
        The type of padding in the convolutional layers,
        (same, valid, causal). If "valid", no padding is performed.
    dilation : int
        Amount of dilation in convolutional layers.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example
    -------
    >>> x = torch.randn(14, 100, 10)
    >>> DSconv =DynamicDepthwiseSeparableConv(x.shape, 10, 11, 1, 'same', 1)
    >>> y = DSconv(x)
    >>> y.shape
    torch.Size([14, 100, 10])

    """

    def __init__(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        norm_type="gLN",
        causal=False,
        se_kernel_size=20,
        padding="same",
        bias=True,
        pool="global",
        attention_type="se",
        num_heads=4
    ):
        super().__init__(input_shape=input_shape)

        batchsize, time, in_channels = input_shape

        # Depthwise [M, K, H] -> [M, K, H]
        self.append(
            DynamicDepthwiseConvolution,
            kernel_size=kernel_size,
            # input_shape=input_shape,
            stride=stride,
            dilation=dilation,
            se_kernel_size=se_kernel_size,
            padding=padding,
            bias=bias,
            pool=pool,
            layer_name="conv_0",
            attention_type=attention_type,
            num_heads=num_heads
        )

        if causal:
            self.append(Chomp1d(padding), layer_name="chomp")

        self.append(nn.PReLU(), layer_name="act")
        self.append(choose_norm(norm_type, in_channels), layer_name="act")

        # Pointwise [M, K, H] -> [M, K, B]
        self.append(
            sb.nnet.CNN.Conv1d,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            layer_name="conv_1",
        )

        

if __name__ == '__main__':
    batch_size, N, L = 4, 512, 33210
    P=3

    x = torch.rand((batch_size, N, L))

    # se = SqueezeExciteAttention(
    #     kernel_size=300,
    #     input_d=N,
    #     excite_d=4,
    #     output_d=2,
    # )

    # 
    # x = se(x)
    # 
    # print(x[0,0])

    N=N
    B=N//4
    H=N
    P=3
    X=1
    R=2
    C=2

    ddc = MaskNet(
        N=N,
        B=B,
        H=H,
        P=P,
        X=X,
        R=R,
        C=C,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
        se_kernel_size=20,
        bias=True,
        pool="global",
        attention_type="se"
    )

    print(x.shape,x[0,0,:3])
    x = ddc(x)
    print(x.shape,x[0,0,0,:3])
