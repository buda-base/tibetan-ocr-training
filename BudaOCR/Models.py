import torch
import torch.nn.functional as F
from torch import nn

"""
MultiScaleConv-Block
"""


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(
            in_channels, out_channels // 4, kernel_size=3, padding=1
        )
        self.branch2 = nn.Conv1d(
            in_channels, out_channels // 4, kernel_size=5, padding=2
        )
        self.branch3 = nn.Conv1d(
            in_channels, out_channels // 4, kernel_size=7, padding=3
        )
        self.branch4 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = torch.cat(
            [
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
                self.branch4(x),
            ],
            dim=1,
        )
        return self.relu(self.bn(out))


"""
Vanilla CRNN
"""


class ConvRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        strides: int = 1,
        padding: int = 1,
        use_bn: bool = False,
        dropout_rate: float = 0.2,
        leaky_relu: bool = False,
    ):
        super(ConvRelu, self).__init__()

        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=strides,
            padding=padding,
        )
        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = leaky_relu
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv2d(x)

        if self.use_bn:
            x = self.bn(x)

        if self.leaky_relu:
            x = nn.LeakyReLU(negative_slope=0.2)(x)
        else:
            x = nn.ReLU(inplace=True)(x)
        x = self.dropout(x)

        return x


class VanillaCRNN(nn.Module):

    def __init__(
        self,
        img_height: int = 80,
        img_width: int = 2000,
        img_channels: int = 1,
        charset_size: int = 68,
        map_to_seq_hidden: int = 64,
        rnn_hidden: int = 256,
        leaky_relu: bool = False,
        rnn: str = "lstm",
    ):
        super(VanillaCRNN, self).__init__()

        self.input_channels = img_channels
        self.input_height = img_height
        self.input_width = img_width
        self.classes = charset_size
        self.map_to_seq_hidden = map_to_seq_hidden

        self.conv_block_0 = ConvRelu(
            in_channels=self.input_channels, out_channels=64, leaky_relu=leaky_relu
        )
        self.max_pool_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block_1 = ConvRelu(
            in_channels=64, out_channels=128, leaky_relu=leaky_relu
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block_2 = ConvRelu(
            in_channels=128, out_channels=256, leaky_relu=leaky_relu
        )
        self.conv_block_3 = ConvRelu(
            in_channels=256, out_channels=256, leaky_relu=leaky_relu
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv_block_4 = ConvRelu(
            in_channels=256, out_channels=512, use_bn=True, leaky_relu=leaky_relu
        )
        self.conv_block_5 = ConvRelu(
            in_channels=512, out_channels=512, use_bn=True, leaky_relu=leaky_relu
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv_block_6 = ConvRelu(
            in_channels=512,
            out_channels=512,
            kernel=2,
            padding=0,
            leaky_relu=leaky_relu,
        )
        self.linear = nn.Linear(
            512 * (self.input_height // 16 - 1), self.map_to_seq_hidden
        )

        if rnn == "lstm":
            self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
            self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        else:
            self.rnn1 = nn.GRU(map_to_seq_hidden, rnn_hidden, bidirectional=True)
            self.rnn2 = nn.GRU(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, charset_size)

    def forward(self, images):
        x = self.conv_block_0(images)
        x = self.max_pool_0(x)
        x = self.conv_block_1(x)
        x = self.max_pool_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.max_pool_2(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.max_pool_3(x)
        x = self.conv_block_6(x)

        batch, channel, height, width = x.size()

        x = x.view(batch, channel * height, width)
        x = x.permute(2, 0, 1)
        x = self.linear(x)

        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.dense(x)

        return x


"""
An experimental PyTorch implementation of the Easter2 Architecture
    -   original Easter2-Architecture: https://github.com/kartikgill/Easter2
    -   Note: This is an early experimental version that needs some optimization and 
        parameterization for the cli.
    -   On the padding behaviour of the Conv-layers in Pytorch, 
        see here: https://github.com/pytorch/pytorch/issues/67551. I opted for manually padding the output.
"""


class GlobalContext(nn.Module):
    """
    Note: When training a Easter-ViT variant using this GlobalContext module,
    adding a parameter for the sigmoid function might be adivsable.
    If the ViT head gradients become too small, replace sigmoid with tanh or use sigmoid() * 1.5 (light scaling).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mean_pool: bool = True,
        activation: str = "sigmoid",
        activation_scale: float = 1.0,
    ):
        super(GlobalContext, self).__init__()
        self.mean_pool = mean_pool
        self.pool = nn.AvgPool1d(kernel_size=out_channels)
        self.linear1 = nn.Linear(in_channels, out_channels // 8)
        self.linear2 = nn.Linear(out_channels // 8, out_channels)
        self.relu = nn.ReLU()

        if activation == "sigmoid":
            self.activation = (
                nn.Sigmoid()
            )  # or sigmoid(x) * 1.5 activation_scale like in CBAM / ECA variants
        elif activation == "tanh":
            self.activation = (
                nn.Tanh()
            )  # maybe 1 + 0.5*tanh(x) like in FiLM / AdaIN / modulated SE

    def forward(self, data):
        pool = self.pool(data)

        if self.mean_pool:
            pool = torch.mean(pool, -1)
        else:
            pool = pool[:, :, 0]

        pool = self.linear1(pool)
        pool = self.relu(pool)
        pool = self.linear2(pool)
        pool = self.activation(pool)
        pool = torch.unsqueeze(pool, -1)
        pool = torch.multiply(pool, data)

        return pool


class GlobalContextFixed(nn.Module):
    """
    Note: When training a Easter-ViT variant using this GlobalContext module,
    adding a parameter for the sigmoid function might be adivsable.
    If the ViT head gradients become too small, replace sigmoid with tanh or use sigmoid() * 1.5 (light scaling).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mean_pool: bool = True,
        activation: str = "sigmoid",
        activation_scale: float = 1.0,
    ):
        super(GlobalContextFixed, self).__init__()
        self.mean_pool = mean_pool
        self.pool = nn.AdaptiveAvgPool1d(1)

        # kernel = min(out_channels, data.shape[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.linear1 = nn.Linear(in_channels, out_channels // 8)
        self.linear2 = nn.Linear(out_channels // 8, out_channels)
        self.relu = nn.ReLU()

        if activation == "sigmoid":
            self.activation = (
                nn.Sigmoid()
            )  # or sigmoid(x) * 1.5 activation_scale like in CBAM / ECA variants
        elif activation == "tanh":
            self.activation = (
                nn.Tanh()
            )  # maybe 1 + 0.5*tanh(x) like in FiLM / AdaIN / modulated SE

    def forward(self, data):
        pool = self.pool(data)  # (B, C, 1)
        pool = pool.squeeze(-1)  # (B, C)
        pool = self.linear1(pool)
        pool = self.relu(pool)
        pool = self.linear2(pool)
        pool = self.activation(pool)
        pool = pool.unsqueeze(-1)  # (B, C, 1)
        return pool * data


class EasterUnit(nn.Module):
    """
    Note: If stochastic depth (DropPath) is applied inside the EasterUnit,
    it should be applied after GlobalContext, not inside it.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride,
        dropout,
        bn_eps=1e-5,
        bn_decay=0.997,
        mean_pool: bool = True,
        use_global_context: bool = True,
    ):
        super(EasterUnit, self).__init__()
        self.bn_eps = bn_eps
        self.bn_decay = bn_decay
        self.dropout = dropout
        self.mean_pool = mean_pool
        self.use_global_context = use_global_context

        self.conv1d_1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=1, groups=1
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=1, groups=1
        )
        self.conv1d_3 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel, stride=stride, groups=1
        )
        self.conv1d_4 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel, stride=stride, groups=1
        )
        self.conv1d_5 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel, stride=stride, groups=1
        )

        self.bn_1 = nn.BatchNorm1d(
            num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_2 = nn.BatchNorm1d(
            num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_3 = nn.BatchNorm1d(
            num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_4 = nn.BatchNorm1d(
            num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_5 = nn.BatchNorm1d(
            num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay
        )

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()

        self.drop_1 = nn.Dropout(p=self.dropout)
        self.drop_2 = nn.Dropout(p=self.dropout)
        self.drop_3 = nn.Dropout(p=self.dropout)
        self.global_context = GlobalContext(
            in_channels=out_channels,
            out_channels=out_channels,
            mean_pool=self.mean_pool,
        )

    def forward(self, inputs):
        old, data = inputs
        old = self.conv1d_1(old)
        old = self.bn_1(old)

        this = self.conv1d_2(data)
        this = self.bn_2(this)
        old = torch.add(old, this)

        # First Block
        data = self.conv1d_3(data)
        pad_val = old.shape[-1] - data.shape[-1]
        data = nn.ZeroPad1d(padding=(0, pad_val))(data)
        # data = F.pad(data, (pad_val//2, pad_val - pad_val//2))
        data = self.bn_3(data)
        data = self.relu_1(data)
        data = self.drop_1(data)

        # Second Block
        data = self.conv1d_4(data)
        pad_val = old.shape[-1] - data.shape[-1]
        data = nn.ZeroPad1d(padding=(0, pad_val))(data)
        # data = F.pad(data, (pad_val//2, pad_val - pad_val//2))
        data = self.bn_4(data)
        data = self.relu_2(data)
        data = self.drop_2(data)

        # Third Block
        data = self.conv1d_5(data)
        pad_val = old.shape[-1] - data.shape[-1]
        data = nn.ZeroPad1d(padding=(0, pad_val))(data)
        # data = F.pad(data, (pad_val//2, pad_val - pad_val//2))
        data = self.bn_5(data)

        data = self.global_context(data)
        data = torch.add(old, data)
        data = self.relu_3(data)
        data = self.drop_3(data)

        return data, old


class Easter2(nn.Module):
    def __init__(
        self,
        input_channels: int,  # dynamically provided from CNN frontend
        vocab_size: int = 77,
        bn_eps: float = 1e-5,
        bn_decay: float = 0.997,
        mean_pooling: bool = True,
        dropout: float = 0.2,
        use_global_context: bool = True,
    ):
        super(Easter2, self).__init__()

        self.vocab_size = vocab_size
        self.bn_eps = bn_eps
        self.bn_decay = bn_decay
        self.mean_pooling = mean_pooling
        self.dropout = dropout

        # --- Initial Conv stages ---
        self.zero_pad = nn.ZeroPad1d(padding=(0, 1))
        self.conv1d_1 = nn.Conv1d(input_channels, 128, kernel_size=3, stride=2)
        # self.conv1d_1 = nn.Conv1d(input_channels, 128, kernel_size=3, stride=2, padding=1)
        self.bn_1 = nn.BatchNorm1d(128, eps=self.bn_eps, momentum=self.bn_decay)
        self.relu_1 = nn.ReLU()
        self.drop_1 = nn.Dropout(p=self.dropout)

        self.conv1d_2 = nn.Conv1d(128, 128, kernel_size=3, stride=2)
        # self.conv1d_2 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn_2 = nn.BatchNorm1d(128, eps=self.bn_eps, momentum=self.bn_decay)
        self.relu_2 = nn.ReLU()
        self.drop_2 = nn.Dropout(p=self.dropout)

        # --- Easter units ---
        self.easter1 = EasterUnit(
            128,
            128,
            5,
            1,
            0.2,
            mean_pool=self.mean_pooling,
            use_global_context=use_global_context,
        )
        self.easter2 = EasterUnit(
            128,
            256,
            5,
            1,
            0.2,
            mean_pool=self.mean_pooling,
            use_global_context=use_global_context,
        )
        self.easter3 = EasterUnit(
            256,
            256,
            7,
            1,
            0.2,
            mean_pool=self.mean_pooling,
            use_global_context=use_global_context,
        )
        self.easter4 = EasterUnit(
            256,
            256,
            9,
            1,
            0.3,
            mean_pool=self.mean_pooling,
            use_global_context=use_global_context,
        )

        # --- Final projection ---
        self.conv1d_3 = nn.Conv1d(256, 512, kernel_size=11, stride=1, dilation=2)
        self.bn_3 = nn.BatchNorm1d(512, eps=self.bn_eps, momentum=self.bn_decay)
        self.relu_3 = nn.ReLU()
        self.drop_3 = nn.Dropout(p=0.4)

        self.conv1d_4 = nn.Conv1d(512, 512, kernel_size=1)
        self.bn_4 = nn.BatchNorm1d(512, eps=self.bn_eps, momentum=self.bn_decay)
        self.relu_4 = nn.ReLU()
        self.drop_4 = nn.Dropout(p=0.4)

        self.conv1d_5 = nn.Conv1d(512, self.vocab_size, kernel_size=1)

        print(f"Created Easter Network with Input-Channels: {input_channels}")

    def forward(self, inputs):
        """
        Args:
            inputs: (B, C, L) — from CNNFrontEnd
        Returns:
            logits: (B, vocab_size, L')
        """
        x = self.conv1d_1(inputs)
        x = self.zero_pad(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.drop_1(x)

        x = self.conv1d_2(x)
        x = self.zero_pad(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)

        old = x
        data, old = self.easter1((old, x))
        data, old = self.easter2((old, data))
        data, old = self.easter3((old, data))
        data, old = self.easter4((old, data))

        x = self.conv1d_3(data)
        x = nn.ZeroPad1d(padding=(10, 10))(x)
        x = self.bn_3(x)
        x = self.relu_3(x)
        x = self.drop_3(x)

        x = self.conv1d_4(x)
        x = self.bn_4(x)
        x = self.relu_4(x)
        x = self.drop_4(x)

        features = x  # (B, 512, L)
        logits = self.conv1d_5(features)  # (B, vocab_size, L)

        return logits, features


"""
Easter2Fixes for A/B-Testing trying to remove the eating of pixels at the beginning of a line
"""


class EasterUnitB(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride,
        dropout,
        bn_eps=1e-5,
        bn_decay=0.997,
        mean_pool=True,
    ):
        super().__init__()
        self.dropout = dropout
        self.mean_pool = mean_pool

        # 1x1 convs unchanged
        self.conv1d_1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # "same" padding for main conv paths
        pad = kernel // 2

        self.conv1d_3 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad
        )

        self.conv1d_4 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad
        )

        self.conv1d_5 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad
        )

        # batchnorms unchanged
        self.bn_1 = nn.BatchNorm1d(out_channels, eps=bn_eps, momentum=bn_decay)
        self.bn_2 = nn.BatchNorm1d(out_channels, eps=bn_eps, momentum=bn_decay)
        self.bn_3 = nn.BatchNorm1d(out_channels, eps=bn_eps, momentum=bn_decay)
        self.bn_4 = nn.BatchNorm1d(out_channels, eps=bn_eps, momentum=bn_decay)
        self.bn_5 = nn.BatchNorm1d(out_channels, eps=bn_eps, momentum=bn_decay)

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()

        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)
        self.drop_3 = nn.Dropout(dropout)

        self.global_context = GlobalContextFixed(
            in_channels=out_channels,
            out_channels=out_channels,
            mean_pool=mean_pool,
        )

    def forward(self, inputs):
        old, data = inputs

        # skip branch
        old = self.bn_1(self.conv1d_1(old))

        # merge branch
        this = self.bn_2(self.conv1d_2(data))
        old = old + this

        # block 1 -- SAME padding preserves length
        data = self.conv1d_3(data)
        data = self.bn_3(data)
        data = self.relu_1(data)
        data = self.drop_1(data)

        # block 2
        data = self.conv1d_4(data)
        data = self.bn_4(data)
        data = self.relu_2(data)
        data = self.drop_2(data)

        # block 3
        data = self.conv1d_5(data)
        data = self.bn_5(data)

        data = self.global_context(data)
        data = old + data
        data = self.relu_3(data)
        data = self.drop_3(data)

        return data, old


class Easter2b(nn.Module):
    def __init__(
        self,
        input_height=80,
        bn_eps=1e-5,
        bn_decay=0.997,
        vocab_size=77,
        mean_pooling=True,
    ):
        super().__init__()

        self.input_height = input_height
        self.vocab_size = vocab_size
        self.mean_pooling = mean_pooling

        # SAME padding for stride=2
        self.conv1d_1 = nn.Conv1d(input_height, 128, kernel_size=3, stride=2, padding=1)
        self.conv1d_2 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)

        self.bn_1 = nn.BatchNorm1d(128, eps=bn_eps, momentum=bn_decay)
        self.bn_2 = nn.BatchNorm1d(128, eps=bn_eps, momentum=bn_decay)

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

        # Replace EasterUnit with fixed version
        self.easter1 = EasterUnitB(128, 128, 5, 1, 0.2, mean_pool=mean_pooling)
        self.easter2 = EasterUnitB(128, 256, 5, 1, 0.2, mean_pool=mean_pooling)
        self.easter3 = EasterUnitB(256, 256, 7, 1, 0.2, mean_pool=mean_pooling)
        self.easter4 = EasterUnitB(256, 256, 9, 1, 0.3, mean_pool=mean_pooling)

        self.drop_1 = nn.Dropout(0.2)
        self.drop_2 = nn.Dropout(0.2)
        self.drop_3 = nn.Dropout(0.4)
        self.drop_4 = nn.Dropout(0.4)

        # SAME padding for dilated conv
        # effective kernel = 11 + (11-1)*(dilation-1) = 21 -> pad 10
        self.conv1d_3 = nn.Conv1d(
            256, 512, kernel_size=11, stride=1, dilation=2, padding=10
        )

        self.conv1d_4 = nn.Conv1d(512, 512, kernel_size=1, padding=0)
        self.conv1d_5 = nn.Conv1d(512, vocab_size, kernel_size=1, padding=0)

        self.bn_3 = nn.BatchNorm1d(512, eps=bn_eps, momentum=bn_decay)
        self.bn_4 = nn.BatchNorm1d(512, eps=bn_eps, momentum=bn_decay)

        self.relu_3 = nn.ReLU()
        self.relu_4 = nn.ReLU()

    def forward(self, inputs):
        x = self.relu_1(self.bn_1(self.conv1d_1(inputs)))
        x = self.drop_1(x)

        x = self.relu_2(self.bn_2(self.conv1d_2(x)))
        x = self.drop_2(x)

        old = x
        x, old = self.easter1((old, x))
        x, old = self.easter2((old, x))
        x, old = self.easter3((old, x))
        x, old = self.easter4((old, x))

        x = self.relu_3(self.bn_3(self.conv1d_3(x)))
        x = self.drop_3(x)

        x = self.relu_4(self.bn_4(self.conv1d_4(x)))
        x = self.drop_4(x)

        features = x  # (B, 512, L)
        logits = self.conv1d_5(features)  # (B, vocab_size, L)

        return logits, features


"""
Models: Easter2PlusLight with lightweight CNN and Self-Attention Head
"""


class CNNFrontEnd(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # reduce height but not width
        )

    def forward(self, x):
        x = self.features(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)  # convert to (B, H*C, W)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or dim
        assert (
            attn_dim % heads == 0
        ), f"attn_dim {attn_dim} must be divisible by num_heads {heads}"

        self.proj_in = nn.Linear(dim, attn_dim) if attn_dim != dim else nn.Identity()
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=heads, batch_first=True
        )
        self.norm = nn.LayerNorm(attn_dim)
        self.mlp = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.GELU(),
            nn.Linear(attn_dim * 4, attn_dim),
        )
        self.proj_out = nn.Linear(attn_dim, dim) if attn_dim != dim else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        x = self.proj_in(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm(x + self.mlp(x))
        x = self.proj_out(x)
        return x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)


class Easter2PlusLight(nn.Module):
    """
    Easter2 with:
      - CNNFrontEnd2 for better vertical sensitivity
      - 1x1 projection to match Easter2 input channels
      - optional front GroupNorm
      - lightweight SelfAttention over logits
    """

    def __init__(
        self,
        vocab_size: int = 80,
        attention_dim: int = 128,
        input_height: int = 100,
        easter_in_ch: int = 128,
        use_front_norm: bool = False,
        easter_variant: str = "default",
    ):
        super().__init__()
        self.cnn_front = CNNFrontEnd(in_channels=1, out_channels=64)

        # Probe CNN output shape once
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, 1000)
            feat = self.cnn_front(dummy)  # (1, C_big, L)
            C_big = feat.shape[1]
            # print(f"[E2+Light] CNNFrontEnd2 out_channels = {C_big}")

        # Optional normalization of CNN features
        self.norm_front = nn.GroupNorm(32, C_big) if use_front_norm else nn.Identity()

        # Project CNN features -> Easter2 input channels
        self.proj = nn.Conv1d(C_big, easter_in_ch, kernel_size=1, bias=False)

        # Easter2 backbone must be configured to expect 'easter_in_ch' channels
        if easter_variant == "default":
            self.backbone = Easter2(
                input_channels=easter_in_ch,
                vocab_size=vocab_size,
                use_global_context=False,
            )
            print(f"Using Easter2 standard")
        else:
            self.backbone = Easter2b(
                input_height=easter_in_ch, vocab_size=vocab_size
            )
            print(f"Using Easter2 Fixed")

        # Attention over logits (B, V, T)
        self.attention = SelfAttention(dim=vocab_size, heads=4, attn_dim=attention_dim)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.cnn_front(x)  # (B, C_big, W')
        x = self.norm_front(x)  # (B, C_big, W')
        x = self.proj(x)  # (B, C_easter, W')

        logits, _ = self.backbone(x)  # (B, V, T)
        x = self.attention(logits)  # (B, V, T) with context mixing
        return x  # return logits-like tensor for CTC


"""
Models: Easter2Plus with ViT Module

"""


class ConvFrontEnd(nn.Module):
    def __init__(self, out_ch=64, input_height=100):
        super().__init__()
        # simple 2D frontend that reduces height by 4 and keeps width
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(32, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),
        )
        self.out_ch = out_ch
        self.input_height = input_height

    def forward(self, x):
        f = self.net(x)          # B x C x H/4 x W
        b, c, h, w = f.shape
        return f.view(b, c * h, w)  # B x (C*H') x W

# -------------- ConvPatchViTEncoder (reusable) ----------------
class ConvPatchViTEncoder(nn.Module):
    """Lightweight ViT encoder with conv-based patch embedding.

    Expects input (B, C, L) and returns (B, embed_dim, L') where L' depends on patch_stride.
    """
    def __init__(self, in_ch, embed_dim=512, patch_kernel=3, patch_stride=1, num_layers=2, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.patch_proj = nn.Conv1d(in_ch, embed_dim, kernel_size=patch_kernel, stride=patch_stride, padding=patch_kernel//2)
        self.norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, C, L)
        x = self.patch_proj(x)           # (B, embed_dim, L')
        x = x.permute(0, 2, 1)           # (B, L', embed_dim)
        x = self.norm(x)
        x = self.transformer(x)         # (B, L', embed_dim)
        x = x.permute(0, 2, 1)          # (B, embed_dim, L')
        return x

# -------------- Easter2PlusViT (no aux head) ------------------
class Easter2PlusViT(nn.Module):
    """CNNFrontEnd -> Backbone -> ConvPatchViTEncoder -> classifier logits

    backbone MUST return (features, feat_intermediate)
    """
    def __init__(self, cnn_front, backbone, vit_cfg, vocab_size=77):
        super().__init__()
        self.cnn_front = cnn_front
        self.backbone = backbone
        self.vit = ConvPatchViTEncoder(**vit_cfg)
        self.classifier = nn.Conv1d(vit_cfg['embed_dim'], vocab_size, kernel_size=1)

    def forward(self, x):
        _, features = self.backbone(self.cnn_front(x))  # feat: (B, C, L)
        vit_out = self.vit(features)                    # (B, embed_dim, L')
        logits = self.classifier(vit_out)           # (B, vocab, L')
        return logits
