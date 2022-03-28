import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from einops import rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Pre-defined CCT Models
__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8', 'cct_14', 'cct_16']


def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None,
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               *args, **kwargs)


def GELU():
    def gelu(x, approximate=False):
        if approximate:
            coeff = tf.cast(0.044715, x.dtype)
            return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
        else:
            return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

    return nn.Activation(gelu)

def drop_path(x, drop_prob=0.0, training=False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = [x.shape[0]] + [1] * (tf.rank(x).numpy() - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + tf.random.uniform(shape=shape, minval=0.0, maxval=1.0, dtype=x.dtype)
    random_tensor = tf.floor(random_tensor) # binarize
    x = tf.divide(x, keep_prob) * random_tensor
    return x

class DropPath(Layer):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=True):
        return drop_path(x, self.drop_prob, training=training)

class Attention(Layer):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Dense(units=dim * 3, use_bias=False)
        self.attend = nn.Softmax()
        self.attn_drop = nn.Dropout(rate=attention_dropout)

        self.proj = [
            nn.Dense(units=dim),
            nn.Dropout(rate=projection_dropout)
        ]

        self.proj = Sequential(self.proj)

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale

        attn = self.attend(dots)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x, training=training)
        return x

class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.pre_norm = nn.LayerNormalization()
        self.self_attn = Attention(dim=d_model, num_heads=nhead, attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = nn.Dense(units=dim_feedforward)
        self.dropout1 = nn.Dropout(rate=dropout)
        self.norm1 = nn.LayerNormalization()
        self.linear2 = nn.Dense(units=d_model)
        self.dropout2 = nn.Dropout(rate=dropout)
        self.drop_path_rate = drop_path_rate

        if drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)

        self.activation = GELU()

    def call(self, src, training=True):
        if self.drop_path_rate > 0.0:
            src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        else:
            src = src + self.self_attn(self.pre_norm(src))

        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src)), training=training))
        src2 = self.dropout2(src2, training=training)

        if self.drop_path_rate > 0.0:
            src = src + self.drop_path(src2, training=training)
        else:
            src = src + src2

        return src

class Tokenizer(Layer):
    def __init__(self,
                 kernel_size, stride,
                 pooling_kernel_size=3, pooling_stride=2,
                 n_conv_layers=1,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        conv_layers = []

        for i in range(n_conv_layers):
            if i == n_conv_layers-1:
                channels = n_output_channels
            else:
                channels = in_planes

            conv_layers += [nn.Conv2D(filters=channels, kernel_size=kernel_size, strides=stride, padding='SAME', use_bias=conv_bias)]
            if activation is not None:
                conv_layers += [activation()]
            if max_pool:
                conv_layers += [nn.MaxPool2D(pool_size=pooling_kernel_size, strides=pooling_stride, padding='SAME')]

        self.conv_layers = Sequential(conv_layers)

    def sequence_length(self, n_channels=3, height=224, width=224):
        x = tf.zeros(shape=[1, height, width, n_channels])
        x = self.call(x)
        x = x.shape[1]

        return x

    def call(self, x, **kwargs):
        x = self.conv_layers(x)
        x = tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

        return x

class TransformerClassifier(Layer):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        super(TransformerClassifier, self).__init__()

        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = tf.Variable(tf.zeros([1, 1, self.embedding_dim]))
        else:
            self.attention_pool = nn.Dense(units=1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = tf.Variable(tf.random.truncated_normal(shape=[1, sequence_length, embedding_dim], stddev=0.2))
            else:
                self.positional_emb = tf.Variable(self.sinusoidal_embedding(sequence_length, embedding_dim), trainable=False)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(rate=dropout_rate)
        dpr = [x.numpy() for x in tf.linspace(0.0, stochastic_depth_rate, num_layers)]

        self.blocks = Sequential()
        for i in range(num_layers):
            self.blocks.add(TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i]))
        self.norm = nn.LayerNormalization()
        self.fc = nn.Dense(units=num_classes)

    def sinusoidal_embedding(self, n_channels, dim):
        pe = tf.cast(([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)]), tf.float32)
        pe[:, 0::2] = tf.sin(pe[:, 0::2])
        pe[:, 1::2] = tf.cos(pe[:, 1::2])
        pe = tf.expand_dims(pe, axis=0)

        return pe

    def call(self, x, training=True):
        if self.positional_emb is None and x.shape[1] < self.sequence_length :

            x = tf.pad(x, [[0, 0], [0, self.sequence_length - x.shape[1]], [0, 0]])
        if not self.seq_pool:
            cls_token = tf.tile(self.class_emb, multiples=[x.shape[0], 1, 1])
            x = tf.concat([cls_token, x], axis=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x, training=training)

        x = self.blocks(x, training=training)
        x = self.norm(x)

        if self.seq_pool:
            x_init = x
            x = self.attention_pool(x)
            x = tf.nn.softmax(x, axis=1)
            x = tf.transpose(x, perm=[0, 2, 1])
            x = tf.matmul(x, x_init)
            x = tf.squeeze(x, axis=1)
        else:
            x = x[:, 0]

        x = self.fc(x)

        return x

class CCT(Model):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 *args, **kwargs):
        super(CCT, self).__init__()
        img_height, img_width = pair(img_size)
        self.tokenizer = Tokenizer(n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_height,
                                                           width=img_width),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth_rate=0.1,
            *args, **kwargs)


    def call(self, img, training=None, **kwargs):
        x = self.tokenizer(img, training=training)
        x = self.classifier(x, training=training)
        return x

""" Usage 
v = CCT(
    img_size = (224, 448),
    embedding_dim = 384,
    n_conv_layers = 2,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_layers = 14,
    num_heads = 6,
    mlp_radio = 3.,
    num_classes = 1000,
    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
)

# cct = cct_2(
#     img_size = 224,
#     n_conv_layers = 1,
#     kernel_size = 7,
#     stride = 2,
#     padding = 3,
#     pooling_kernel_size = 3,
#     pooling_stride = 2,
#     pooling_padding = 1,
#     num_classes = 1000,
#     positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
# )

img = tf.random.normal(shape=[5, 224, 224, 3])
preds = v(img) # (1, 1000)
"""