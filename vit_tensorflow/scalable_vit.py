import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from einops import rearrange
from einops.layers.tensorflow import Reduce

from functools import partial

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

class GELU(Layer):
    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return gelu(x, self.approximate)

class Identity(Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x, training=True):
        return tf.identity(x)

class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.g = tf.Variable(tf.ones([1, 1, 1, dim]))
        self.b = tf.Variable(tf.zeros([1, 1, 1, dim]))

    def call(self, x, training=True):
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)

        x = (x - mean) / tf.sqrt((var + self.eps)) * self.g + self.b
        return x

class PreNorm(Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()

        self.norm = LayerNorm(dim)
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)

class Downsample(Layer):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2D(filters=dim, kernel_size=3, strides=2, padding='SAME')

    def call(self, x, training=True):
        x = self.conv(x)
        return x

class PEG(Layer):
    def __init__(self, dim, kernel_size=3):
        super(PEG, self).__init__()
        self.proj = nn.Conv2D(filters=dim, kernel_size=kernel_size, strides=1, padding='SAME', groups=dim)

    def call(self, x, training=True):
        x = self.proj(x) + x
        return x

class MLP(Layer):
    def __init__(self, dim, expansion_factor=4, dropout=0.0):
        super(MLP, self).__init__()
        inner_dim = dim * expansion_factor
        self.net = Sequential([
            nn.Conv2D(filters=inner_dim, kernel_size=1, strides=1),
            GELU(),
            nn.Dropout(rate=dropout),
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class ScalableSelfAttention(Layer):
    def __init__(self, dim, heads=8, dim_key=32, dim_value=32, dropout=0.0, reduction_factor=1):
        super(ScalableSelfAttention, self).__init__()

        self.heads = heads
        self.scale = dim_key ** -0.5
        self.attend = nn.Softmax()

        self.to_q = nn.Conv2D(filters=dim_key * heads, kernel_size=1, strides=1, use_bias=False)
        self.to_k = nn.Conv2D(filters=dim_key * heads, kernel_size=reduction_factor, strides=reduction_factor, use_bias=False)
        self.to_v = nn.Conv2D(filters=dim_value * heads, kernel_size=reduction_factor, strides=reduction_factor, use_bias=False)

        self.to_out = Sequential([
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        _, height, width, _ = x.shape
        heads = self.heads

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # split out heads
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> b h (...) d', h=heads), (q, k, v))

        # similarity
        dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale

        # attention
        attn = self.attend(dots)

        # aggregate values
        out = tf.matmul(attn, v)

        # merge back heads
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x=height, y=width)
        out = self.to_out(out, training=training)

        return out

class InteractiveWindowedSelfAttention(Layer):
    def __init__(self, dim, window_size, heads=8, dim_key=32, dim_value=32, dropout=0.0):
        super(InteractiveWindowedSelfAttention, self).__init__()

        self.heads = heads
        self.scale = dim_key ** -0.5
        self.window_size = window_size
        self.attend = nn.Softmax()

        self.local_interactive_module = nn.Conv2D(filters=dim_value * heads, kernel_size=3, strides=1, padding='SAME')

        self.to_q = nn.Conv2D(filters=dim_key * heads, kernel_size=1, strides=1, use_bias=False)
        self.to_k = nn.Conv2D(filters=dim_key * heads, kernel_size=1, strides=1, use_bias=False)
        self.to_v = nn.Conv2D(filters=dim_value * heads, kernel_size=1, strides=1, use_bias=False)

        self.to_out = Sequential([
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        _, height, width, _ = x.shape
        heads = self.heads
        wsz = self.window_size

        wsz_h, wsz_w = default(wsz, height), default(wsz, width)
        assert (height % wsz_h) == 0 and (width % wsz_w) == 0, f'height ({height}) or width ({width}) of feature map is not divisible by the window size ({wsz_h}, {wsz_w})'

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # get output of LIM
        local_out = self.local_interactive_module(v)

        # divide into window (and split out heads) for efficient self attention
        q, k, v = map(lambda t: rearrange(t, 'b (x w1) (y w2) (h d) -> (b x y) h (w1 w2) d', h = heads, w1 = wsz_h, w2 = wsz_w), (q, k, v))

        # similarity
        dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale

        # attention
        attn = self.attend(dots)

        # aggregate values
        out = tf.matmul(attn, v)

        # reshape the windows back to full feature map (and merge heads)
        out = rearrange(out, '(b x y) h (w1 w2) d -> b (x w1) (y w2) (h d)', x = height // wsz_h, y = width // wsz_w, w1 = wsz_h, w2 = wsz_w)

        # add LIM output
        out = out + local_out

        out = self.to_out(out, training=training)

        return out

class Transformer(Layer):
    def __init__(self,
                 dim,
                 depth,
                 heads=8,
                 ff_expansion_factor=4,
                 dropout=0.,
                 ssa_dim_key=32,
                 ssa_dim_value=32,
                 ssa_reduction_factor=1,
                 iwsa_dim_key=32,
                 iwsa_dim_value=32,
                 iwsa_window_size=None,
                 norm_output=True
                 ):
        super(Transformer, self).__init__()

        self.layers = []

        for ind in range(depth):
            is_first = ind == 0

            self.layers.append([
                PreNorm(dim, ScalableSelfAttention(dim, heads=heads, dim_key=ssa_dim_key, dim_value=ssa_dim_value,
                                                   reduction_factor=ssa_reduction_factor, dropout=dropout)),
                PreNorm(dim, MLP(dim, expansion_factor=ff_expansion_factor, dropout=dropout)),
                PEG(dim) if is_first else None,
                PreNorm(dim, MLP(dim, expansion_factor=ff_expansion_factor, dropout=dropout)),
                PreNorm(dim, InteractiveWindowedSelfAttention(dim, heads=heads, dim_key=iwsa_dim_key, dim_value=iwsa_dim_value,
                                                              window_size=iwsa_window_size,
                                                              dropout=dropout))
            ])

        self.norm = LayerNorm(dim) if norm_output else Identity()

    def call(self, x, training=True):
        for ssa, ff1, peg, iwsa, ff2 in self.layers:
            x = ssa(x, training=training) + x
            x = ff1(x, training=training) + x

            if exists(peg):
                x = peg(x)

            x = iwsa(x, training=training) + x
            x = ff2(x, training=training) + x

        x = self.norm(x)

        return x

class ScalableViT(Model):
    def __init__(self,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 reduction_factor,
                 window_size=None,
                 iwsa_dim_key=32,
                 iwsa_dim_value=32,
                 ssa_dim_key=32,
                 ssa_dim_value=32,
                 ff_expansion_factor=4,
                 channels=3,
                 dropout=0.0
                 ):
        super(ScalableViT, self).__init__()

        self.to_patches = nn.Conv2D(filters=dim, kernel_size=7, strides=4, padding='SAME')

        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        num_stages = len(depth)
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))

        hyperparams_per_stage = [
            heads,
            ssa_dim_key,
            ssa_dim_value,
            reduction_factor,
            iwsa_dim_key,
            iwsa_dim_value,
            window_size,
        ]

        hyperparams_per_stage = list(map(partial(cast_tuple, length=num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))

        self.scalable_layers = []

        for ind, (layer_dim, layer_depth, layer_heads, layer_ssa_dim_key, layer_ssa_dim_value, layer_ssa_reduction_factor, layer_iwsa_dim_key, layer_iwsa_dim_value, layer_window_size) in enumerate(zip(dims, depth, *hyperparams_per_stage)):
            is_last = ind == (num_stages - 1)

            self.scalable_layers.append([
                Transformer(dim=layer_dim, depth=layer_depth, heads=layer_heads,
                            ff_expansion_factor=ff_expansion_factor, dropout=dropout, ssa_dim_key=layer_ssa_dim_key,
                            ssa_dim_value=layer_ssa_dim_value, ssa_reduction_factor=layer_ssa_reduction_factor,
                            iwsa_dim_key=layer_iwsa_dim_key, iwsa_dim_value=layer_iwsa_dim_value,
                            iwsa_window_size=layer_window_size),
                Downsample(layer_dim * 2) if not is_last else None
            ])

        self.mlp_head = Sequential([
            Reduce('b h w d-> b d', 'mean'),
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ])

    def call(self, img, training=True, **kwargs):
        x = self.to_patches(img)

        for transformer, downsample in self.scalable_layers:
            x = transformer(x, training=training)

            if exists(downsample):
                x = downsample(x)

        x = self.mlp_head(x)

        return x

""" Usage
v = ScalableViT(
    num_classes = 1000,
    dim = 64,                               # starting model dimension. at every stage, dimension is doubled
    heads = (2, 4, 8, 16),                  # number of attention heads at each stage
    depth = (2, 2, 20, 2),                  # number of transformer blocks at each stage
    ssa_dim_key = (40, 40, 40, 32),         # the dimension of the attention keys (and queries) for SSA. in the paper, they represented this as a scale factor on the base dimension per key (ssa_dim_key / dim_key)
    reduction_factor = (8, 4, 2, 1),        # downsampling of the key / values in SSA. in the paper, this was represented as (reduction_factor ** -2)
    window_size = (64, 32, None, None),     # window size of the IWSA at each stage. None means no windowing needed
    dropout = 0.1,                          # attention and feedforward dropout
)

img = tf.random.normal(shape=[1, 256, 256, 3])
preds = v(img) # (1, 1000)
"""
