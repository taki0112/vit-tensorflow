import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from tensorflow import einsum

from einops import rearrange
from einops.layers.tensorflow import Rearrange, Reduce

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

# cross embed layer
class CrossEmbedLayer(Layer):
    def __init__(self, dim, kernel_sizes, stride=2):
        super(CrossEmbedLayer, self).__init__()

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim - sum(dim_scales)]

        self.convs = []
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2D(filters=dim_scale, kernel_size=kernel, strides=stride, padding='SAME'))

    def call(self, x, training=True):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        x = tf.concat(fmaps, axis=-1)
        return x

# dynamic positional bias
class DynamicPositionBias(Layer):
    def __init__(self, dim):
        super(DynamicPositionBias, self).__init__()

        self.dpb_layers = Sequential([
            nn.Dense(units=dim),
            nn.LayerNormalization(),
            nn.ReLU(),
            nn.Dense(units=dim),
            nn.LayerNormalization(),
            nn.ReLU(),
            nn.Dense(units=dim),
            nn.LayerNormalization(),
            nn.ReLU(),
            nn.Dense(units=1),
            Rearrange('... () -> ...')
        ])

    def call(self, x, training=True):
        x = self.dpb_layers(x)
        return x

# transformer classes
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

class MLP(Layer):
    def __init__(self, dim, mult=4, dropout=0.0):
        super(MLP, self).__init__()

        self.net = Sequential([
            LayerNorm(dim),
            nn.Conv2D(filters=dim*mult, kernel_size=1, strides=1),
            GELU(),
            nn.Dropout(rate=dropout),
            nn.Conv2D(filters=dim, kernel_size=1, strides=1)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, attn_type, window_size, dim_head=32, dropout=0.0):
        super(Attention, self).__init__()

        assert attn_type in {'short', 'long'}, 'attention type must be one of local or distant'
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.attn_type = attn_type
        self.window_size = window_size

        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv2D(filters=inner_dim * 3, kernel_size=1, strides=1, use_bias=False)
        self.to_out = nn.Conv2D(filters=dim, kernel_size=1, strides=1)

        # positions
        self.dpb = DynamicPositionBias(dim // 4)
        self.attend = nn.Softmax()

        # calculate and store indices for retrieving bias
        pos = tf.range(window_size)
        grid = tf.stack(tf.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        self.rel_pos_indices = tf.reduce_sum(rel_pos * tf.convert_to_tensor([2 * window_size - 1, 1]), axis=-1)

    def call(self, x, training=True):
        _, height, width, _ = x.shape
        heads = self.heads
        wsz = self.window_size

        # prenorm
        x = self.norm(x)

        # rearrange for short or long distance attention

        if self.attn_type == 'short':
            x = rearrange(x, 'b (h s1) (w s2) d -> (b h w) s1 s2 d', s1=wsz, s2=wsz)
        elif self.attn_type == 'long':
            x = rearrange(x, 'b (l1 h) (l2 w) d -> (b h w) l1 l2 d', l1=wsz, l2=wsz)

        # queries / keys / values
        qkv = self.to_qkv(x)
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> b h (x y) d', h=heads), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add dynamic positional bias
        pos = tf.range(-wsz, wsz + 1)
        rel_pos = tf.stack(tf.meshgrid(pos, pos, indexing='ij'))
        rel_pos = rearrange(rel_pos, 'c i j -> (i j) c')
        biases = self.dpb(tf.cast(rel_pos, tf.float32))
        rel_pos_bias = biases.numpy()[self.rel_pos_indices.numpy()]

        sim = sim + rel_pos_bias

        # attend
        attn = self.attend(sim)

        # merge heads
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d) ', x=wsz, y=wsz)
        out = self.to_out(out)
        # rearrange back for long or short distance attention
        if self.attn_type == 'short':
            out = rearrange(out, '(b h w) s1 s2 d -> b (h s1) (w s2) d', h=height // wsz, w=width // wsz)
        elif self.attn_type == 'long':
            out = rearrange(out, '(b h w) l1 l2 d -> b (l1 h) (l2 w) d', h=height // wsz, w=width // wsz)

        return out

class Transformer(Layer):
    def __init__(self, dim, local_window_size, global_window_size, depth=4, dim_head=32, attn_dropout=0.0, ff_dropout=0.0):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                Attention(dim, attn_type='short', window_size=local_window_size, dim_head=dim_head, dropout=attn_dropout),
                MLP(dim, dropout=ff_dropout),
                Attention(dim, attn_type='long', window_size=global_window_size, dim_head=dim_head, dropout=attn_dropout),
                MLP(dim, dropout=ff_dropout)
            ])

    def call(self, x, training=True):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x, training=training) + x
            x = long_attn(x) + x
            x = long_ff(x, training=training) + x

        return x

class CrossFormer(Model):
    def __init__(self,
                 dim=(64, 128, 256, 512),
                 depth=(2, 2, 8, 2),
                 global_window_size=(8, 4, 2, 1),
                 local_window_size=7,
                 cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
                 cross_embed_strides=(4, 2, 2, 2),
                 num_classes=1000,
                 attn_dropout=0.0,
                 ff_dropout=0.0,
                 ):
        super(CrossFormer, self).__init__()
        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)

        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4

        # layers
        self.crossformer_layers = []

        for dim_out, layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim, depth,
                                                                                        global_window_size, local_window_size,
                                                                                        cross_embed_kernel_sizes, cross_embed_strides):
            self.crossformer_layers.append([
                CrossEmbedLayer(dim_out, cel_kernel_sizes, stride=cel_stride),
                Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz, depth=layers,
                            attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            ])

        # final logits
        self.to_logits = Sequential([
            Reduce('b h w c -> b c', 'mean'),
            nn.Dense(units=num_classes)
        ])

    def call(self, x, training=True, **kwargs):
        for cel, transformer in self.crossformer_layers:
            x = cel(x)
            x = transformer(x, training=training)

        x = self.to_logits(x)

        return x
""" Usage
v = CrossFormer(
    num_classes = 1000,                # number of output classes
    dim = (64, 128, 256, 512),         # dimension at each stage
    depth = (2, 2, 8, 2),              # depth of transformer at each stage
    global_window_size = (8, 4, 2, 1), # global window sizes at each stage
    local_window_size = 7,             # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
)

img = tf.random.normal(shape=[1, 224, 224, 3])
preds = v(img) # (1, 1000)
"""