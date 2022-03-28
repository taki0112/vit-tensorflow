import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from tensorflow import einsum
from einops import rearrange

from math import ceil

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, l = 3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))

def always(val):
    return lambda *args, **kwargs: val

def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

class HardSwish(Layer):
    def __init__(self):
        super(HardSwish, self).__init__()

    def call(self, x, training=True):
        x = x * tf.nn.relu6(x + 3.0) / 6.0
        return x

class GELU(Layer):
    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return gelu(x, self.approximate)

class MLP(Layer):
    def __init__(self, dim, mult, dropout=0.0):
        super(MLP, self).__init__()

        self.net = [
            nn.Conv2D(filters=dim * mult, kernel_size=1, strides=1),
            HardSwish(),
            nn.Dropout(rate=dropout),
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            nn.Dropout(rate=dropout)
        ]
        self.net = Sequential(self.net)

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, fmap_size, heads=8, dim_key=32, dim_value=64, dropout=0.0, dim_out=None, downsample=False):
        super(Attention, self).__init__()
        inner_dim_key = dim_key * heads
        inner_dim_value = dim_value * heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        self.to_q = Sequential([
            nn.Conv2D(filters=inner_dim_key, kernel_size=1, strides=(2 if downsample else 1), use_bias=False),
            nn.BatchNormalization(momentum=0.9, epsilon=1e-05),
        ])

        self.to_k = Sequential([
            nn.Conv2D(filters=inner_dim_key, kernel_size=1, strides=1, use_bias=False),
            nn.BatchNormalization(momentum=0.9, epsilon=1e-05),
        ])

        self.to_v = Sequential([
            nn.Conv2D(filters=inner_dim_value, kernel_size=1, strides=1, use_bias=False),
            nn.BatchNormalization(momentum=0.9, epsilon=1e-05),
        ])

        self.attend = nn.Softmax()

        out_batch_norm = nn.BatchNormalization(momentum=0.9, epsilon=1e-05, gamma_initializer='zeros')

        self.to_out = Sequential([
            GELU(),
            nn.Conv2D(filters=dim_out, kernel_size=1, strides=1),
            out_batch_norm,
            nn.Dropout(rate=dropout)
        ])

        # positional bias
        self.pos_bias = nn.Embedding(input_dim=fmap_size * fmap_size, output_dim=heads)
        q_range = tf.range(0, fmap_size, delta=(2 if downsample else 1))
        k_range = tf.range(fmap_size)

        q_pos = tf.stack(tf.meshgrid(q_range, q_range, indexing='ij'), axis=-1)
        k_pos = tf.stack(tf.meshgrid(k_range, k_range, indexing='ij'), axis=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = tf.abs((q_pos[:, None, ...] - k_pos[None, :, ...]))

        x_rel, y_rel = tf.unstack(rel_pos, axis=-1)
        self.pos_indices = (x_rel * fmap_size) + y_rel

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def call(self, x, training=True):
        b, height, width, n = x.shape
        q = self.to_q(x)

        h = self.heads
        y = q.shape[1] # height

        qkv = (q, self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> b h (...) d', h=h), qkv)

        # i,j = height*width
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h (x y) d -> b x y (h d)', h=h, y=y)
        x = self.to_out(x, training=training)

        return x

class Transformer(Layer):
    def __init__(self, dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult=2, dropout=0.0, dim_out=None, downsample=False):
        super(Transformer, self).__init__()

        dim_out = default(dim_out, dim)
        self.attn_residual = (not downsample) and dim == dim_out
        self.layers = []

        for _ in range(depth):
            self.layers.append([
                Attention(dim, fmap_size=fmap_size, heads=heads, dim_key=dim_key, dim_value=dim_value,
                          dropout=dropout, downsample=downsample, dim_out=dim_out),
                MLP(dim_out, mlp_mult, dropout=dropout)
            ])

    def call(self, x, training=True):
        for attn, mlp in self.layers:
            attn_res = (x if self.attn_residual else 0)
            x = attn(x, training=training) + attn_res
            x = mlp(x, training=training) + x

        return x

class LeViT(Model):
    def __init__(self,
                 image_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_mult,
                 stages=3,
                 dim_key=32,
                 dim_value=64,
                 dropout=0.0,
                 num_distill_classes=None
                 ):
        super(LeViT, self).__init__()

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), \
            'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'

        self.conv_embedding = Sequential([
            nn.Conv2D(filters=32, kernel_size=3, strides=2, padding='SAME'),
            nn.Conv2D(filters=64, kernel_size=3, strides=2, padding='SAME'),
            nn.Conv2D(filters=128, kernel_size=3, strides=2, padding='SAME'),
            nn.Conv2D(filters=dims[0], kernel_size=3, strides=2, padding='SAME')
        ])

        fmap_size = image_size // (2 ** 4)
        self.backbone = Sequential()

        for ind, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
            is_last = ind == (stages - 1)
            self.backbone.add(Transformer(dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult, dropout))

            if not is_last:
                next_dim = dims[ind + 1]
                self.backbone.add(Transformer(dim, fmap_size, 1, heads * 2, dim_key, dim_value, dim_out=next_dim, downsample=True))
                fmap_size = ceil(fmap_size / 2)

        self.pool = Sequential([
            nn.GlobalAvgPool2D()
        ])

        self.distill_head = nn.Dense(units=num_distill_classes) if exists(num_distill_classes) else always(None)
        self.mlp_head = nn.Dense(units=num_classes)


    def call(self, img, training=True, **kwargs):
        x = self.conv_embedding(img)

        x = self.backbone(x)

        x = self.pool(x)
        out = self.mlp_head(x)
        distill = self.distill_head(x)

        if exists(distill):
            return out, distill

        return out

# """ Usage
levit = LeViT(
    image_size = 224,
    num_classes = 1000,
    stages = 3,             # number of stages
    dim = (256, 384, 512),  # dimensions at each stage
    depth = 4,              # transformer of depth 4 at each stage
    heads = (4, 6, 8),      # heads at each stage
    mlp_mult = 2,
    dropout = 0.1
)

img = tf.random.normal(shape=[1, 224, 224, 3])
preds = levit(img) # (1, 1000)
# """
