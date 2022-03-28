import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from tensorflow import einsum

from einops import rearrange
from einops.layers.tensorflow import Rearrange, Reduce

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

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

class GELU(Layer):
    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return gelu(x, self.approximate)

class MLP(Layer):
    def __init__(self, dim, mult=4, dropout=0.0):
        super(MLP, self).__init__()

        self.net = [
            nn.Conv2D(filters=dim * mult, kernel_size=1, strides=1),
            GELU(),
            nn.Dropout(rate=dropout),
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            nn.Dropout(rate=dropout)
        ]
        self.net = Sequential(self.net)

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, heads=8, dropout=0.0):
        super(Attention, self).__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax()
        self.to_qkv = nn.Conv2D(filters=inner_dim * 3, kernel_size=1, strides=1, use_bias=False)

        self.to_out = Sequential([
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        b, h, w, c = x.shape
        heads = self.heads

        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> b h (x y) d', h=heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x=h, y=w)
        out = self.to_out(out, training=training)

        return out

class Aggregate(Layer):
    def __init__(self, dim):
        super(Aggregate, self).__init__()

        self.ag_layers = Sequential([
            nn.Conv2D(filters=dim, kernel_size=3, strides=1, padding='SAME'),
            LayerNorm(dim),
            nn.MaxPool2D(pool_size=3, strides=2, padding='SAME')
        ])

    def call(self, x, training=True):
        x = self.ag_layers(x)
        return x

class Transformer(Layer):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout=0.0):
        super(Transformer, self).__init__()
        self.layers = []
        self.pos_emb = tf.Variable(tf.random.normal([seq_len]))

        for _ in range(depth):
            self.layers.append([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, MLP(dim, mlp_mult, dropout=dropout))
            ])

    def call(self, x, training=True):
        _, h, w, c = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () h w ()', h = h, w = w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x, training=training) + x
            x = ff(x, training=training) + x

        return x

class NesT(Model):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 heads,
                 num_hierarchies,
                 block_repeats,
                 mlp_mult=4,
                 dropout=0.0
                 ):
        super(NesT, self).__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        fmap_size = image_size // patch_size
        blocks = 2 ** (num_hierarchies - 1)

        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        self.patch_embedding = Sequential([
            Rearrange('b (h p1) (w p2) c -> b h w (p1 p2 c) ', p1=patch_size, p2=patch_size),
            nn.Conv2D(filters=layer_dims[0], kernel_size=1, strides=1)
        ])

        block_repeats = cast_tuple(block_repeats, num_hierarchies)

        self.nest_layers = []

        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat

            self.nest_layers.append([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_out) if not is_last else Identity()
            ])

        self.mlp_head = Sequential([
            LayerNorm(last_dim),
            Reduce('b h w c -> b c', 'mean'),
            nn.Dense(units=num_classes)
        ])

    def call(self, img, training=True, **kwargs):
        x = self.patch_embedding(img)

        num_hierarchies = len(self.nest_layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.nest_layers):
            block_size = 2 ** level
            x = rearrange(x, 'b (b1 h) (b2 w) c -> (b b1 b2) h w c', b1 = block_size, b2 = block_size)
            x = transformer(x, training=training)
            x = rearrange(x, '(b b1 b2) h w c -> b (b1 h) (b2 w) c', b1 = block_size, b2 = block_size)
            x = aggregate(x)

        x = self.mlp_head(x)

        return x

""" Usage
v = NesT(
    image_size = 224,
    patch_size = 4,
    dim = 96,
    heads = 3,
    num_hierarchies = 3,        # number of hierarchies
    block_repeats = (2, 2, 8),  # the number of transformer blocks at each heirarchy, starting from the bottom
    num_classes = 1000
)

img = tf.random.normal(shape=[1, 224, 224, 3])
preds = v(img) # (1, 1000)
"""


