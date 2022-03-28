import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from einops import rearrange
from einops.layers.tensorflow import Rearrange, Reduce

def exists(val):
    return val is not None

def default(val ,d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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

class PatchMerger(Layer):
    def __init__(self, dim, num_tokens_out):
        super(PatchMerger, self).__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNormalization()
        self.queries = tf.Variable(tf.random.normal([num_tokens_out, dim]))

    def call(self, x, training=True):
        x = self.norm(x)
        sim = tf.matmul(self.queries, tf.transpose(x, perm=[0, 2, 1]) * self.scale)
        attn = tf.nn.softmax(sim, axis=-1)
        x = tf.matmul(attn, x)

        return x

class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()

        self.norm = nn.LayerNormalization()
        self.fn = fn

    def call(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.net = Sequential([
            nn.Dense(units=hidden_dim),
            GELU(),
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax()
        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False)

        self.to_out = Sequential([
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ]) if project_out else Identity()

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        attn = self.attend(dots)

        x = tf.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x, training=training)

        return x

class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, patch_merge_layer=None, patch_merge_num_tokens=8):
        super(Transformer, self).__init__()

        self.layers = []
        self.patch_merge_layer_index = default(patch_merge_layer, depth // 2) - 1  # default to mid-way through transformer, as shown in paper
        self.patch_merger = PatchMerger(dim=dim, num_tokens_out=patch_merge_num_tokens)

        for _ in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout))
            ])

    def call(self, x, training=True):
        for index, (attn, ff) in enumerate(self.layers):
            x = attn(x, training=training) + x
            x = ff(x, training=training) + x

            if index == self.patch_merge_layer_index:
                x = self.patch_merger(x)

        return x

class ViT(Model):
    def __init__(self, image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 patch_merge_layer=None,
                 patch_merge_num_tokens=8,
                 dim_head=64,
                 dropout=0.0,
                 emb_dropout=0.0
                 ):
        super(ViT, self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_embedding = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Dense(units=dim)
        ])

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, dim]))
        self.dropout = nn.Dropout(rate=emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, patch_merge_layer, patch_merge_num_tokens)

        self.mlp_head = Sequential([
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ])

    def call(self, img, training=True, **kwargs):
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x, training=training)

        x = self.transformer(x, training=training)
        x = self.mlp_head(x)

        return x

""" Usage
v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 12,
    heads = 8,
    patch_merge_layer = 6,        # at which transformer layer to do patch merging
    patch_merge_num_tokens = 8,   # the output number of tokens from the patch merge
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = tf.random.normal(shape=[4, 256, 256, 3])
preds = v(img) # (4, 1000)

merger = PatchMerger(
    dim = 1024,
    num_tokens_out = 8   # output number of tokens
)

features = tf.random.normal(shape=[4, 256, 1024])
x = merger(features)
"""