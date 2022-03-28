import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

import numpy as np

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def shift(x):
    b, h, w, c = x.shape
    shifted_x = []

    shifts = [1, -1] # [shift, axis]

    # width
    z = tf.zeros([b, h, 1, c], dtype=tf.float32)
    for idx, shift in enumerate(shifts):
        if idx == 0:
            s = tf.roll(x, shift, axis=2)[:, :, shift:, :]
            concat = tf.concat([z, s], axis=2)


        else:
            s = tf.roll(x, shift, axis=2)[:, :, :shift, :]
            concat = tf.concat([s, z], axis=2)

        shifted_x.append(concat)

    # height
    z = tf.zeros([b, 1, w, c], dtype=tf.float32)
    for idx, shift in enumerate(shifts):
        if idx == 0:
            s = tf.roll(x, shift, axis=1)[:, shift:, :, :]
            concat = tf.concat([z, s], axis=1)
        else:
            s = tf.roll(x, shift, axis=1)[:, :shift, :, :]
            concat = tf.concat([s, z], axis=1)

        shifted_x.append(concat)

    return shifted_x

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

class LSA(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(LSA, self).__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = tf.Variable(tf.math.log(dim_head ** -0.5))

        self.attend = nn.Softmax()
        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False)

        self.to_out = Sequential([
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * tf.math.exp(self.temperature)

        mask = tf.eye(dots.shape[-1], dtype=tf.bool)
        mask_value = -np.finfo(dots.dtype.as_numpy_dtype).max
        dots = tf.where(mask, mask_value, dots)

        attn = self.attend(dots)

        out = tf.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out, training=training)

        return out

class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout))
            ])

    def call(self, x, training=True):
        for attn, ff in self.layers:
            x = attn(x, training=training) + x
            x = ff(x, training=training) + x

        return x

class SPT(Layer):
    def __init__(self, dim, patch_size):
        super(SPT, self).__init__()

        self.to_patch_tokens = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNormalization(),
            nn.Dense(units=dim)
        ])

    def call(self, x, training=True):
        shifted_x = shift(x)
        x_with_shifts = tf.concat([x, *shifted_x], axis=-1)
        x = self.to_patch_tokens(x_with_shifts)

        return x

class ViT(Model):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 dim_head=64,
                 dropout=0.0,
                 emb_dropout=0.0
                 ):
        super(ViT, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = SPT(dim=dim, patch_size=patch_size)

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]))
        self.dropout = nn.Dropout(rate=emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ])

    def call(self, img, training=True, **kwargs):
        x = self.patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x, training=training)

        x = self.transformer(x, training=training)

        if self.pool == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x

""" Usage 
v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = tf.random.normal(shape=[4, 256, 256, 3])
preds = v(img) # (4, 1000)

spt = SPT(
    dim = 1024,
    patch_size = 16
)

tokens = spt(img) # (4, 256, 1024)
"""
