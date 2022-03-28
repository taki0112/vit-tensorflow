import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from tensorflow import einsum

from einops import rearrange
from einops.layers.tensorflow import Rearrange, Reduce

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def divisible_by(val, d):
    return (val % d) == 0

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
    def __init__(self, dim, mult=4, dropout=0.0):
        super(MLP, self).__init__()

        self.net = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=dim * mult),
            GELU(),
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.0):
        super(Attention, self).__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNormalization()
        self.attend = nn.Softmax()
        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False)

        self.to_out = nn.Dense(units=dim)

    def call(self, x, rel_pos_bias=None, training=True):
        h = self.heads

        # prenorm
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add relative positional bias for local tokens
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias
        attn = self.attend(sim)

        # merge heads

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x)

        return x

class R2LTransformer(Layer):
    def __init__(self, dim, window_size, depth=4, heads=4, dim_head=32, attn_dropout=0.0, ff_dropout=0.0):
        super(R2LTransformer, self).__init__()

        self.layers = []

        self.window_size = window_size
        rel_positions = 2 * window_size - 1
        self.local_rel_pos_bias = nn.Embedding(rel_positions ** 2, heads)

        for _ in range(depth):
            self.layers.append([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                MLP(dim, dropout=ff_dropout)
            ])


    def call(self, local_tokens, region_tokens=None, training=True):
        lh, lw = local_tokens.shape[1:3]
        rh, rw = region_tokens.shape[1:3]
        window_size_h, window_size_w = lh // rh, lw // rw

        local_tokens = rearrange(local_tokens, 'b h w c -> b (h w) c')
        region_tokens = rearrange(region_tokens, 'b h w c -> b (h w) c')

        # calculate local relative positional bias
        h_range = tf.range(window_size_h)
        w_range = tf.range(window_size_w)

        grid_x, grid_y = tf.meshgrid(h_range, w_range, indexing='ij')
        grid = tf.stack([grid_x, grid_y])
        grid = rearrange(grid, 'c h w -> c (h w)')
        grid = (grid[:, :, None] - grid[:, None, :]) + (self.window_size - 1)

        bias_indices = tf.reduce_sum((grid * tf.convert_to_tensor([1, self.window_size * 2 - 1])[:, None, None]), axis=0)
        rel_pos_bias = self.local_rel_pos_bias(bias_indices)
        rel_pos_bias = rearrange(rel_pos_bias, 'i j h -> () h i j')
        rel_pos_bias = tf.pad(rel_pos_bias, paddings=[[0, 0], [0, 0], [1, 0], [1, 0]])

        # go through r2l transformer layers
        for attn, ff in self.layers:
            region_tokens = attn(region_tokens) + region_tokens

            # concat region tokens to local tokens

            local_tokens = rearrange(local_tokens, 'b (h w) d -> b h w d', h=lh)
            local_tokens = rearrange(local_tokens, 'b (h p1) (w p2) d -> (b h w) (p1 p2) d', p1=window_size_h, p2=window_size_w)
            region_tokens = rearrange(region_tokens, 'b n d -> (b n) () d')

            # do self attention on local tokens, along with its regional token
            region_and_local_tokens = tf.concat([region_tokens, local_tokens], axis=1)
            region_and_local_tokens = attn(region_and_local_tokens, rel_pos_bias=rel_pos_bias) + region_and_local_tokens

            # feedforward
            region_and_local_tokens = ff(region_and_local_tokens, training=training) + region_and_local_tokens

            # split back local and regional tokens
            region_tokens, local_tokens = region_and_local_tokens[:, :1], region_and_local_tokens[:, 1:]
            local_tokens = rearrange(local_tokens, '(b h w) (p1 p2) d -> b (h p1 w p2) d', h=lh // window_size_h, w=lw // window_size_w, p1=window_size_h)
            region_tokens = rearrange(region_tokens, '(b n) () d -> b n d', n=rh * rw)

        local_tokens = rearrange(local_tokens, 'b (h w) c -> b h w c', h=lh, w=lw)
        region_tokens = rearrange(region_tokens, 'b (h w) c -> b h w c', h=rh, w=rw)

        return local_tokens, region_tokens

class RegionViT(Model):
    def __init__(self,
                 dim=(64, 128, 256, 512),
                 depth=(2, 2, 8, 2),
                 window_size=7,
                 num_classes=1000,
                 tokenize_local_3_conv=False,
                 local_patch_size=4,
                 use_peg=False,
                 attn_dropout=0.0,
                 ff_dropout=0.0,
                 ):
        super(RegionViT, self).__init__()
        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        assert len(dim) == 4, 'dim needs to be a single value or a tuple of length 4'
        assert len(depth) == 4, 'depth needs to be a single value or a tuple of length 4'

        self.local_patch_size = local_patch_size

        region_patch_size = local_patch_size * window_size
        self.region_patch_size = local_patch_size * window_size

        init_dim, *_, last_dim = dim

        # local and region encoders
        if tokenize_local_3_conv:
            self.local_encoder = Sequential([
                nn.Conv2D(filters=init_dim, kernel_size=3, strides=2, padding='SAME'),
                nn.LayerNormalization(),
                GELU(),
                nn.Conv2D(filters=init_dim, kernel_size=3, strides=2, padding='SAME'),
                nn.LayerNormalization(),
                GELU(),
                nn.Conv2D(filters=init_dim, kernel_size=3, strides=1, padding='SAME')
            ])
        else:
            self.local_encoder = nn.Conv2D(filters=init_dim, kernel_size=8, strides=4, padding='SAME')

        self.region_encoder = Sequential([
            Rearrange('b (h p1) (w p2) c -> b h w (c p1 p2) ', p1=region_patch_size, p2=region_patch_size),
            nn.Conv2D(filters=init_dim, kernel_size=1, strides=1)
        ])

        # layers
        self.region_layers = []

        for ind, dim, num_layers in zip(range(4), dim, depth):
            not_first = ind != 0
            need_downsample = not_first
            need_peg = not_first and use_peg

            self.region_layers.append([
                Downsample(dim) if need_downsample else Identity(),
                PEG(dim) if need_peg else Identity(),
                R2LTransformer(dim, depth=num_layers, window_size=window_size, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            ])

        # final logits
        self.to_logits = Sequential([
            Reduce('b h w c -> b c', 'mean'),
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ])

    def call(self, x, training=True, **kwargs):
        _, h, w, _ = x.shape
        assert divisible_by(h, self.region_patch_size) and divisible_by(w, self.region_patch_size), 'height and width must be divisible by region patch size'
        assert divisible_by(h, self.local_patch_size) and divisible_by(w, self.local_patch_size), 'height and width must be divisible by local patch size'

        local_tokens = self.local_encoder(x)
        region_tokens = self.region_encoder(x)

        for down, peg, transformer in self.region_layers:
            local_tokens, region_tokens = down(local_tokens), down(region_tokens)
            local_tokens = peg(local_tokens)
            local_tokens, region_tokens = transformer(local_tokens, region_tokens, training=training)

        x = self.to_logits(region_tokens)
        return x

""" Usage 
v = RegionViT(
    dim = (64, 128, 256, 512),      # tuple of size 4, indicating dimension at each stage
    depth = (2, 2, 8, 2),           # depth of the region to local transformer at each stage
    window_size = 7,                # window size, which should be either 7 or 14
    num_classes = 1000,             # number of output classes
    tokenize_local_3_conv = False,  # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
    use_peg = False,                # whether to use positional generating module. they used this for object detection for a boost in performance
)

img = tf.random.normal(shape=[1, 224, 224, 3])
preds = v(img) # (1, 1000)
"""