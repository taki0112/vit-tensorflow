import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from tensorflow import einsum
from einops import rearrange, repeat

from math import sqrt


def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num

def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()

        self.norm = nn.LayerNormalization()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)

class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()

        def GELU():
            def gelu(x, approximate=False):
                if approximate:
                    coeff = tf.cast(0.044715, x.dtype)
                    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
                else:
                    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

            return nn.Activation(gelu)

        self.net = [
            nn.Dense(units=hidden_dim),
            GELU(),
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ]
        self.net = Sequential(self.net)

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

        if project_out:
            self.to_out = [
                nn.Dense(units=dim),
                nn.Dropout(rate=dropout)
            ]
        else:
            self.to_out = []

        self.to_out = Sequential(self.to_out)

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x, training=training)

        return x

class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout))
            ])

    def call(self, x, training=True):
        for attn, mlp in self.layers:
            x = attn(x, training=training) + x
            x = mlp(x, training=training) + x

        return x

class Unfold(Layer):
    def __init__(self, kernel_size, stride):
        super(Unfold, self).__init__()

        self.kernel_size = [1, kernel_size, kernel_size, 1]
        self.stride = [1, stride, stride, 1]
        self.rates = [1, 1, 1, 1]

    def call(self, x, training=True):
        x = tf.image.extract_patches(x, sizes=self.kernel_size, strides=self.stride, rates=self.rates, padding='VALID')
        x = rearrange(x, 'b h w c -> b (h w) c')

        return x

# depthwise convolution, for pooling
class DepthWiseConv2d(Layer):
    def __init__(self, dim_in, dim_out, kernel_size, stride, bias=True):
        super(DepthWiseConv2d, self).__init__()

        net = []
        net += [nn.Conv2D(filters=dim_out, kernel_size=kernel_size, strides=stride, padding='SAME', groups=dim_in, use_bias=bias)]
        net += [nn.Conv2D(filters=dim_out, kernel_size=1, strides=1, use_bias=bias)]

        self.net = Sequential(net)

    def call(self, x, training=True):
        x = self.net(x)
        return x

# pooling layer
class Pool(Layer):
    def __init__(self, dim):
        super(Pool, self).__init__()
        self.downsample = DepthWiseConv2d(dim, dim*2, kernel_size=3, stride=2)
        self.cls_ff = nn.Dense(units=dim*2)

    def call(self, x, training=True):
        cls_token, tokens = x[:, :1], x[:, 1:]
        cls_token = self.cls_ff(cls_token)

        tokens = rearrange(tokens, 'b (h w) c -> b h w c', h=int(sqrt(tokens.shape[1])))
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, 'b h w c -> b (h w) c')

        x = tf.concat([cls_token, tokens], axis=1)

        return x

class PiT(Model):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 dim_head=64,
                 dropout=0.0,
                 emb_dropout=0.0,
                 ):
        super(PiT, self).__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert isinstance(depth,
                          tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'

        heads = cast_tuple(heads, len(depth))

        self.patch_embedding = Sequential([
            Unfold(kernel_size=patch_size, stride=patch_size // 2),
            nn.Dense(units=dim)
        ], name='patch_embedding')

        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size ** 2

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]))
        self.dropout = nn.Dropout(rate=emb_dropout)

        self.transformer_layers = Sequential()

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) < 1)

            self.transformer_layers.add(Transformer(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout))

            if not_last:
                self.transformer_layers.add(Pool(dim))
                dim *= 2

        self.mlp_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ], name='mlp_head')

    def call(self, img, training=True, **kwargs):
        x = self.patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x, training=training)

        x = self.transformer_layers(x, training=training)
        x = self.mlp_head(x[:, 0])

        return x

""" Usage
v = PiT(
    image_size = 224,
    patch_size = 14,
    dim = 256,
    num_classes = 1000,
    depth = (3, 3, 3),     # list of depths, indicating the number of rounds of each stage before a downsample
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
img = tf.random.normal(shape=[1, 224, 224, 3])
preds = v(img) # (1, 1000)
"""



