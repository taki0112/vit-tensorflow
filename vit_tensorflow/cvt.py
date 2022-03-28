import tensorflow as tf
from tensorflow import einsum
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from einops import rearrange

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

class LayerNorm(Layer): # layernorm, but done in the channel dimension #1
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

class DepthWiseConv2d(Layer):
    def __init__(self, dim_in, dim_out, kernel_size, stride, bias=True):
        super(DepthWiseConv2d, self).__init__()

        net = []
        net += [nn.Conv2D(filters=dim_in, kernel_size=kernel_size, strides=stride, padding='SAME', groups=dim_in, use_bias=bias)]
        net += [nn.BatchNormalization(momentum=0.9, epsilon=1e-5)]
        net += [nn.Conv2D(filters=dim_out, kernel_size=1, strides=1, use_bias=bias)]

        self.net = Sequential(net)

    def call(self, x, training=True):
        x = self.net(x, training=training)
        return x

class Attention(Layer):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax()

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, stride=1, bias=False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, stride=kv_proj_stride, bias=False)

        self.to_out = Sequential([
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        b, _, y, n = x.shape
        h = self.heads
        q = self.to_q(x, training=training)
        kv = self.to_kv(x, training=training)
        k, v = tf.split(kv, num_or_size_splits=2, axis=-1)
        qkv = (q, k, v)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> (b h) (x y) d', h=h), qkv)

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = self.attend(dots)

        x = einsum('b i j, b j d -> b i d', attn, v)
        x = rearrange(x, '(b h) (x y) d -> b x y (h d)', h=h, y=y)
        x = self.to_out(x, training=training)

        return x

class Transformer(Layer):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(dim, Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, MLP(dim, mlp_mult, dropout=dropout))
            ])


    def call(self, x, training=True):
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x, training=training) + x
            x = ff(x, training=training) + x

        return x

class CvT(Model):
    def __init__(self,
                 num_classes,
                 s1_emb_dim=64,
                 s1_emb_kernel=7,
                 s1_emb_stride=4,
                 s1_proj_kernel=3,
                 s1_kv_proj_stride=2,
                 s1_heads=1,
                 s1_depth=1,
                 s1_mlp_mult=4,
                 s2_emb_dim=192,
                 s2_emb_kernel=3,
                 s2_emb_stride=2,
                 s2_proj_kernel=3,
                 s2_kv_proj_stride=2,
                 s2_heads=3,
                 s2_depth=2,
                 s2_mlp_mult=4,
                 s3_emb_dim=384,
                 s3_emb_kernel=3,
                 s3_emb_stride=2,
                 s3_proj_kernel=3,
                 s3_kv_proj_stride=2,
                 s3_heads=6,
                 s3_depth=10,
                 s3_mlp_mult=4,
                 dropout=0.
                 ):

        super(CvT, self).__init__()
        kwargs = dict(locals())

        self.cvt_layers = Sequential()

        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            self.cvt_layers.add(Sequential([
                nn.Conv2D(filters=config['emb_dim'], kernel_size=config['emb_kernel'], padding='SAME', strides=config['emb_stride']),
                LayerNorm(config['emb_dim']),
                Transformer(dim=config['emb_dim'], proj_kernel=config['proj_kernel'],
                            kv_proj_stride=config['kv_proj_stride'], depth=config['depth'], heads=config['heads'],
                            mlp_mult=config['mlp_mult'], dropout=dropout)
            ]))


        self.cvt_layers.add(Sequential([
            nn.GlobalAvgPool2D(),
            nn.Dense(units=num_classes)
       ]))

    def call(self, img, training=True, **kwargs):
        x = self.cvt_layers(img, training=training)
        return x

""" Usage 
v = CvT(
    num_classes = 1000,
    s1_emb_dim = 64,        # stage 1 - dimension
    s1_emb_kernel = 7,      # stage 1 - conv kernel
    s1_emb_stride = 4,      # stage 1 - conv stride
    s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
    s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
    s1_heads = 1,           # stage 1 - heads
    s1_depth = 1,           # stage 1 - depth
    s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
    s2_emb_dim = 192,       # stage 2 - (same as above)
    s2_emb_kernel = 3,
    s2_emb_stride = 2,
    s2_proj_kernel = 3,
    s2_kv_proj_stride = 2,
    s2_heads = 3,
    s2_depth = 2,
    s2_mlp_mult = 4,
    s3_emb_dim = 384,       # stage 3 - (same as above)
    s3_emb_kernel = 3,
    s3_emb_stride = 2,
    s3_proj_kernel = 3,
    s3_kv_proj_stride = 2,
    s3_heads = 4,
    s3_depth = 10,
    s3_mlp_mult = 4,
    dropout = 0.
)

img = tf.random.normal(shape=[1, 224, 224, 3])
preds = v(img) # (1, 1000)
"""
