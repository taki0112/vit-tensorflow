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

class Identity(Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x, training=True):
        return tf.identity(x)

class Residual(Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(x, training=training) + x

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

class PatchEmbedding(Layer):
    def __init__(self, dim_out, patch_size):
        super(PatchEmbedding, self).__init__()
        self.dim_out = dim_out
        self.patch_size = patch_size
        self.proj = nn.Conv2D(filters=dim_out, kernel_size=1, strides=1)

    def call(self, fmap, training=True):
        p = self.patch_size
        fmap = rearrange(fmap, 'b (h p1) (w p2) c -> b h w (c p1 p2)', p1 = p, p2 = p)
        x = self.proj(fmap)

        return x

class PEG(Layer):
    def __init__(self, dim, kernel_size=3):
        super(PEG, self).__init__()
        self.proj = Residual(nn.Conv2D(filters=dim, kernel_size=kernel_size, strides=1, padding='SAME', groups=dim))

    def call(self, x, training=True):
        x = self.proj(x)
        return x

class LocalAttention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, patch_size=7):
        super(LocalAttention, self).__init__()
        inner_dim = dim_head * heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax()

        self.to_q = nn.Conv2D(filters=inner_dim, kernel_size=1, strides=1, use_bias=False)
        self.to_kv = nn.Conv2D(filters=inner_dim * 2, kernel_size=1, strides=1, use_bias=False)

        self.to_out = Sequential([
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            nn.Dropout(rate=dropout)
        ])

    def call(self, fmap, training=True):
        b, x, y, n = fmap.shape
        h = self.heads
        p = self.patch_size
        x, y = map(lambda t: t // p, (x, y))

        fmap = rearrange(fmap, 'b (x p1) (y p2) c -> (b x y) p1 p2 c', p1=p, p2=p)
        q = self.to_q(fmap)
        kv = self.to_kv(fmap)
        k, v = tf.split(kv, num_or_size_splits=2, axis=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p1 p2 (h d) -> (b h) (p1 p2) d', h=h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b x y h) (p1 p2) d -> b (x p1) (y p2) (h d) ', h=h, x=x, y=y, p1=p, p2=p)
        out = self.to_out(out, training=training)

        return out

class GlobalAttention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, k=7):
        super(GlobalAttention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax()

        self.to_q = nn.Conv2D(filters=inner_dim, kernel_size=1, use_bias=False)
        self.to_kv = nn.Conv2D(filters=inner_dim * 2, kernel_size=k, strides=k, use_bias=False)

        self.to_out = Sequential([
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        b, _, y, n = x.shape
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(x)
        k, v = tf.split(kv, num_or_size_splits=2, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> (b h) (x y) d', h=h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b x y (h d)', h=h, y=y)
        out = self.to_out(out, training=training)
        return out

class Transformer(Layer):
    def __init__(self, dim, depth, heads=8, dim_head=64, mlp_mult=4, local_patch_size=7, global_k=7, dropout=0.0, has_local=True):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                Residual(PreNorm(dim, LocalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, patch_size=local_patch_size))) if has_local else Identity(),
                Residual(PreNorm(dim, MLP(dim, mlp_mult, dropout=dropout))) if has_local else Identity(),
                Residual(PreNorm(dim, GlobalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, k=global_k))),
                Residual(PreNorm(dim, MLP(dim, mlp_mult, dropout=dropout)))
            ])

    def call(self, x, training=True):
        for local_attn, ff1, global_attn, ff2 in self.layers:
            x = local_attn(x, training=training)
            x = ff1(x, training=training)
            x = global_attn(x, training=training)
            x = ff2(x, training=training)

        return x

class TwinsSVT(Model):
    def __init__(self,
                 num_classes,
                 s1_emb_dim=64,
                 s1_patch_size=4,
                 s1_local_patch_size=7,
                 s1_global_k=7,
                 s1_depth=1,
                 s2_emb_dim=128,
                 s2_patch_size=2,
                 s2_local_patch_size=7,
                 s2_global_k=7,
                 s2_depth=1,
                 s3_emb_dim=256,
                 s3_patch_size=2,
                 s3_local_patch_size=7,
                 s3_global_k=7,
                 s3_depth=5,
                 s4_emb_dim=512,
                 s4_patch_size=2,
                 s4_local_patch_size=7,
                 s4_global_k=7,
                 s4_depth=4,
                 peg_kernel_size=3,
                 dropout=0.0
                 ):
        super(TwinsSVT, self).__init__()
        kwargs = dict(locals())

        self.svt_layers = Sequential()

        for prefix in ('s1', 's2', 's3', 's4'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            is_last = prefix == 's4'

            dim_next = config['emb_dim']

            self.svt_layers.add(Sequential([
                PatchEmbedding(dim_out=dim_next, patch_size=config['patch_size']),
                Transformer(dim=dim_next, depth=1, local_patch_size=config['local_patch_size'],
                            global_k=config['global_k'], dropout=dropout, has_local=not is_last),
                PEG(dim=dim_next, kernel_size=peg_kernel_size),
                Transformer(dim=dim_next, depth=config['depth'], local_patch_size=config['local_patch_size'],
                            global_k=config['global_k'], dropout=dropout, has_local=not is_last)
            ]))

        self.svt_layers.add(Sequential([
            nn.GlobalAvgPool2D(),
            nn.Dense(units=num_classes)
        ]))

    def call(self, x, training=True, **kwargs):
        x = self.svt_layers(x, training=training)
        return x

""" Usage
v = TwinsSVT(
    num_classes = 1000,       # number of output classes
    s1_emb_dim = 64,          # stage 1 - patch embedding projected dimension
    s1_patch_size = 4,        # stage 1 - patch size for patch embedding
    s1_local_patch_size = 7,  # stage 1 - patch size for local attention
    s1_global_k = 7,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
    s1_depth = 1,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
    s2_emb_dim = 128,         # stage 2 (same as above)
    s2_patch_size = 2,
    s2_local_patch_size = 7,
    s2_global_k = 7,
    s2_depth = 1,
    s3_emb_dim = 256,         # stage 3 (same as above)
    s3_patch_size = 2,
    s3_local_patch_size = 7,
    s3_global_k = 7,
    s3_depth = 5,
    s4_emb_dim = 512,         # stage 4 (same as above)
    s4_patch_size = 2,
    s4_local_patch_size = 7,
    s4_global_k = 7,
    s4_depth = 4,
    peg_kernel_size = 3,      # positional encoding generator kernel size
    dropout = 0.              # dropout
)

img = tf.random.normal(shape=[1, 224, 224, 3])
preds = v(img) # (1, 1000)
"""