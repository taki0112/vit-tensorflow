import tensorflow as tf
from tensorflow import einsum
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()

        self.norm = nn.LayerNormalization()
        self.fn = fn

    def call(self, x, training=True, **kwargs):
        return self.fn(self.norm(x), training=training, **kwargs)

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

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax()
        self.to_q = nn.Dense(units=inner_dim, use_bias=False)
        self.to_kv = nn.Dense(units=inner_dim * 2, use_bias=False)

        self.to_out = [
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ]

        self.to_out = Sequential(self.to_out)

    def call(self, x, context=None, kv_include_self=False, training=True):

        context = default(context, x)

        if kv_include_self:
            context = tf.concat([x, context], axis=1) # cross attention requires CLS token includes itself as key / value

        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = tf.split(kv, num_or_size_splits=2, axis=-1)
        qkv = (q, k, v)

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
        self.norm = nn.LayerNormalization()

        for _ in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout))
            ])

    def call(self, x, training=True):
        for attn, mlp in self.layers:
            x = attn(x, training=training) + x
            x = mlp(x, training=training) + x

        x = self.norm(x)

        return x

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
class ProjectInOut(Layer):
    def __init__(self, dim_in, dim_out, fn):
        super(ProjectInOut, self).__init__()
        self.fn = fn

        self.need_projection = dim_in != dim_out
        if self.need_projection:
            self.project_in = nn.Dense(units=dim_out)
            self.project_out = nn.Dense(units=dim_in)

    def call(self, x, training=True, *args, **kwargs):
        # args check
        if self.need_projection:
            x = self.project_in(x)

        x = self.fn(x, training=training, *args, **kwargs)

        if self.need_projection:
            x = self.project_out(x)

        return x

# cross attention transformer
class CrossTransformer(Layer):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super(CrossTransformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([ProjectInOut(sm_dim, lg_dim, PreNorm(Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                                ProjectInOut(lg_dim, sm_dim, PreNorm(Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout)))]
                               )

    def call(self, inputs, training=True):
        sm_tokens, lg_tokens = inputs
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context=lg_patch_tokens, kv_include_self=True, training=training) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context=sm_patch_tokens, kv_include_self=True, training=training) + lg_cls

        sm_tokens = tf.concat([sm_cls, sm_patch_tokens], axis=1)
        lg_tokens = tf.concat([lg_cls, lg_patch_tokens], axis=1)

        return sm_tokens, lg_tokens

# multi-scale encoder
class MultiScaleEncoder(Layer):
    def __init__(self,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head=64,
        dropout=0.0):
        super(MultiScaleEncoder, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                                Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                                CrossTransformer(sm_dim=sm_dim, lg_dim=lg_dim,
                                                 depth=cross_attn_depth, heads=cross_attn_heads, dim_head=cross_attn_dim_head, dropout=dropout)
                                ]
            )


    def call(self, inputs, training=True):
        sm_tokens, lg_tokens = inputs
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens, training=training), lg_enc(lg_tokens, training=training)
            sm_tokens, lg_tokens = cross_attend([sm_tokens, lg_tokens], training=training)

        return sm_tokens, lg_tokens

# patch-based image to token embedder
class ImageEmbedder(Layer):
    def __init__(self,
                 dim,
                 image_size,
                 patch_size,
                 dropout=0.0):
        super(ImageEmbedder, self).__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Dense(units=dim)
        ], name='patch_embedding')

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]))
        self.dropout = nn.Dropout(rate=dropout)

    def call(self, x, training=True):
        x = self.patch_embedding(x)

        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x, training=training)

        return x

# cross ViT class
class CrossViT(Model):
    def __init__(self,
                 image_size,
                 num_classes,
                 sm_dim,
                 lg_dim,
                 sm_patch_size=12,
                 sm_enc_depth=1,
                 sm_enc_heads=8,
                 sm_enc_mlp_dim=2048,
                 sm_enc_dim_head=64,
                 lg_patch_size=16,
                 lg_enc_depth=4,
                 lg_enc_heads=8,
                 lg_enc_mlp_dim=2048,
                 lg_enc_dim_head=64,
                 cross_attn_depth=2,
                 cross_attn_heads=8,
                 cross_attn_dim_head=64,
                 depth=3,
                 dropout=0.1,
                 emb_dropout=0.1):
        super(CrossViT, self).__init__()
        self.sm_image_embedder = ImageEmbedder(dim=sm_dim, image_size=image_size, patch_size=sm_patch_size, dropout=emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim=lg_dim, image_size=image_size, patch_size=lg_patch_size, dropout=emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth,
            sm_dim=sm_dim,
            lg_dim=lg_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            sm_enc_params=dict(
                depth=sm_enc_depth,
                heads=sm_enc_heads,
                mlp_dim=sm_enc_mlp_dim,
                dim_head=sm_enc_dim_head
            ),
            lg_enc_params=dict(
                depth=lg_enc_depth,
                heads=lg_enc_heads,
                mlp_dim=lg_enc_mlp_dim,
                dim_head=lg_enc_dim_head
            ),
            dropout=dropout
        )

        self.sm_mlp_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ], name='sm_mlp_head')

        self.lg_mlp_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ], name='lg_mlp_head')

    def call(self, img, training=True, **kwargs):
        sm_tokens = self.sm_image_embedder(img, training=training)
        lg_tokens = self.lg_image_embedder(img, training=training)

        sm_tokens, lg_tokens = self.multi_scale_encoder([sm_tokens, lg_tokens], training=training)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        x = sm_logits + lg_logits

        return x

""" Usage 
v = CrossViT(
    image_size = 256,
    num_classes = 1000,
    depth = 4,               # number of multi-scale encoding blocks
    sm_dim = 192,            # high res dimension
    sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
    sm_enc_depth = 2,        # high res depth
    sm_enc_heads = 8,        # high res heads
    sm_enc_mlp_dim = 2048,   # high res feedforward dimension
    lg_dim = 384,            # low res dimension
    lg_patch_size = 64,      # low res patch size
    lg_enc_depth = 3,        # low res depth
    lg_enc_heads = 8,        # low res heads
    lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
    cross_attn_depth = 2,    # cross attention rounds
    cross_attn_heads = 8,    # cross attention heads
    dropout = 0.1,
    emb_dropout = 0.1
)

img = tf.random.normal(shape=[1, 256, 256, 3])
preds = v(img) # (1, 1000)
"""