import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from tensorflow import einsum
from tensorflow.keras.preprocessing.sequence import pad_sequences

from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

import numpy as np

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# adaptive token sampling functions and classes

def log(t, eps = 1e-6):
    return tf.math.log(t + eps)

def sample_gumbel(shape, dtype, eps = 1e-6):
    u = tf.random.uniform(shape, dtype=dtype)
    return -log(-log(u, eps), eps)

def torch_gather(x, indices, gather_axis):
    # if pytorch gather indices are
    # [[[0, 10, 20], [0, 10, 20], [0, 10, 20]],
    #  [[0, 10, 20], [0, 10, 20], [0, 10, 20]]]
    # tf nd_gather needs to be
    # [[0,0,0], [0,0,10], [0,0,20], [0,1,0], [0,1,10], [0,1,20], [0,2,0], [0,2,10], [0,2,20],
    #  [1,0,0], [1,0,10], [1,0,20], [1,1,0], [1,1,10], [1,1,20], [1,2,0], [1,2,10], [1,2,20]]

    indices = tf.cast(indices, tf.int64)
    # create a tensor containing indices of each element
    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

    # splice in our pytorch style index at the correct axis
    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(gather_locations)
        else:
            gather_indices.append(all_indices[:, axis])

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = tf.tile(indices, multiples=[1] * len(indices_shape) + [*value_dims])
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    dim += value_expand_len

    values = torch_gather(values, indices, dim)
    return values

class AdaptiveTokenSampling(Layer):
    def __init__(self, output_num_tokens, eps=1e-6):
        super(AdaptiveTokenSampling, self).__init__()
        self.eps = eps
        self.output_num_tokens = output_num_tokens

    def call(self, attn, value=None, mask=None, training=True):
        heads, output_num_tokens, eps, dtype = attn.shape[1], self.output_num_tokens, self.eps, attn.dtype

        # first get the attention values for CLS token to all other tokens
        cls_attn = attn[..., 0, 1:]

        # calculate the norms of the values, for weighting the scores, as described in the paper
        value_norms = tf.norm(value[..., 1:, :], axis=-1)

        # weigh the attention scores by the norm of the values, sum across all heads
        cls_attn = einsum('b h n, b h n -> b n', cls_attn, value_norms)

        # normalize to 1
        normed_cls_attn = cls_attn / (tf.reduce_sum(cls_attn, axis=-1, keepdims=True) + eps)

        # instead of using inverse transform sampling, going to invert the softmax and use gumbel-max sampling instead
        pseudo_logits = log(normed_cls_attn)

        # mask out pseudo logits for gumbel-max sampling
        mask_without_cls = mask[:, 1:]
        mask_value = -np.finfo(attn.dtype.as_numpy_dtype).max / 2
        pseudo_logits = tf.where(~mask_without_cls, mask_value, pseudo_logits)

        # expand k times, k being the adaptive sampling number
        pseudo_logits = repeat(pseudo_logits, 'b n -> b k n', k=output_num_tokens)
        pseudo_logits = pseudo_logits + sample_gumbel(pseudo_logits.shape, dtype=dtype)

        # gumble-max and add one to reserve 0 for padding / mask
        sampled_token_ids = tf.argmax(pseudo_logits, axis=-1) + 1

        # calculate unique using torch.unique and then pad the sequence from the right
        unique_sampled_token_ids_list = []
        for t in tf.unstack(sampled_token_ids):
            t = tf.cast(t, tf.int32)
            t, _ = tf.unique(t)
            x = tf.sort(t)
            unique_sampled_token_ids_list.append(x)


        unique_sampled_token_ids = pad_sequences(unique_sampled_token_ids_list)

        # calculate the new mask, based on the padding
        new_mask = unique_sampled_token_ids != 0

        # CLS token never gets masked out (gets a value of True)
        new_mask = tf.pad(new_mask, paddings=[[0, 0], [1, 0]], constant_values=True)

        # prepend a 0 token id to keep the CLS attention scores
        unique_sampled_token_ids = tf.pad(unique_sampled_token_ids, paddings=[[0, 0], [1, 0]])
        expanded_unique_sampled_token_ids = repeat(unique_sampled_token_ids, 'b n -> b h n', h=heads)

        # gather the new attention scores
        new_attn = batched_index_select(attn, expanded_unique_sampled_token_ids, dim=2)

        # return the sampled attention scores, new mask (denoting padding), as well as the sampled token indices (for the residual)
        return new_attn, new_mask, unique_sampled_token_ids

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

class Attention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, output_num_tokens=None):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax()
        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False)

        self.output_num_tokens = output_num_tokens
        self.ats = AdaptiveTokenSampling(output_num_tokens) if exists(output_num_tokens) else None

        self.to_out = Sequential([
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, mask=None, training=True):
        num_tokens = x.shape[1]

        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d)-> b h n d', h=self.heads), qkv)

        dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale

        if exists(mask):
            mask_f = tf.cast(mask, tf.float32)
            dots_mask = rearrange(mask_f, 'b i -> b 1 i 1') * rearrange(mask_f, 'b j -> b 1 1 j')
            dots_mask = tf.cast(dots_mask, tf.bool)
            mask_value = -np.finfo(dots.dtype.as_numpy_dtype).max
            dots = tf.where(~dots_mask, mask_value, dots)

        attn = self.attend(dots)

        sampled_token_ids = None

        # if adaptive token sampling is enabled
        # and number of tokens is greater than the number of output tokens
        if exists(self.output_num_tokens) and (num_tokens - 1) > self.output_num_tokens:
            attn, mask, sampled_token_ids = self.ats(attn, v, mask=mask)

        out = tf.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out, training=training)

        return out, mask, sampled_token_ids

class Transformer(Layer):
    def __init__(self, dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()
        assert len(max_tokens_per_depth) == depth, 'max_tokens_per_depth must be a tuple of length that is equal to the depth of the transformer'
        assert sorted(max_tokens_per_depth, reverse=True) == list(max_tokens_per_depth), 'max_tokens_per_depth must be in decreasing order'
        assert min(max_tokens_per_depth) > 0, 'max_tokens_per_depth must have at least 1 token at any layer'

        self.layers = []
        for _, output_num_tokens in zip(range(depth), max_tokens_per_depth):
            self.layers.append([
                PreNorm(Attention(dim, output_num_tokens=output_num_tokens, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout))
            ])

    def call(self, x, training=True):
        b, n = x.shape[:2]

        # use mask to keep track of the paddings when sampling tokens
        # as the duplicates (when sampling) are just removed, as mentioned in the paper
        mask = tf.ones([b, n], dtype=tf.bool)

        token_ids = tf.range(n)
        token_ids = repeat(token_ids, 'n -> b n', b = b)

        for attn, ff in self.layers:
            attn_out, mask, sampled_token_ids = attn(x, mask=mask, training=training)

            # when token sampling, one needs to then gather the residual tokens with the sampled token ids
            if exists(sampled_token_ids):
                x = batched_index_select(x, sampled_token_ids, dim=1)
                token_ids = batched_index_select(token_ids, sampled_token_ids, dim=1)

            x = x + attn_out

            x = ff(x, training=training) + x

        return x, token_ids

class ViT(Model):
    def __init__(self,
                 image_size, 
                 patch_size, 
                 num_classes, 
                 dim, 
                 depth, 
                 max_tokens_per_depth, 
                 heads, 
                 mlp_dim,
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
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]))
        self.dropout = nn.Dropout(rate=emb_dropout)

        self.transformer = Transformer(dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ])


    def call(self, img, return_sampled_token_ids=False, training=True, **kwargs):
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x, training=training)

        x, token_ids = self.transformer(x, training=training)

        logits = self.mlp_head(x[:, 0])

        if return_sampled_token_ids:
            # remove CLS token and decrement by 1 to make -1 the padding
            token_ids = token_ids[:, 1:] - 1
            return logits, token_ids

        return logits

v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    max_tokens_per_depth = (256, 128, 64, 32, 16, 8), # a tuple that denotes the maximum number of tokens that any given layer should have. if the layer has greater than this amount, it will undergo adaptive token sampling
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = tf.random.normal(shape=[4, 256, 256, 3])
preds = v(img) # (1, 1000)
print(preds.shape)