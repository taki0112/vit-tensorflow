import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from vit import Transformer

from einops import rearrange, repeat
import math

def exists(val):
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

class RearrangeUnfoldTransformer(Layer):
    def __init__(self, is_first, is_last, kernel_size, stride,
                 dim, heads, depth, dim_head, mlp_dim, dropout):
        super(RearrangeUnfoldTransformer, self).__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.kernel_size = [1, kernel_size, kernel_size, 1]
        self.stride = [1, stride, stride, 1]
        self.rates = [1, 1, 1, 1]

        # transformer
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        if not self.is_last:
            self.transformer_layer = Transformer(dim=self.dim, heads=self.heads, depth=self.depth, dim_head=self.dim_head, mlp_dim=self.mlp_dim, dropout=self.dropout)


    def call(self, x, training=True):
        if not self.is_first:
            x = rearrange(x, 'b (h w) c -> b h w c', h=int(math.sqrt(x.shape[1])))
        x = tf.image.extract_patches(x, sizes=self.kernel_size, strides=self.stride, rates=self.rates, padding='SAME')
        x = rearrange(x, 'b h w c -> b (h w) c')
        if not self.is_last:
            x = self.transformer_layer(x, training=training)

        return x

class T2TViT(Model):
    def __init__(self, image_size, num_classes, dim,
                 depth=None, heads=None, mlp_dim=None, pool='cls', channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0,
                 transformer=None, t2t_layers=((7, 4), (3, 2), (3, 2))):
        super(T2TViT, self).__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = Sequential()
        layer_dim = channels
        output_image_size = image_size

        for i, (kernel_size, stride) in enumerate(t2t_layers):
            layer_dim *= kernel_size ** 2
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)

            layers.add(RearrangeUnfoldTransformer(is_first, is_last, kernel_size, stride,
                                                  dim=layer_dim, heads=1, depth=1, dim_head=layer_dim, mlp_dim=layer_dim, dropout=dropout)
            )

        layers.add(nn.Dense(units=dim))
        self.patch_embedding = layers

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, output_image_size ** 2 + 1, dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]))
        self.dropout = nn.Dropout(rate=emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool

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

        x = self.transformer(x, training=training)

        if self.pool == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x
""" Usage 
v = T2TViT(
    dim = 512,
    image_size = 224,
    depth = 5,
    heads = 8,
    mlp_dim = 512,
    num_classes = 1000,
    t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
)

img = tf.random.normal(shape=[1, 224, 224, 3])
preds = v(img) # (1, 1000)
"""