import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from einops import repeat
from vit import Transformer, ViT

class Identity(Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x, training=True):
        return tf.identity(x)

class MAE(Model):
    def __init__(self,
                 image_size,
                 encoder,
                 decoder_dim,
                 masking_ratio=0.75,
                 decoder_depth=1,
                 decoder_heads=8,
                 decoder_dim_head=64
                 ):
        super(MAE, self).__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # build
        encoder.build(input_shape=(1, image_size, image_size, 3))

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.patch_embedding.layers[:2]
        pixel_values_per_patch = self.patch_to_emb.weights[0].shape[0]

        # decoder parameters
        self.enc_to_dec = nn.Dense(units=decoder_dim) if encoder_dim != decoder_dim else Identity()
        self.mask_token = tf.Variable(tf.random.normal([decoder_dim]))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head, mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Dense(units=pixel_values_per_patch)

    def call(self, img, training=True, **kwargs):
        # get patches
        patches = self.to_patch(img, training=training)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches, training=training)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = tf.argsort(tf.random.uniform([batch, num_patches]), axis=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = tf.range(batch)[:, None]
        tokens = tokens.numpy()[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches.numpy()[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens, training=training)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens, training=training)

        # reapply decoder position embedding to unmasked tokens
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices, training=training)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices, training=training)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = tf.concat([mask_tokens, decoder_tokens], axis=1)
        decoded_tokens = self.decoder(decoder_tokens, training=training)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens, training=training)

        # calculate reconstruction loss
        recon_loss = tf.reduce_mean(tf.square(pred_pixel_values, masked_patches))

        return recon_loss

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

mae = MAE(
    image_size = 256,
    encoder = v,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)

img = tf.random.normal(shape=[8, 256, 256, 3])
loss = mae(img)
print(loss)