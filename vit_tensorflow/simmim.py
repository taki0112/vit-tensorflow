import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.layers as nn

from einops import repeat
import numpy as np
from vit import ViT

def scatter_numpy(x, dim, index, src):
    """
    Writes all values from the Tensor src into x at the indices specified in the index Tensor.

    :param dim: The axis along which to index
    :param index: The indices of elements to scatter
    :param src: The source element(s) to scatter
    :return: x
    """

    if x.ndim != index.ndim:
        raise ValueError("Index should have the same number of dimensions as output")
    if dim >= x.ndim or dim < -x.ndim:
        raise IndexError("dim is out of range")
    if dim < 0:
        # Not sure why scatter should accept dim < 0, but that is the behavior in PyTorch's scatter
        dim = x.ndim + dim
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = x.shape[:dim] + x.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and output should be the same size")
    if (index >= x.shape[dim]).any() or (index < 0).any():
        raise IndexError("The values of index must be between 0 and (self.shape[dim] -1)")

    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        slc = tuple(slc)
        return slc

    # We use index and dim parameters to create idx
    # idx is in a form that can be used as a NumPy advanced index for scattering of src param. in self
    idx = [[*np.indices(idx_xsection_shape).reshape(index.ndim - 1, -1),
            index[make_slice(index, dim, i)].reshape(1, -1)[0]] for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())

    if not np.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError("Dimension " + str(dim) + "of index can not be bigger than that of src ")
        src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError("Except for dimension " +
                             str(dim) + ", all dimensions of index and src should be the same size")
        # src_idx is a NumPy advanced index for indexing of elements in the src
        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(dim, np.repeat(np.arange(index.shape[dim]), np.prod(idx_xsection_shape)))
        idx = tuple(idx)
        x[idx] = src[src_idx]

    else:
        idx = tuple(idx)
        x[idx] = src

    return x

class SimMIM(Model):
    def __init__(self, image_size, encoder, masking_ratio=0.5):
        super(SimMIM, self).__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # build
        encoder.build(input_shape=(1, image_size, image_size, 3))

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.patch_embedding.layers[:2]
        pixel_values_per_patch = self.patch_to_emb.weights[0].shape[0]

        # simpler linear head
        self.mask_token = tf.Variable(tf.random.normal([encoder_dim]))
        self.to_pixels = nn.Dense(units=pixel_values_per_patch)

    def call(self, img, training=True, **kwargs):
        # get patches
        patches = self.to_patch(img, training=training)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes
        batch_range = tf.range(batch)[:, None]

        # get positions
        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches, training=training)
        tokens = tokens + pos_emb

        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked
        num_masked = int(self.masking_ratio * num_patches)

        masked_indices = tf.math.top_k(tf.random.uniform(shape=[batch, num_patches]), k=num_masked).indices
        masked_bool_mask = scatter_numpy(np.zeros(shape=[batch, num_patches]), dim=-1, index=masked_indices.numpy(), src=1)
        masked_bool_mask = tf.cast(masked_bool_mask, tf.bool)

        # mask tokens
        tokens = tf.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer
        encoded = self.encoder.transformer(tokens, training=training)

        # get the masked tokens
        encoded_mask_tokens = encoded.numpy()[batch_range, masked_indices]

        # small linear projection for predicted pixel values
        pred_pixel_values = self.to_pixels(encoded_mask_tokens, training=training)

        # get the masked patches for the final reconstruction loss
        masked_patches = patches.numpy()[batch_range, masked_indices]

        # calculate reconstruction loss
        recon_loss = tf.reduce_mean(tf.abs(pred_pixel_values - masked_patches)) / num_masked

        return recon_loss

""" Usage
v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

mim = SimMIM(
    image_size = 256,
    encoder = v,
    masking_ratio = 0.5  # they found 50% to yield the best results
)

img = tf.random.normal(shape=[8, 256, 256, 3])
loss = mim(img) # (8, 1000)
"""
