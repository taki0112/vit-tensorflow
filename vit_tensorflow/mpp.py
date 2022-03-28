import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from einops import rearrange, repeat, reduce
import numpy as np
import math
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

def exists(val):
    return val is not None

def prob_mask_like(t, prob):
    batch, seq_length, _ = t.shape
    x = tf.random.uniform([batch, seq_length], dtype=tf.float32) < prob
    return x

def get_mask_subset_with_prob(patched_input, prob):
    batch, seq_len, _ = patched_input.shape
    max_masked = math.ceil(prob * seq_len)

    rand = tf.random.uniform([batch, seq_len])
    _, sampled_indices = tf.math.top_k(rand, k=max_masked)

    new_mask = tf.zeros([batch, seq_len])
    new_mask = scatter_numpy(new_mask.numpy(), 1, sampled_indices.numpy(), 1)
    new_mask = tf.cast(new_mask, tf.bool)
    return new_mask

class MPPLoss(Layer):
    def __init__(self,
                 patch_size,
                 channels,
                 output_channel_bits,
                 max_pixel_val,
                 mean,
                 std
                 ):
        super(MPPLoss, self).__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.output_channel_bits = output_channel_bits
        self.max_pixel_val = max_pixel_val

        self.mean = tf.reshape(tf.convert_to_tensor(mean, dtype=tf.float32), [-1, 1, 1]) if mean else None
        self.std = tf.reshape(tf.convert_to_tensor(std, dtype=tf.float32), [-1, 1, 1]) if std else None

    def call(self, predicted_patches, target=None, mask=None, training=True):
        p, c, mpv, bits = self.patch_size, self.channels, self.max_pixel_val, self.output_channel_bits
        bin_size = mpv / (2 ** bits)

        # un-normalize input
        if exists(self.mean) and exists(self.std):
            target = target * self.std + self.mean

        # reshape target to patches
        target = tf.clip_by_value(target, clip_value_min=tf.reduce_min(mpv), clip_value_max=mpv) # clamp just in case
        avg_target = reduce(target, 'b (h p1) (w p2) c -> b (h w) c', 'mean', p1=p, p2=p)

        channel_bins = np.arange(bin_size, mpv, bin_size)
        discretized_target = tf.compat.v1.raw_ops.Bucketize(input=avg_target, boundaries=channel_bins)

        bin_mask = (2 ** bits) ** tf.range(0, c)
        bin_mask = rearrange(bin_mask, 'c -> () () c')

        target_label = tf.reduce_sum(bin_mask * discretized_target, axis=-1, keepdims=True)

        loss = tf.nn.softmax_cross_entropy_with_logits(tf.cast(predicted_patches[mask], tf.float32), tf.cast(target_label[mask], tf.float32))
        loss = tf.reduce_mean(loss)

        return loss

class MPP(Model):
    def __init__(self,
                 image_size,
                 transformer,
                 patch_size,
                 output_channel_bits=3,
                 channels=3,
                 max_pixel_val=1.0,
                 mask_prob=0.15,
                 replace_prob=0.5,
                 random_patch_prob=0.5,
                 mean=None,
                 std=None
                 ):
        super(MPP, self).__init__()
        # build
        transformer.build(input_shape=(20, image_size, image_size, 3))

        self.transformer = transformer
        self.loss = MPPLoss(patch_size, channels, output_channel_bits, max_pixel_val, mean, std)

        # output transformation
        self.to_bits = nn.Dense(units=2 ** (output_channel_bits * channels))

        # vit related dimensions
        self.patch_size = patch_size

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob

        # token ids
        self.mask_token = tf.Variable(tf.random.normal([1, 1, channels * patch_size ** 2]))

    def call(self, inputs, training=True, **kwargs):

        transformer = self.transformer
        # clone original image for loss
        img = tf.stop_gradient(tf.identity(inputs))

        # reshape raw image to patches
        p = self.patch_size
        inputs = rearrange(inputs,'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=p, p2=p)

        mask = get_mask_subset_with_prob(inputs, self.mask_prob)

        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = tf.stop_gradient(tf.identity(inputs))

        # if random token probability > 0 for mpp
        if self.random_patch_prob > 0:
            random_patch_sampling_prob = self.random_patch_prob / (1 - self.replace_prob)
            random_patch_prob = prob_mask_like(inputs, random_patch_sampling_prob)

            bool_random_patch_prob = tf.cast(tf.cast(mask, tf.float32) * tf.cast((random_patch_prob == True), tf.float32), tf.bool).numpy()
            random_patches = tf.random.uniform(shape=[inputs.shape[0], inputs.shape[1]], minval=0, maxval=inputs.shape[1], dtype=tf.int32)

            randomized_input = masked_input.numpy()[tf.expand_dims(tf.range(masked_input.shape[0]), axis=-1), random_patches]
            masked_input.numpy()[bool_random_patch_prob] = randomized_input[bool_random_patch_prob]

        # [mask] input
        replace_prob = prob_mask_like(inputs, self.replace_prob)
        bool_mask_replace = tf.cast(((tf.cast(mask, tf.float32) * tf.cast(replace_prob, tf.float32)) == True), tf.int32)
        masked_input.numpy()[bool_mask_replace.numpy()] = self.mask_token.numpy()

        # linear embedding of patches
        masked_input = transformer.patch_embedding.layers[-1](masked_input, training=training)

        # add cls token to input sequence
        b, n, _ = masked_input.shape
        cls_tokens = repeat(transformer.cls_token, '() n d -> b n d', b=b)
        masked_input = tf.concat([cls_tokens, masked_input], axis=1)

        # add positional embeddings to input
        masked_input += transformer.pos_embedding[:, :(n + 1)]
        masked_input = transformer.dropout(masked_input, training=training)

        # get generator output and get mpp loss
        masked_input = transformer.transformer(masked_input, training=training)
        cls_logits = self.to_bits(masked_input)
        logits = cls_logits[:, 1:, :]

        mpp_loss = self.loss(logits, img, mask)

        return mpp_loss


model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

mpp_trainer = MPP(
    image_size=256,
    transformer=model,
    patch_size=32,
    mask_prob=0.15,          # probability of using token in masked prediction task
    random_patch_prob=0.30,  # probability of randomly replacing a token being used for mpp
    replace_prob=0.50,       # probability of replacing a token being used for mpp with the mask token
)


""" Usage
def sample_unlabelled_images():
    return tf.random.normal([20, 256, 256, 3])

for _ in range(100):
    with tf.GradientTape() as tape:
        images = sample_unlabelled_images()
        loss = mpp_trainer(images)
"""