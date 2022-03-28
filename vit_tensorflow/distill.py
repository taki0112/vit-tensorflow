import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from einops import rearrange, repeat

from vit import ViT
from t2t import T2TViT
from efficient import ViT as EfficientViT

def exists(val):
    return val is not None

class DistillMixin:
    def call(self, img, distill_token=None, training=True):
        distilling = exists(distill_token)
        x = self.patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]

        if distilling:
            distill_tokens = repeat(distill_token, '() n d -> b n d', b = b)
            x = tf.concat([x, distill_tokens], axis=1)

        x = self._attend(x, training=training)

        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]

        if self.pool == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        if distilling:
            return x, distill_tokens
        else:
            return x

class DistillableViT(DistillMixin, ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def _attend(self, x, training=True):
        x = self.dropout(x, training=training)
        x = self.transformer(x, training=training)
        return x


class DistillableT2TViT(DistillMixin, T2TViT):
    def __init__(self, *args, **kwargs):
        super(DistillableT2TViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def _attend(self, x, training=True):
        x = self.dropout(x, training=training)
        x = self.transformer(x, training=training)
        return x

class DistillableEfficientViT(DistillMixin, EfficientViT):
    def __init__(self, *args, **kwargs):
        super(DistillableEfficientViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def _attend(self, x, training=True):
        x = self.dropout(x, training=training)
        x = self.transformer(x, training=training)
        return x

class DistillWrapper(Model):
    def __init__(self, teacher, student, temperature=1.0, alpha=0.5, hard=False):
        super(DistillWrapper, self).__init__()

        assert (isinstance(student, (DistillableViT, DistillableT2TViT, DistillableEfficientViT))), 'student must be a vision transformer'

        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard
        self.distillation_token = tf.Variable(tf.random.normal([1, 1, dim]))

        self.distill_mlp = Sequential([
                nn.LayerNormalization(),
                nn.Dense(units=num_classes)
        ], name='distill_mlp')

    def call(self, inputs, temperature=None, alpha=None, training=True, **kwargs):
        img, labels = inputs
        b, *_ = img.shape
        alpha = alpha if exists(alpha) else self.alpha
        T = temperature if exists(temperature) else self.temperature

        teacher_logits = tf.stop_gradient(self.teacher(img, training=training))

        student_logits, distill_tokens = self.student(img, distill_token=self.distillation_token, training=training)
        distill_logits = self.distill_mlp(distill_tokens)

        loss = tf.keras.losses.categorical_crossentropy(y_true=labels, y_pred=student_logits, from_logits=True)

        if not self.hard:
            x = tf.nn.log_softmax(distill_logits / T, axis=-1)
            y = tf.nn.softmax(teacher_logits / T, axis=-1)
            distill_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(y_true=y, y_pred=x)

            batch = distill_loss.shape[0]
            distill_loss = tf.reduce_sum(distill_loss) / batch

            distill_loss *= T ** 2
        else:
            teacher_labels = tf.argmax(teacher_logits, axis=-1)
            distill_loss = tf.keras.losses.categorical_crossentropy(y_true=teacher_labels, y_pred=distill_logits, from_logits=True)

        return loss * (1 - alpha) + distill_loss * alpha


""" Usage
teacher = tf.keras.applications.resnet50.ResNet50()

v = DistillableViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

distiller = DistillWrapper(
    student = v,
    teacher = teacher,
    temperature = 3,           # temperature of distillation
    alpha = 0.5,               # trade between main loss and distillation loss
    hard = False               # whether to use soft or hard distillation
)

img = tf.random.normal([2, 256, 256, 3])
labels = tf.random.uniform(shape=[2, ], minval=0, maxval=1000, dtype=tf.int32)
labels = tf.one_hot(labels, depth=1000, axis=-1)

loss = distiller([img, labels])
"""

