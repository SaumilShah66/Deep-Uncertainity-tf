import tensorflow as tf
import numpy as np
from numbers import Number

def normcdf(value, mu=0.0, stddev=1.0, dtype=tf.float32):
    sinv = (1.0 / stddev) if isinstance(stddev, Number) else tf.reciprocal(stddev)
    return 0.5 * (1.0 + tf.erf((value - mu) * sinv / tf.cast(np.sqrt(2.0), dtype)))

def _normal_log_pdf(value, mu, stddev, dtype=tf.float32):
    var = (stddev ** 2)
    pi = tf.constant(np.pi)
    log_scale = np.log(stddev) if isinstance(stddev, Number) else tf.log(stddev)
    return -((value - mu) ** 2) / (2.0*var) - log_scale - tf.log(tf.cast(tf.sqrt(2.0*pi), dtype))

# Tested against Matlab: Works correctly!
def normpdf(value, mu=0.0, stddev=1.0):
    return tf.exp(_normal_log_pdf(value, mu, stddev))