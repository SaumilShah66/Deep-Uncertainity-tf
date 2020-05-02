import tensorflow as tf
import numpy as np
from numbers import Number

def normcdf(value, mu=0.0, stddev=1.0, dtype=tf.float32):
    # sinv = (1.0 / stddev) if isinstance(stddev, Number) else tf.reciprocal(stddev)
    sinv = tf.reciprocal(stddev)
    return tf.constant(0.5, dtype) * (tf.constant(1.0, dtype) + tf.erf((value - mu) * sinv / tf.constant(np.sqrt(2.0), dtype)))

def _normal_log_pdf(value, mu, stddev, dtype=tf.float32):
    var = (stddev ** 2)
    pi = tf.constant(np.pi)
    # log_scale = np.log(stddev) if isinstance(stddev, Number) else tf.log(stddev)
    log_scale = tf.log(stddev)
    return -((value - mu) ** 2) / (tf.constant(2.0)*var) - log_scale - tf.log(tf.cast(tf.sqrt(2.0*pi), dtype))

# Tested against Matlab: Works correctly!
def normpdf(value, mu=0.0, stddev=1.0):
    return tf.exp(_normal_log_pdf(value, mu, stddev))