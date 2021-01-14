# This gets rid of NumPy FutureWarnings that occur at TF import
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# This gets rid of TF 2.0 related deprecation warnings ############ tf.compat.v1 is only compatible with tf >= 1.14. However, the requirements are still 1.9 <= tf <= 1.14
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
