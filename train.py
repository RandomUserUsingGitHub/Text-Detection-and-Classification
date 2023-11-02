import tensorflow as tf
import numpy as np
import pandas as pd

from Generator.Utils.FileUtils import define_csv_name

df = pd.read_csv(define_csv_name())
# classes = 

model = tf.keras.models.Sequential{
    tf.keras.layers.Dense(input_shape=)
}