# Use the following code to get your computer get for GPU deep learning
# conda update --force conda
# conda create --name gpu_env keras tensorflow cudatoolkit cudnn
# conda activate gpu_env
# pip install tensorflow-addons

# Test 1
print("hi")

# Test 2
import tensorflow as tf

Image_Input = tf.keras.layers.Input(shape=(28, 28, 1))

# Test 3
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.keras import backend
from tensorflow.compat.v1 import ConfigProto, Session

print(device_lib.list_local_devices())
print("Num of GPUs available: ", len(tf.test.gpu_device_name()))
if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
