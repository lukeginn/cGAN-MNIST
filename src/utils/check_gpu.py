import tensorflow as tf


def check_gpu():
    # Set the device configuration to use GPU
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            tf.config.set_visible_devices(physical_devices[0], "GPU")
            print("Using GPU: ", physical_devices[0])
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, using CPU")

    return physical_devices
