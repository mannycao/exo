import tensorflow as tf

# Configure TensorFlow for Metal
try:
    # Enable Metal backend
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Metal device found and configured")
    else:
        print("No Metal device found, using CPU")
except Exception as e:
    print(f"Error configuring Metal: {e}")
    print("Falling back to CPU")
