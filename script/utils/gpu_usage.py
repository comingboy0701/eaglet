import os
# import tensorflow as tf


# def limit_gpu_memory(memory_limit, gpu_no=0):
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#         tf.config.experimental.set_visible_devices(gpus[gpu_no], 'GPU')
#         try:
#             tf.config.experimental.set_virtual_device_configuration(
#                 gpus[gpu_no],
#                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#         except RuntimeError as e:
#             # Virtual devices must be set before GPUs have been initialized
#             print(e)
