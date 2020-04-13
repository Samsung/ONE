# We may need to support multiple tensorflow version
#   Example)
#     For ubuntu: tensorflow lite v1.13.1
#     For tizen: tensorflow lite v1.9
set(SUPPORT_TFLITE_VERSION "1.13.1" CACHE STRING "Supporting TensorFlow lite version")

add_subdirectories()
