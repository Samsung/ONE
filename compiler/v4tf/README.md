# v4tf

## What is this?

*v4tf* is is a wrapper interface to use TensorFlow by its C API.
The name was originated from the movie, *V for Vendetta*, where the main character *V* hides his face by wearing a mask.

## Why do we need this?

In *nncc*, some tests use TensorFlow, which uses Protocol Buffers.
For example, TensorFlow 1.13.1 uses Protocol Buffers 3.6.1.2.

Some of *nncc* modules use different version Protocol Buffers for internal purpose.
If such modules also try to use TensorFlow API, errors were thrown due to resolution of wrong symbols of different versions of Protocol Buffers.

To prevent these errors, *v4tf* loads TensorFlow dynamically with all of its symbols resolved.
