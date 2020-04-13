# style_transfer_app

A sample app that runs `style transfer models`

It reads a neural network model from an `nnpackage` and an input image, converts the image through the network, and produces an output image.

It supports both JPG and BMP image formats. It uses **runtime API** internally.

## How to use

```
$ ./style_transfer_app --nnpackage path_to_nnpackage --input input_image --output output_image
```

## Install libjpeg

To read/write JPG images, you should install `libjpeg` on the host and target device.

```bash
$ sudo apt-get install libjpeg-dev
```

If `libjpeg` is not installed on the host or target, this app only supports the BMP file format.
