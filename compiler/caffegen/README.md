# caffegen

`caffegen` is a tool for generating caffe model and decoding binary file of caffe model

## How caffegen works

Some of commands in `caffegen` use standard input for reading data and standard output for exporting result.
In this case, we strongly recommend you to use pipe, not copy & paste the content of file itself.

Otherwise, `caffegen` use arguments to pass some directories.

## Supported command

Basically, caffegen command is used as `caffegen [COMMAND]` and there are four `COMMAND` types.
 - init : initialize parameters using prototxt.
 - encode : make a binary file (caffemodel) using initialized data
 - decode : decode a binary file (caffemodel) and reproduce the initialized data
 - merge : copy the trained weights from a caffemodel into a prototxt file

## How to use each command

1. Init (Using stdin and stdout)
 - `./build/compiler/caffegen/caffegen init`
     - Type the prototxt by yourself
     - Then you can get the result on the shell.
 - `cat ./res/BVLCCaffeTests/Convolution_000/test.prototxt | ./build/compiler/caffegen/caffegen init`
     - Prototxt will be automatically passed
     - Then you can get the result on the shell.

2. Encode (Using stdin and stdout)
 - `./build/compiler/caffegen/caffegen encode`
     - Type the initialized data by yourself
     - Then you can get the result on the shell.
 - `cat ./res/BVLCCaffeTests/Convolution_000/test.prototxt | ./build/compiler/caffegen/caffegen init | ./build/compiler/caffegen/caffegen encode > Convolution_000.caffemodel`
     - The initialized data will be automatically passed
     - The encoded result will be automatically saved in caffemodel file

3. Decode (Using stdin and stdout)
 - `cat Convolution_000.caffemodel | ./build/compiler/caffegen/caffegen decode`
     - Caffemodel file will be automatically passed
     - Then you can get the result on the shell

4. Merge (Using arguments)
 - `./build/compiler/caffegen/caffegen merge ./res/BVLCCaffeTests/Convolution_000/test.prototxt Convolution_000.caffemodel`
 - `./build/compiler/caffegen/caffegen merge ./res/BVLCCaffeTests/Convolution_000/test.prototxt Convolution_000.caffemodel > Convolution_000.merged`
