/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <unistd.h> // NOLINT(build/include_order)

#include "bitmap_helper.h"

#define LOG(x) std::cerr

namespace StyleTransferApp
{

unsigned char *BitmapHelper::createBitmapFileHeader(int height, int width, int paddingSize)
{
  int fileSize = fileHeaderSize + infoHeaderSize + (bytesPerPixel * width + paddingSize) * height;

  static unsigned char fileHeader[] = {
    0, 0,       /// signature
    0, 0, 0, 0, /// image file size in bytes
    0, 0, 0, 0, /// reserved
    0, 0, 0, 0, /// start of pixel array
  };

  fileHeader[0] = (unsigned char)('B');
  fileHeader[1] = (unsigned char)('M');
  fileHeader[2] = (unsigned char)(fileSize);
  fileHeader[3] = (unsigned char)(fileSize >> 8);
  fileHeader[4] = (unsigned char)(fileSize >> 16);
  fileHeader[5] = (unsigned char)(fileSize >> 24);
  fileHeader[10] = (unsigned char)(fileHeaderSize + infoHeaderSize);

  return fileHeader;
}

unsigned char *BitmapHelper::createBitmapInfoHeader(int height, int width)
{
  static unsigned char infoHeader[] = {
    0, 0, 0, 0, /// header size
    0, 0, 0, 0, /// image width
    0, 0, 0, 0, /// image height
    0, 0,       /// number of color planes
    0, 0,       /// bits per pixel
    0, 0, 0, 0, /// compression
    0, 0, 0, 0, /// image size
    0, 0, 0, 0, /// horizontal resolution
    0, 0, 0, 0, /// vertical resolution
    0, 0, 0, 0, /// colors in color table
    0, 0, 0, 0, /// important color count
  };

  // Minus height means top to bottom write
  height = -height;

  infoHeader[0] = (unsigned char)(infoHeaderSize);
  infoHeader[4] = (unsigned char)(width);
  infoHeader[5] = (unsigned char)(width >> 8);
  infoHeader[6] = (unsigned char)(width >> 16);
  infoHeader[7] = (unsigned char)(width >> 24);
  infoHeader[8] = (unsigned char)(height);
  infoHeader[9] = (unsigned char)(height >> 8);
  infoHeader[10] = (unsigned char)(height >> 16);
  infoHeader[11] = (unsigned char)(height >> 24);
  infoHeader[12] = (unsigned char)(1);
  infoHeader[14] = (unsigned char)(bytesPerPixel * 8);

  return infoHeader;
}

std::vector<uint8_t> BitmapHelper::decode_bmp(const uint8_t *input, int row_size, int width,
                                              int height, int channels, bool top_down)
{
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++)
  {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++)
    {
      if (!top_down)
      {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      }
      else
      {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels)
      {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          // BGR -> RGB
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          LOG(FATAL) << "Unexpected number of channels: " << channels;
          break;
      }
    }
  }
  return output;
}

int BitmapHelper::read_bmp(const std::string &input_bmp_name, std::vector<float> &input,
                           int model_width, int model_height)
{
  int begin, end;
  int width, height, channels;

  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file)
  {
    LOG(FATAL) << "Error opening " << input_bmp_name << "\n";
    exit(-1);
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

  std::vector<uint8_t> img_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(img_bytes.data()), len);
  const int32_t header_size = *(reinterpret_cast<const int32_t *>(img_bytes.data() + 10));
  width = *(reinterpret_cast<const int32_t *>(img_bytes.data() + 18));
  height = *(reinterpret_cast<const int32_t *>(img_bytes.data() + 22));
  const int32_t bpp = *(reinterpret_cast<const int32_t *>(img_bytes.data() + 28));
  channels = bpp / 8;

  // TODO: Implement resize function
  assert(model_width == width);
  assert(model_height == height);

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * channels * width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (height < 0);

  // Decode image, allocating tensor once the image size is known
  const uint8_t *bmp_pixels = &img_bytes[header_size];
  std::vector<uint8_t> bmp =
    decode_bmp(bmp_pixels, row_size, width, abs(height), channels, top_down);
  for (uint32_t j = 0; j < bmp.size(); j++)
  {
    input.push_back(static_cast<float>(bmp[j]));
  }
  return 0;
}

int BitmapHelper::write_bmp(const std::string &output_bmp_name, std::vector<float> &output,
                            int width, int height, int channels)
{
  std::ofstream file(output_bmp_name, std::ios::out | std::ios::binary);
  if (!file)
  {
    LOG(FATAL) << "Error opening " << output_bmp_name << "\n";
    exit(-1);
  }

  unsigned char padding[3] = {0, 0, 0};
  int paddingSize = (4 - (width * channels) % 4) % 4;

  const unsigned char *fileHeader = createBitmapFileHeader(height, width, paddingSize);
  const unsigned char *infoHeader = createBitmapInfoHeader(height, width);

  file.write((char *)fileHeader, fileHeaderSize);
  file.write((char *)infoHeader, infoHeaderSize);

  // RGB to BGR
  for (int i = 0; i < output.size(); i += 3)
  {
    file << static_cast<unsigned char>(output[i + 2]);
    file << static_cast<unsigned char>(output[i + 1]);
    file << static_cast<unsigned char>(output[i]);
    for (int j = 0; j < paddingSize; j++)
    {
      file << padding;
    }
  }
  file.close();
  return 0;
}

} // namespace StyleTransferApp
