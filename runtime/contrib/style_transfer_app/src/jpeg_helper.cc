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

#include "jpeg_helper.h"

#include <cassert>
#include <stdio.h>
#include <jpeglib.h>
#include <stdlib.h>
#include <vector>

namespace StyleTransferApp
{

JpegHelper::JpegHelper(int bytes_per_pixel, J_COLOR_SPACE color_space)
  : _bytes_per_pixel(bytes_per_pixel), _color_space(color_space)
{
  // DO NOTHING
}

int JpegHelper::readJpeg(const std::string filename, std::vector<float> &raw_image, int width,
                         int height)
{
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  FILE *infile = fopen(filename.c_str(), "rb");
  unsigned long location = 0;
  int i = 0;

  if (!infile)
  {
    printf("Error opening jpeg file %s\n!", filename.c_str());
    return -1;
  }

  cinfo.err = jpeg_std_error(&jerr);

  jpeg_create_decompress(&cinfo);

  jpeg_stdio_src(&cinfo, infile);

  jpeg_read_header(&cinfo, TRUE);

  jpeg_start_decompress(&cinfo);

  // TODO: Implement resize function
  assert(cinfo.output_width == width);
  assert(cinfo.output_height == height);

  raw_image.resize(cinfo.output_width * cinfo.output_height * cinfo.num_components);

  unsigned char *ptr = new unsigned char[cinfo.output_width * cinfo.num_components];

  while (cinfo.output_scanline < cinfo.image_height)
  {
    jpeg_read_scanlines(&cinfo, &ptr, 1);
    for (i = 0; i < cinfo.image_width * cinfo.num_components; i++)
    {
      raw_image[location++] = static_cast<float>(ptr[i]);
    }
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  delete (ptr);
  fclose(infile);

  return 1;
}

int JpegHelper::writeJpeg(const std::string filename, std::vector<float> &raw_image, int width,
                          int height)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  unsigned long location = 0;

  FILE *outfile = fopen(filename.c_str(), "wb");

  if (!outfile)
  {
    printf("Error opening output jpeg file %s\n!", filename.c_str());
    return -1;
  }
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = _bytes_per_pixel;
  cinfo.in_color_space = _color_space;

  jpeg_set_defaults(&cinfo);

  jpeg_start_compress(&cinfo, TRUE);

  unsigned char *ptr = new unsigned char[cinfo.image_width * cinfo.input_components];

  while (cinfo.next_scanline < cinfo.image_height)
  {
    for (int i = 0; i < cinfo.image_width * cinfo.input_components; i++)
    {
      ptr[i] = static_cast<unsigned char>(raw_image[location++]);
    }
    jpeg_write_scanlines(&cinfo, &ptr, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  delete (ptr);

  jpeg_destroy_compress(&cinfo);

  return 1;
}

} // namespace StyleTransferApp
