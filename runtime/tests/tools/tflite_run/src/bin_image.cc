/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <iostream>
#include <fstream>

#include "bin_image.h"

BinImage::BinImage(unsigned int width, unsigned int height, unsigned int channels)
  : _width(width), _height(height), _channels(channels)
{
}

BinImage::~BinImage() {}

void BinImage::loadImage(const std::string &filename)
{
  std::ifstream fin(filename);

  if (!fin)
  {
    std::cerr << "image filename is not specified. "
              << "Input image will not be set." << std::endl;
    return;
  }

  _image.reserve(_width * _height * _channels);

  // Assuption: binary image is stored in the order of [H,W,C]
  for (unsigned int i = 0; i < _width * _height * _channels; ++i)
    _image.push_back(fin.get());
}

void BinImage::AssignTensor(TfLiteTensor *t)
{
  float *p = t->data.f;
  const int IMAGE_MEAN = 128;
  const float IMAGE_STD = 128.0f;

  // to prevent runtime exception
  if (_image.size() < _width * _height * _channels)
  {
    std::cerr << "Input image size is smaller than the size required by the model."
              << " Input will not be set." << std::endl;
    return;
  }

  for (int x = 0; x < _width; ++x)
  {
    for (int y = 0; y < _height; ++y)
    {
      for (int c = 0; c < _channels; ++c)
      {
        *p++ = (_image[y * _width * _channels + x * _channels + c] - IMAGE_MEAN) / IMAGE_STD;
      }
    }
  }
}
