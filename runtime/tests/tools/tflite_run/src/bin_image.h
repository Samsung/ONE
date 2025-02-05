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

#ifndef __TFLITE_RUN_LIBJPEG_H__
#define __TFLITE_RUN_LIBJPEG_H__

#include <string>
#include <vector>

#include "tensorflow/lite/context.h"

class BinImage
{
public:
  BinImage(unsigned int width, unsigned int height, unsigned int channel);
  ~BinImage();

  void loadImage(const std::string &filename);

  void AssignTensor(TfLiteTensor *t);

private:
  unsigned int _width;
  unsigned int _height;
  unsigned int _channels;

  std::vector<unsigned char> _image;
};

#endif // __TFLITE_RUN_LIBJPEG_H__
