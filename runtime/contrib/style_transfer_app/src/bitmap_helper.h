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

#ifndef __STYLE_TRANSFER_APP_BITMAP_HELPER_H__
#define __STYLE_TRANSFER_APP_BITMAP_HELPER_H__

#include <vector>

namespace StyleTransferApp
{

class BitmapHelper
{
public:
  BitmapHelper(){/* DO NOTHING */};
  int read_bmp(const std::string &input_bmp_name, std::vector<float> &input, int model_width,
               int model_height);
  int write_bmp(const std::string &output_bmp_name, std::vector<float> &output, int width,
                int height, int channels);

private:
  unsigned char *createBitmapFileHeader(int height, int width, int paddingSize);
  unsigned char *createBitmapInfoHeader(int height, int width);
  std::vector<uint8_t> decode_bmp(const uint8_t *input, int row_size, int width, int height,
                                  int channels, bool top_down);

  const int fileHeaderSize = 14;
  const int infoHeaderSize = 40;
  const int bytesPerPixel = 3;
};

} // namespace StyleTransferApp

#endif // __STYLE_TRANSFER_APP_BITMAP_HELPER_H__
