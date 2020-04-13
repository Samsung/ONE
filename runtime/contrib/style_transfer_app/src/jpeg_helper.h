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

#ifndef __STYLE_TRANSFER_APP_JPEG_HELPER_H__
#define __STYLE_TRANSFER_APP_JPEG_HELPER_H__

#include <vector>
#include <string>
#include <jpeglib.h>

namespace StyleTransferApp
{

class JpegHelper
{
public:
  JpegHelper(){/* DO NOTHING */};
  JpegHelper(int bytes_per_pixel, J_COLOR_SPACE color_space);

  int readJpeg(const std::string filename, std::vector<float> &raw_image, int width, int height);
  int writeJpeg(const std::string filename, std::vector<float> &raw_image, int width, int height);

private:
  int _bytes_per_pixel = 3;             /* or 1 for GRACYSCALE images */
  J_COLOR_SPACE _color_space = JCS_RGB; /* or JCS_GRAYSCALE for grayscale images */
};

} // namespace StyleTransferApp

#endif // __STYLE_TRANSFER_APP_JPEG_HELPER_H__
