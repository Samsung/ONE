/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __TFL_MODEL_H__
#define __TFL_MODEL_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <mio/tflite/schema_generated.h>

namespace tflite2circle
{

class TFLModel
{
private:
  using DataBuffer = std::vector<char>;

public:
  TFLModel(void) = delete;
  TFLModel(const std::string &path);

public:
  const tflite::Model *get_model(void);
  // NOTE TFLModel lifetime should be longer than users
  const std::vector<char> &raw_data(void) const { return _data; }

public:
  bool verify_data(void);

private:
  std::ifstream _infile;
  DataBuffer _data;

  friend class CircleModel;
};

} // namespace tflite2circle

#endif // __TFL_MODEL_H__
