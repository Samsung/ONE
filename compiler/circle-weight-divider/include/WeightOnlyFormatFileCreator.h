/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_WEIGHT_DIVIDER_WEIGHT_ONLY_FORMAT_FILE_CREATOR_H__
#define __CIRCLE_WEIGHT_DIVIDER_WEIGHT_ONLY_FORMAT_FILE_CREATOR_H__

#include <loco.h>
#include <luci/CircleExporter.h>
#include <luci/IR/Module.h>
#include <oops/InternalExn.h>

#include <luci/Importer.h>

#include <string>
#include <fstream>
#include <iostream>

namespace luci
{

class WeightOnlyFormatFileCreator
{
  size_t calculateFileSize();

public:
  explicit WeightOnlyFormatFileCreator(const std::vector<char> &model_data)
  {
    _model = circle::GetModel(model_data.data());
  }

  std::tuple<std::unique_ptr<char[]>, size_t> create();

private:
  const circle::Model *_model;
};

} // namespace luci

#endif // __CIRCLE_WEIGHT_DIVIDER_WEIGHT_ONLY_FORMAT_FILE_CREATOR_H__
