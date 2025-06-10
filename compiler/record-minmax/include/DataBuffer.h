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

#ifndef __RECORD_MINMAX_DATA_BUFFER_H__
#define __RECORD_MINMAX_DATA_BUFFER_H__

#include <loco/IR/DataType.h>
#include <loco/IR/Dimension.h>

#include <vector>

namespace record_minmax
{

struct DataBuffer
{
  loco::DataType dtype = loco::DataType::Unknown;
  std::vector<loco::Dimension> shape;
  std::vector<char> data;
};

} // namespace record_minmax

#endif // __RECORD_MINMAX_DATA_BUFFER_H__
