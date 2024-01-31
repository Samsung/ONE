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

#ifndef __LUCI_QUANTIZATION_PARAMETERS_H__
#define __LUCI_QUANTIZATION_PARAMETERS_H__

#include <loco.h>

#include <string>

namespace luci
{

enum QuantizationGranularity
{
  LayerWise = 0,
  ChannelWise = 1,
};

enum struct QuantizationAlgorithm
{
  Common = 0,
  MinimumMSE = 1,
};

struct LayerInfo
{
  std::string name;
  loco::DataType dtype;
  QuantizationGranularity granularity;
};

} // namespace luci

#endif // __LUCI_QUANTIZATION_PARAMETERS_H__
