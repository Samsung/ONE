/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONE_SERVICE_NPUD_CORE_IR_DATATYPE_H__
#define __ONE_SERVICE_NPUD_CORE_IR_DATATYPE_H__

#include <cstdlib>

namespace npud
{
namespace core
{
namespace ir
{

enum class DataType
{
  INT8 = 0,
  UINT8,
  QUANT_UINT8_ASYMM,
  INT16,
  UINT16,
  QUANT_INT16_SYMM,
  INT32,
  UINT32,
  FLOAT32,
  INT64,
  UINT64,
  FLOAT64,
};

} // namespace ir
} // namespace core
} // namespace npud

#endif // __ONE_SERVICE_NPUD_CORE_IR_DATATYPE_H__
