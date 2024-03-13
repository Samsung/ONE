/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_DATATYPE_H__
#define __ONERT_IR_DATATYPE_H__

#include <cstdlib>

namespace onert
{
namespace ir
{

enum class DataType
{
  FLOAT32 = 0,
  INT32 = 1,
  UINT32 = 2,
  QUANT_UINT8_ASYMM = 3,
  BOOL8 = 4,
  UINT8 = 5,
  QUANT_INT8_SYMM = 6,
  FLOAT16 = 7,
  INT64 = 8,
  QUANT_INT8_ASYMM = 9,
  QUANT_INT16_ASYMM = 10,
  QUANT_INT8_SYMM_PER_CHANNEL = 11,
  QUANT_INT16_SYMM = 12,
  QUANT_INT4_SYMM = 13
};

size_t sizeOfDataType(DataType data_type);

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_DATATYPE_H__
