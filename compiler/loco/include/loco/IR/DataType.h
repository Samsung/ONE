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

#ifndef __LOCO_IR_DATA_TYPE_H__
#define __LOCO_IR_DATA_TYPE_H__

namespace loco
{

/**
 * @brief "scalar" value type
 */
enum class DataType
{
  Unknown, // Unknown type (serves as a default value)

  U4,  // 4-bit unsigned integer
  U8,  // 8-bit unsigned integer
  U16, // 16-bit unsigned integer
  U32, // 32-bit unsigned integer
  U64, // 64-bit unsigned integer

  S4,  // 4-bit signed integer
  S8,  // 8-bit signed integer
  S16, // 16-bit signed integer
  S32, // 32-bit signed integer
  S64, // 64-bit signed integer

  FLOAT16, // IEEE 16-bit floating-point
  FLOAT32, // IEEE 32-bit floating-point
  FLOAT64, // IEEE 64-bit floating-point

  // WARNING the size of Bool may vary for NN frameworks
  // TODO we need to find a way to resolve this issue
  BOOL, // Boolean

  // WARNING STRING is NOT fully supported yet
  STRING, // String
};

} // namespace loco

#endif // __LOCO_IR_DATA_TYPE_H__
