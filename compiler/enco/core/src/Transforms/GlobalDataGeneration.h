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

#ifndef __ENCO_TRANSFORM_GLOBAL_DATA_GENERATION_H__
#define __ENCO_TRANSFORM_GLOBAL_DATA_GENERATION_H__

#include "Code.h"

#include <ostream>

namespace enco
{

using GlobalOffset = uint32_t;

struct GlobalData
{
  static GlobalOffset data_offset(const ann::Operand *);
  /**
   * @brief Return the weight offset of a given bag
   *
   * @note The behavior of "data_offset" is undefined if a bag has no weight.
   */
  static GlobalOffset data_offset(const coco::Bag *);

  static GlobalOffset name_offset(const coco::Input *);
  static GlobalOffset dims_offset(const coco::Input *);
  static GlobalOffset name_offset(const coco::Output *);
  static GlobalOffset dims_offset(const coco::Output *);
};

/**
 * @brief Generate 'Global' weight array.
 *
 * NOTE Succeeding passes can access offsets via "GlobalData"
 */
void generate_global_data(std::ostream &, enco::Code *);

} // namespace enco

#endif // __ENCO_TRANSFORM_GLOBAL_DATA_GENERATION_H__
