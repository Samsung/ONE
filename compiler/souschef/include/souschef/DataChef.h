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

#ifndef __SOUSCHEF_DATA_CHEF_H__
#define __SOUSCHEF_DATA_CHEF_H__

#include "Arguments.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace souschef
{

using Data = std::vector<uint8_t>;

/**
 * @brief Data Generator
 */
struct DataChef
{
  virtual ~DataChef() = default;

  // TODO Allow users to query the type of elements that this DataChef generates

  /**
   * @brief Generate a sequence of 'count' elements as a byte sequence
   *
   * Let D be the return value of generate(N).
   * Then, D.size() == N * sizeof(T) where T is the element type.
   */
  virtual Data generate(int32_t count) const = 0;
};

/**
 * @brief Data Generator Factory
 */
struct DataChefFactory
{
  virtual ~DataChefFactory() = default;

  virtual std::unique_ptr<DataChef> create(const Arguments &args) const = 0;
};

} // namespace souschef

#endif // __SOUSCHEF_DATA_CHEF_H__
