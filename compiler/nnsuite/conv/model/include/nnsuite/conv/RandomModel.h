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

#ifndef __NNSUITE_CONV2D_RANDOM_MODEL_H__
#define __NNSUITE_CONV2D_RANDOM_MODEL_H__

#include "nnsuite/conv/Model.h"

#include <nncc/core/ADT/kernel/Buffer.h>

#include <string>

namespace nnsuite
{
namespace conv
{

class RandomModel final : public Model
{
public:
  explicit RandomModel(int32_t seed);

public:
  const nncc::core::ADT::feature::Shape &ifm_shape(void) const override { return _ifm_shape; }
  const std::string &ifm_name(void) const override { return _ifm_name; }

public:
  const nncc::core::ADT::feature::Shape &ofm_shape(void) const override { return _ofm_shape; }
  const std::string &ofm_name(void) const override { return _ofm_name; }

public:
  const nncc::core::ADT::kernel::Shape &ker_shape(void) const override
  {
    return _ker_buffer.shape();
  }

  const nncc::core::ADT::kernel::Reader<float> &ker_data(void) const override
  {
    return _ker_buffer;
  }

private:
  const nncc::core::ADT::feature::Shape _ifm_shape;
  const std::string _ifm_name;

private:
  const nncc::core::ADT::feature::Shape _ofm_shape;
  const std::string _ofm_name;

private:
  nncc::core::ADT::kernel::Buffer<float> _ker_buffer;
};

} // namespace conv
} // namespace nnsuite

#endif // __NNSUTIE_CONV2D_RANDOM_MODEL_H__
