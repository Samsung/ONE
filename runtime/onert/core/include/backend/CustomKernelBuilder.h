/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CUSTOM_KERNEL_BUILDER_H__
#define __ONERT_BACKEND_CUSTOM_KERNEL_BUILDER_H__

#include "backend/IPortableTensor.h"
#include "ir/Shape.h"
#include "ir/DataType.h"

#include <vector>
#include <memory>

namespace onert
{
namespace exec
{

class IFunction;

} // namespace exec
} // namespace onert

namespace onert
{
namespace backend
{
namespace custom
{

struct TypeInfo
{
  ir::Shape shape;
  ir::DataType dtype;
};

struct CustomKernelConfigParams
{
  std::vector<backend::IPortableTensor *> input_tensors;
  std::vector<TypeInfo> input_types;

  std::vector<backend::IPortableTensor *> output_tensors;
  std::vector<TypeInfo> output_types;

  char *userdata;
  size_t userdata_size;
};

class IKernelBuilder
{
public:
  virtual ~IKernelBuilder() = default;
  virtual std::unique_ptr<exec::IFunction> buildKernel(const std::string &id,
                                                       CustomKernelConfigParams &&params) const = 0;
};

} // namespace custom

} // namespace backend

} // namespace onert

#endif // __ONERT_BACKEND_CUSTOM_KERNEL_BUILDER_H__
