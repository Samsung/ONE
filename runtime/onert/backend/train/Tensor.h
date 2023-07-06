/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_TENSOR_H__
#define __ONERT_BACKEND_TRAIN_TENSOR_H__

#include <backend/basic/Tensor.h>

namespace onert
{
namespace backend
{
namespace train
{

// NOTE This class can be replaced with basic::Tensor if this backend supports dynamic tensors.
class Tensor : public basic::Tensor
{
public:
  Tensor() = delete;

public:
  Tensor(const ir::OperandInfo &info, const ir::Layout layout)
    : basic::Tensor{info, layout, nullptr}
  {
    // DO NOTHING
  }

public:
  bool applyShape(const ir::Shape &) override { return false; }
};

using ExternalTensor = basic::ExternalTensor;

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_TENSOR_H__
