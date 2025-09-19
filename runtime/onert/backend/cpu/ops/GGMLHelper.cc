/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in riting, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "GGMLHelper.h"

#include <util/Exceptions.h>

namespace onert::backend::cpu::ops
{

ggml_type getGGMLType(std::string op, ir::DataType type)
{
  switch (type)
  {
    case ir::DataType::FLOAT32:
      return GGML_TYPE_F32;
    case ir::DataType::QUANT_GGML_Q4_0:
      return GGML_TYPE_Q4_0;
    case ir::DataType::QUANT_GGML_Q8_0:
      return GGML_TYPE_Q8_0;
    case ir::DataType::INT32:
      return GGML_TYPE_I32;
    case ir::DataType::INT64:
      return GGML_TYPE_I64;
    default:
      throw UnsupportedDataTypeException{std::move(op), type};
  }
}

struct ggml_tensor getGGMLTensor(std::string op, const IPortableTensor *tensor)
{
  struct ggml_tensor res;

  res.type = getGGMLType(std::move(op), tensor->data_type());
  const auto rank = tensor->getShape().rank();
  for (int i = 0; i < GGML_MAX_DIMS; ++i)
  {
    if (i >= rank)
      res.ne[i] = 1;
    else
      res.ne[i] = tensor->getShape().dim(rank - i - 1);
  }

  res.nb[0] = ggml_type_size(res.type);
  res.nb[1] = res.nb[0] * (res.ne[0] / ggml_blck_size(res.type));
  for (int i = 2; i < GGML_MAX_DIMS; ++i)
    res.nb[i] = res.nb[i - 1] * res.ne[i - 1];

  res.op = GGML_OP_NONE;
  res.grad = nullptr;
  res.data = (void *)(tensor->buffer());

  return res;
}

} // namespace onert::backend::cpu::ops
