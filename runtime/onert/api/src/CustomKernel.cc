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

#include "CustomKernel.h"

namespace onert
{
namespace api
{

using namespace backend::custom;

class APIConverter
{
public:
  static nnfw_operand convertOperand(void *alloc, const TypeInfo &type)
  {
    nnfw_operand api_operand;
    api_operand.allocation = alloc;
    api_operand.type = convertType(type);
    return api_operand;
  }

  static nnfw_tensorinfo convertType(const TypeInfo &type)
  {
    nnfw_tensorinfo api_type;
    api_type.rank = type.shape.rank();
    assert(type.shape.rank() <= 6);
    std::copy(type.shape.dims().begin(), type.shape.dims().end(), std::begin(api_type.dims));

    switch (type.dtype)
    {
      case ir::DataType::FLOAT32:
        api_type.dtype = NNFW_TYPE_TENSOR_FLOAT32;
        break;
      case ir::DataType::INT32:
        api_type.dtype = NNFW_TYPE_TENSOR_INT32;
        break;
      case ir::DataType::QUANT_UINT8_ASYMM:
        api_type.dtype = NNFW_TYPE_TENSOR_QUANT8_ASYMM;
        break;
      case ir::DataType::BOOL8:
        api_type.dtype = NNFW_TYPE_TENSOR_BOOL;
        break;
      default:
        throw std::runtime_error("Unsupported tensor datatype");
    }
    return api_type;
  }
};

CustomKernel::CustomKernel(const nnfw_custom_eval evalFunction)
  : _in_params(), _userdata(nullptr), _userdata_size(0), _evalFunction(evalFunction)
{
}

void CustomKernel::configure(CustomKernelConfigParams &&inParams)
{
  _userdata = inParams.userdata;
  _userdata_size = inParams.userdata_size;

  _in_params = std::move(inParams);
}

void CustomKernel::run()
{
  nnfw_custom_kernel_params params;

  // set input tensor buffer and types
  params.ninputs = _in_params.input_tensors.size();
  params.inputs = new nnfw_operand[params.ninputs];

  for (size_t i = 0; i < params.ninputs; ++i)
  {
    auto *buf = _in_params.input_tensors[i]->buffer();
    assert(buf);
    params.inputs[i] = APIConverter::convertOperand(buf, _in_params.input_types[i]);
  }

  // set output tensor buffer and types
  params.noutputs = _in_params.output_tensors.size();
  params.outputs = new nnfw_operand[params.noutputs];

  for (size_t i = 0; i < params.noutputs; ++i)
  {
    auto *buf = _in_params.output_tensors[i]->buffer();
    assert(buf);
    params.outputs[i] = APIConverter::convertOperand(buf, _in_params.output_types[i]);
  }

  _evalFunction(&params, _userdata, _userdata_size);

  delete[] params.inputs;
  delete[] params.outputs;
}

} // namespace api
} // namespace onert
