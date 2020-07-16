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
namespace frontend
{
namespace custom
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

Kernel::Kernel(const nnfw_custom_eval evalFunction)
    : _params(), _userdata(nullptr), _userdata_size(0), _evalFunction(evalFunction)
{
  _params.inputs = _params.outputs = nullptr;
}

Kernel::~Kernel()
{
  if (_params.inputs)
    delete[] _params.inputs;
  if (_params.outputs)
    delete[] _params.outputs;
}

void Kernel::configure(CustomKernelConfigParams &&inParams)
{
  _userdata = inParams.userdata;
  _userdata_size = inParams.userdata_size;

  _params.ninputs = inParams.input_allocations.size();
  _params.inputs = new nnfw_operand[_params.ninputs];
  for (size_t i = 0; i < _params.ninputs; ++i)
  {
    _params.inputs[i] =
        APIConverter::convertOperand(inParams.input_allocations[i], inParams.input_types[i]);
  }

  _params.noutputs = inParams.output_allocations.size();
  _params.outputs = new nnfw_operand[_params.noutputs];
  for (size_t i = 0; i < _params.noutputs; ++i)
  {
    _params.outputs[i] =
        APIConverter::convertOperand(inParams.output_allocations[i], inParams.output_types[i]);
  }
}

void Kernel::run() { _evalFunction(&_params, _userdata, _userdata_size); }

} // namespace custom
} // namespace frontend
} // namespace onert
