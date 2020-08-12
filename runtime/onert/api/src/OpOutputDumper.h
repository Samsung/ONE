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

#ifndef __ONERT_OP_OUTPUT_DUMPER_H__
#define __ONERT_OP_OUTPUT_DUMPER_H__

#include "exec/IFunctionObserver.h"
#include "backend/ITensor.h"

#include <nnfw_internal.h>

namespace
{

void convert(const onert::backend::ITensor *src, nnfw_output_tensor *dst)
{
  // dtype
  auto one_dtype = src->data_type();
  switch (one_dtype)
  {
    case onert::ir::DataType::BOOL8:
      dst->type.dtype = NNFW_TYPE_TENSOR_BOOL;
      break;
    case onert::ir::DataType::FLOAT32:
      dst->type.dtype = NNFW_TYPE_TENSOR_FLOAT32;
      break;
    case onert::ir::DataType::INT32:
      dst->type.dtype = NNFW_TYPE_TENSOR_INT32;
      break;
    case onert::ir::DataType::INT64:
      dst->type.dtype = NNFW_TYPE_TENSOR_INT64;
      break;
    default:
      throw std::runtime_error("Not supported datatype");
  }

  // shape
  dst->type.rank = src->num_dimensions();
  for (int d = 0; d < dst->type.rank; d++)
    dst->type.dims[d] = src->dimension(d);

  // buffer
  dst->allocation = src->buffer();
}

} // namespace

namespace onert
{
namespace frontend
{

class OpOutputDumper : public exec::IFunctionObserver
{
public:
  OpOutputDumper(nnfw_dump_op_output callback) : _callback(callback) {} //,_dumping_info(nullptr) {
                                                                        //}

  void handleEnd(const ir::Operation * /*op*/, const exec::IFunction &func) override
  {
    // std::cout << "Dumper: " << func.name() << ", " <<  func.id.op_ind.value() << ":" <<
    // func.id.op_seq_ind.value() << ":" << func.id.subg_ind.value() << std::endl;
    assert(_callback);
    assert(func.id.op_ind.valid());
    assert(func.id.op_seq_ind.valid());
    assert(func.id.subg_ind.valid());

    auto output_size = func.getOutputSize();
    for (uint32_t i = 0; i < output_size; ++i)
    {
      auto *tensor = func.getOutput(i);

      nnfw_output_tensor nnfw_tensor;
      convert(tensor, &nnfw_tensor);

      _callback(&nnfw_tensor, func.id.subg_ind.value(), func.id.op_seq_ind.value(),
                func.id.op_ind.value(), i);
    }
  }

private:
  nnfw_dump_op_output _callback;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_OP_OUTPUT_DUMPER_H__
