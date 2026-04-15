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

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "MirInterpreter.h"
#include "mir/ops/ConstantOp.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"
#include "mir/TensorVariant.h"
#include "mir/Index.h"

namespace mir_onnx
{

const int64_t firstUnknownOpset = 13;

template <typename T> static mir::Shape constantToShapeT(const mir::TensorVariant &t)
{
  const mir::Shape &t_shape = t.getShape();
  mir::Tensor<T> input(t);
  if (t_shape.rank() != 1)
    throw std::runtime_error("only 1-d tensors supported as a shape input");

  mir::Shape target_shape;
  std::int32_t rank = t_shape.dim(0);
  target_shape.resize(rank);
  for (int i = 0; i < rank; ++i)
    target_shape.dim(i) = static_cast<std::int32_t>(input.at(mir::Index{i}));
  return target_shape;
}

mir::Shape constantToShape(const mir::ops::ConstantOp *op)
{
  const auto &t = op->getValue();
  mir::DataType d_type = t.getElementType();

  if (t.getType().isQuantized())
    throw std::runtime_error("unsupported data type of shape operator");

  switch (d_type)
  {
    case mir::DataType::FLOAT32:
      return constantToShapeT<float>(t);
      break;
    case mir::DataType::FLOAT64:
      return constantToShapeT<double>(t);
      break;
    case mir::DataType::INT32:
      return constantToShapeT<int32_t>(t);
      break;
    case mir::DataType::INT64:
      return constantToShapeT<int64_t>(t);
      break;
    case mir::DataType::UINT8:
      return constantToShapeT<uint8_t>(t);
      break;
    default:
      throw std::runtime_error{"Unknown datatype in constant"};
      break;
  }
}

mir::DataType onnxDataTypeToMirDataType(onnx::TensorProto::DataType type)
{
  switch (type)
  {
    case onnx::TensorProto_DataType_UINT8:
      return mir::DataType::UINT8;
      break;
    case onnx::TensorProto_DataType_INT32:
      return mir::DataType::INT32;
      break;
    case onnx::TensorProto_DataType_INT64:
      return mir::DataType::INT64;
      break;
    case onnx::TensorProto_DataType_DOUBLE:
      return mir::DataType::FLOAT64;
      break;
    case onnx::TensorProto_DataType_FLOAT:
      return mir::DataType::FLOAT32;
      break;
    case onnx::TensorProto_DataType_UNDEFINED:
      throw std::runtime_error{"Undefined input data type not supported"};
      break;
    default:
      throw std::runtime_error{"Unsupported tensor element data type"};
  }
}

mir::TensorVariant createTensor(const onnx::TensorProto *tensor)
{
  mir::DataType type;
  const void *src_data;
  mir::Shape shape(tensor->dims_size());
  for (int i = 0; i < tensor->dims_size(); ++i)
  {
    shape.dim(i) = tensor->dims(i);
  }

  if (tensor->float_data_size() != 0)
  {
    assert(tensor->data_type() == onnx::TensorProto::FLOAT);
    type = mir::DataType::FLOAT32;
    src_data = tensor->float_data().data();
  }
  else if (tensor->double_data_size() != 0)
  {
    assert(tensor->data_type() == onnx::TensorProto::DOUBLE);
    type = mir::DataType::FLOAT64;
    src_data = tensor->double_data().data();
  }
  else if (tensor->int32_data_size() != 0)
  {
    assert(tensor->data_type() == onnx::TensorProto::INT32);
    type = mir::DataType::INT32;
    src_data = tensor->int32_data().data();
  }
  else if (tensor->int64_data_size() != 0)
  {
    assert(tensor->data_type() == onnx::TensorProto::INT64);
    type = mir::DataType::INT64;
    src_data = tensor->int64_data().data();
  }
  else if (tensor->has_raw_data())
  {
    type = onnxDataTypeToMirDataType((onnx::TensorProto_DataType)tensor->data_type());
    src_data = tensor->raw_data().data();
  }
  else
  {
    throw std::runtime_error("Invalid data in Proto file, investigate");
  }

  return mir::TensorVariant({type, shape}, src_data);
}

mir::Operation *foldConstants(mir::Graph *graph, mir::Operation *op)
{
  if (op->getType() == mir::Operation::Type::constant ||
      op->getType() == mir::Operation::Type::input || op->getType() == mir::Operation::Type::output)
  {
    // don't fold input, output and constant nodes
    return op;
  }

  if (op->getNumOutputs() != 1)
  {
    // this operation either have more than 1 output or none at all
    return op;
  }

  bool is_foldable =
    std::all_of(op->getInputs().begin(), op->getInputs().end(), [](mir::Operation::Output *out) {
      return out->getNode()->getType() == mir::Operation::Type::constant;
    });

  if (!is_foldable)
    return op;

  mir_interpreter::MIRInterpreter interpreter;
  for (mir::Operation::Output *out : op->getInputs())
  {
    auto *constant = static_cast<mir::ops::ConstantOp *>(out->getNode());
    interpreter.setTensor(out, constant->getValue());
  }
  op->accept(&interpreter);
  const mir::TensorVariant &output = interpreter.getTensor(op->getOutput(0));

  return graph->create<mir::ops::ConstantOp>(output);
}

} // namespace mir_onnx
