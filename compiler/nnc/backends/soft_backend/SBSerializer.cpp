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

#include "SBSerializer.h"
#include "mir/ShapeRange.h"
#include "mir/TensorUtil.h"

#include "mir/OpDefs.h"

#include <algorithm>

#define UNUSED(x) ((void)(x))

namespace nnc
{

static_assert(std::numeric_limits<float>::is_iec559, "Unsupported float type");

using namespace std;

using mir::Shape;
using mir::Index;
using mir::ShapeRange;
using mir::TensorVariant;

namespace ops = mir::ops;

namespace
{
// Currently there are no operations with more then 4 dimensions in kernels/weights etc supported
const auto MAX_DIMS = 4;
const auto MAX_DIM_SIZE = numeric_limits<int32_t>::max();
// Assuming there are no large enums
const auto MAX_ENUM_VAL = numeric_limits<char>::max();
} // unnamed namespace

void Serializer::packData(const void *data, size_t size)
{
  auto p = static_cast<const char *>(data);
  size_t old_size = _buffer.size();
  _buffer.resize(old_size + size);
  copy(p, p + size, _buffer.data() + old_size);
}

template <typename T> void Serializer::serializeT(const T &obj) { packData(&obj, sizeof(T)); }

/**
 * @brief Convert enum to it's underlying type
 * @tparam E Enum type
 * @param enum_value Value of enum
 * @return Integer value that correspond to enumVal
 */
template <typename E> typename underlying_type<E>::type etoi(E enum_value)
{
  return static_cast<typename underlying_type<E>::type>(enum_value);
}

void Serializer::serializeShape(const Shape &s)
{
  int32_t rank = s.rank();
  assert(rank <= MAX_DIMS);
  serializeT<int32_t>(s.rank());
  for (int32_t i = 0; i < rank; ++i)
  {
    int32_t dim = s.dim(i);
    serializeT<int32_t>(dim);
  }
}

void Serializer::serializeTensor(const TensorVariant &t)
{
  // serialize type
  assert(etoi(t.getDataType()) < MAX_ENUM_VAL);
  serializeT<int32_t>(etoi(t.getDataType()));
  // seriazlie data size
  size_t element_size = t.getElementSize();
  assert(element_size <= MAX_DIMS);
  serializeT<int32_t>(element_size);
  // serialize shape
  const Shape &shape = t.getShape();
  serializeShape(shape);
  // serialize actual data
  size_t data_size = element_size * shape.numElements();

  size_t old_serialized_data_size = _buffer.size();
  _buffer.reserve(old_serialized_data_size + data_size);
  for (const Index &idx : ShapeRange(shape))
  {
    packData(t.at(idx), element_size);
  }
}

void Serializer::serializeStrides(const vector<int32_t> &strides)
{
  serializeT<int>(strides.size());
  for (const int32_t x : strides)
  {
    serializeT<int32_t>(x);
  }
}

template <typename Op> void Serializer::serializePads(const Op &op, int32_t number_of_pads)
{
  assert(number_of_pads <= MAX_DIMS);
  serializeT<int32_t>(number_of_pads);
  for (int i = 0; i < static_cast<int>(number_of_pads); ++i)
  {
    auto pad = op.getPaddingBefore().at(i);
    assert(pad <= MAX_DIM_SIZE);
    assert(pad >= 0);
    UNUSED(pad);
    serializeT<int32_t>(op.getPaddingBefore().at(i));
  }
}

void Serializer::visit(ops::ConcatOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // axis number should fit into one byte
  assert(op.getAxis() <= MAX_DIMS);
  serializeT<int32_t>(op.getAxis());
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(ops::Conv2DOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // serialize strides
  serializeStrides(op.getStrides());
  // serialize pads
  int32_t padsRank = 2; // op.getInputShape(0).rank();
  serializePads(op, padsRank);
  // serialize output shape
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(ops::DepthwiseConv2DOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // serialize strides
  serializeStrides(op.getStrides());
  // serialize pads
  int32_t padsRank = 2; // kernel.getShape().rank();
  serializePads(op, padsRank);
  // serialize output shape
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(ops::SoftmaxOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // axis number should fit into one byte
  assert(op.getAxis() <= MAX_DIMS);
  serializeT<int32_t>(op.getAxis());
}

void Serializer::visit(ops::AvgPool2DOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // serialize window shape
  serializeShape(Shape(op.getWindowSize()));
  // serialize strindes
  serializeStrides(op.getStrides());
  // serialize pads
  int32_t number_of_pads = 2; // windowShape.rank();
  serializePads(op, number_of_pads);
  serializeT<int32_t>(op.getIncludePad());
  // serialize output shape
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(ops::MaxPool2DOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // serialize window shape
  serializeShape(Shape(op.getWindowSize()));
  // serialize strindes
  serializeStrides(op.getStrides());
  // serialize pads
  int32_t number_of_pads = 2; // windowShape.rank();
  serializePads(op, number_of_pads);
  // serialize output shape
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(ops::FullyConnectedOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(ops::CappedReluOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  serializeT<float>(op.getCap());
}

void Serializer::visit(ops::InputOp & /*op*/)
{
  // no parameters to dump
}

void Serializer::visit(ops::ConstantOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  serializeTensor(op.getValue());
}

void Serializer::visit(ops::ReluOp & /*op*/)
{
  _curOp->paramStartOffset = _buffer.size();
  // no parameters to dump
}

void Serializer::visit(ops::ReshapeOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::SliceOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  serializeShape(op.getStarts());
  serializeShape(op.getSizes());
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::TanhOp & /*op*/)
{
  _curOp->paramStartOffset = _buffer.size();
  // no parameters to dump
}

void Serializer::visit(mir::ops::EluOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  serializeT<float>(op.getAlpha());
}

void Serializer::visit(mir::ops::DeConv2DOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // serialize strides
  serializeStrides(op.getStrides());
  // serialize pads
  int32_t number_of_pads = 2; // op.getInputShape(0).rank();
  serializePads(op, number_of_pads);
  // serialize output shape
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(ops::SqueezeOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::PadOp &op)
{
  _curOp->paramStartOffset = _buffer.size();

  // serialize paddings
  const int num_dims = op.getInputShape(0).rank();

  // serialize output shape
  serializeShape(op.getOutputShape(0));

  // serialize num dimensions
  serializeT<int32_t>(num_dims);

  const auto &padding_before = op.getPaddingBefore();
  const auto &padding_after = op.getPaddingAfter();
  for (int i = 0; i < num_dims; i++)
  {
    serializeT<int32_t>(padding_before[num_dims - 1 - i]);
    serializeT<int32_t>(padding_after[num_dims - 1 - i]);
  }

  // FIXME Make use of padding value.
  assert(op.getPaddingValue() == 0.0f);
}

void Serializer::visit(mir::ops::SqrtOp & /*op*/)
{
  _curOp->paramStartOffset = _buffer.size();
  // no parameters to dump
}

void Serializer::visit(mir::ops::ResizeOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // Result shape is the same as Output shape
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::ReduceMeanOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  serializeShape(Shape(op.getReductionDims())); // reuse shape serialization
  serializeT<int32_t>(op.getKeepDims());
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::TransposeOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // serializer parameters
  auto &axis_order = op.getAxisOrder();
  serializeT(static_cast<int32_t>(axis_order.size()));
  for (auto &axis : axis_order)
    serializeT(static_cast<int32_t>(axis));

  // serialize output shape
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::GatherOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // serialize parameters
  serializeT<int32_t>(op.getAxis());
  // serialize output shape
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::SigmoidOp & /*op*/) { _curOp->paramStartOffset = _buffer.size(); }

void Serializer::visit(mir::ops::LeakyReluOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  serializeT<float>(op.getAlpha());
  serializeShape(op.getOutputShape(0));
}

void Serializer::serialize(vector<unique_ptr<sir::Action>> &inference_sequence)
{
  for (unique_ptr<sir::Action> &action : inference_sequence)
  {
    if (action->type != sir::Action::Type::callFunction)
      continue;
    _curOp = dynamic_cast<sir::CallFunction *>(action.get());
    _curOp->mirOp->accept(this);
  }
}

void Serializer::visit(mir::ops::OutputOp & /*op*/)
{
  // no parameters to dump
}

void Serializer::visit(mir::ops::AbsOp &)
{
  _curOp->paramStartOffset = _buffer.size();
  // no parameters to dump
}

void Serializer::visit(mir::ops::AddOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // Op type is known at codegen Time
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::DivOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // Op type is known at codegen Time
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::MaxOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // Op type is known at codegen Time
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::MulOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // Op type is known at codegen Time
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit(mir::ops::SubOp &op)
{
  _curOp->paramStartOffset = _buffer.size();
  // Op type is known at codegen Time
  serializeShape(op.getOutputShape(0));
}

void Serializer::visit_fallback(mir::Operation &) { throw std::runtime_error("NYI operation"); }

} // namespace nnc
