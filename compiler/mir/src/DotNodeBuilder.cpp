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

#include "DotNodeBuilder.h"
#include "mir/OpDefs.h"

#include <sstream>

namespace mir
{

template <typename T> static std::string toString(const std::vector<T> &v)
{
  std::stringstream ss;
  ss << "[";
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    if (i != 0)
      ss << ", ";
    ss << v[i];
  }
  return ss.str();
}

DotNodeBuilder::DotNodeBuilder(const Operation &op)
{
  _type_name = getTypeName(op.getType());
  _id = op.getId();

  for (std::size_t i = 0; i < op.getNumInputs(); ++i)
  {
    _in_shapes.push_back(toString(op.getInputShape(i)));
  }

  for (std::size_t i = 0; i < op.getNumOutputs(); ++i)
  {
    _out_shapes.push_back(toString(op.getOutputShape(i)));
  }

  // Get attributes.
  const_cast<Operation &>(op).accept(this);
}

void DotNodeBuilder::visit(ops::AvgPool2DOp &op)
{
  addAttribute("Window size", toString(op.getWindowSize()));
  addAttribute("Strides", toString(op.getStrides()));
  addAttribute("Padding before", toString(op.getPaddingBefore()));
  addAttribute("Padding after", toString(op.getPaddingAfter()));
  addAttribute("Include pad", std::to_string(op.getIncludePad()));
}

void DotNodeBuilder::visit(ops::CappedReluOp &op)
{
  addAttribute("Cap", std::to_string(op.getCap()));
}

void DotNodeBuilder::visit(ops::ConcatOp &op)
{
  addAttribute("Axis", std::to_string(op.getAxis()));
}

void DotNodeBuilder::visit(ops::Conv2DOp &op)
{
  addAttribute("Strides", toString(op.getStrides()));
  addAttribute("Padding before", toString(op.getPaddingBefore()));
  addAttribute("Padding after", toString(op.getPaddingAfter()));
  addAttribute("Num groups", std::to_string(op.getNumGroups()));
  addAttribute("Data format", toString(op.getDataFormat()));
}

void DotNodeBuilder::visit(ops::DepthwiseConv2DOp &op)
{
  addAttribute("Strides", toString(op.getStrides()));
  addAttribute("Padding before", toString(op.getPaddingBefore()));
  addAttribute("Padding after", toString(op.getPaddingAfter()));
  addAttribute("Data format", toString(op.getDataFormat()));
}

void DotNodeBuilder::visit(ops::MaxPool2DOp &op)
{
  addAttribute("Window size", toString(op.getWindowSize()));
  addAttribute("Strides", toString(op.getStrides()));
  addAttribute("Padding before", toString(op.getPaddingBefore()));
  addAttribute("Padding after", toString(op.getPaddingAfter()));
  addAttribute("Data format", toString(op.getDataFormat()));
}

void DotNodeBuilder::visit(ops::SoftmaxOp &op)
{
  addAttribute("Axis", std::to_string(op.getAxis()));
}

void DotNodeBuilder::visit(ops::SliceOp &op)
{
  addAttribute("Starts", toString(op.getStarts()));
  addAttribute("Sizes", toString(op.getSizes()));
}

void DotNodeBuilder::visit(ops::DeConv2DOp &op)
{
  addAttribute("Padding before", toString(op.getPaddingBefore()));
  addAttribute("Padding after", toString(op.getPaddingAfter()));
  addAttribute("Strides", toString(op.getStrides()));
  addAttribute("Data format", toString(op.getDataFormat()));
}

void DotNodeBuilder::visit(ops::EluOp &op) { addAttribute("Alpha", std::to_string(op.getAlpha())); }

void DotNodeBuilder::visit(ops::SqueezeOp &op)
{
  addAttribute("Dims to squeeze", toString(op.getDimsToSqueeze()));
}

void mir::DotNodeBuilder::visit(ops::PadOp &op)
{
  addAttribute("Padding before", toString(op.getPaddingBefore()));
  addAttribute("Padding after", toString(op.getPaddingAfter()));
  addAttribute("Padding value", std::to_string(op.getPaddingValue()));
}

void DotNodeBuilder::visit(ops::ReduceMeanOp &op)
{
  addAttribute("Reduction dims", toString(op.getReductionDims()));
  addAttribute("Keep dims", std::to_string(op.getKeepDims()));
}

void DotNodeBuilder::visit(ops::ResizeOp &op)
{
  assert(op.getMode() == ops::ResizeOp::ResizeMethod::nearestNeighbor);
  (void)op;

  addAttribute("Interpolation mode", "nearestNeighbor");
}

void DotNodeBuilder::visit(ops::TransposeOp &op)
{
  addAttribute("Axis order", toString(op.getAxisOrder()));
}

void DotNodeBuilder::visit(ops::GatherOp &op)
{
  addAttribute("Axis", std::to_string(op.getAxis()));
}

void DotNodeBuilder::visit(mir::ops::LeakyReluOp &op)
{
  addAttribute("Alpha", std::to_string(op.getAlpha()));
}

void DotNodeBuilder::addAttribute(std::string name, std::string val)
{
  this->_attributes.emplace_back(std::move(name), std::move(val));
}

std::string DotNodeBuilder::getLabel() const
{
  std::stringstream ss;

  ss << "{" << _type_name << " | {{";

  for (std::size_t i = 0; i < _in_shapes.size(); ++i)
  {
    if (i != 0)
      ss << " | ";
    ss << "in" << i << ": " << _in_shapes[i];
  }

  ss << " | ";

  for (std::size_t i = 0; i < _out_shapes.size(); ++i)
  {
    if (i != 0)
      ss << " | ";
    ss << "out" << i << ": " << _out_shapes[i];
  }

  ss << "} | {";

  for (std::size_t i = 0; i < _attributes.size(); ++i)
  {
    if (i != 0)
      ss << " | ";
    ss << _attributes[i].first << ": " << _attributes[i].second;
  }

  ss << "}}}";

  return ss.str();
}

} // namespace mir
