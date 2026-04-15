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

#include "passes/transformations/DataFormatSwitcher.h"

#include "mir/TensorUtil.h"
#include "mir/ops/AvgPool2DOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/Deconv2DOp.h"
#include "mir/ops/DepthwiseConv2DOp.h"
#include "mir/ops/MaxPool2DOp.h"
#include "mir/ops/TransposeOp.h"

namespace nnc
{
DataFormatSwitcher::DataFormatSwitcher(const mir::DataFormat target_format)
  : _target_format(target_format)
{
}

DataFormatSwitcher::~DataFormatSwitcher() = default;

PassData DataFormatSwitcher::run(PassData data)
{
  _graph = static_cast<mir::Graph *>(data);
  assert(_graph);

  // Collect nodes which use DataFormat
  for (auto *node : _graph->getNodes())
  {
    switch (node->getType())
    { // nodes using DataFormat
      case mir::Operation::Type::avgPool2D:
      case mir::Operation::Type::conv2D:
      case mir::Operation::Type::deConv2D:
      case mir::Operation::Type::depthwiseConv:
      case mir::Operation::Type::maxPool2D:
        _candidates_for_switch.push_back(node);
        break;
      default:
        break; // not use DataFormat
    }
  }
  // Switch collected ops
  for (auto *op : _candidates_for_switch)
  {
    switch (op->getType())
    {
      case mir::Operation::Type::avgPool2D:
        switchAvgPool2D(dynamic_cast<mir::ops::AvgPool2DOp *>(op));
        break;
      case mir::Operation::Type::conv2D:
        switchConv2D(dynamic_cast<mir::ops::Conv2DOp *>(op));
        break;
      case mir::Operation::Type::deConv2D:
        switchDeConv2D(dynamic_cast<mir::ops::DeConv2DOp *>(op));
        break;
      case mir::Operation::Type::depthwiseConv:
        switchDepthwiseConv2D(dynamic_cast<mir::ops::DepthwiseConv2DOp *>(op));
        break;
      case mir::Operation::Type::maxPool2D:
        switchMaxPool2D(dynamic_cast<mir::ops::MaxPool2DOp *>(op));
        break;
      default:
        assert(false && "Can't switch DataFormat for this operation!");
    }
  }

  return _graph;
}

void DataFormatSwitcher::cleanup() { _candidates_for_switch.clear(); }

mir::Operation::Output *DataFormatSwitcher::insertTransposeBefore(mir::Operation::Output *out)
{
  mir::Operation::Output *new_out;
  if (_target_format == mir::DataFormat::NHWC)
    new_out = _graph->create<mir::ops::TransposeOp>(out, std::vector<std::size_t>{0, 2, 3, 1})
                ->getOutput(0); // NCHW -> NHWC
  else
    new_out = _graph->create<mir::ops::TransposeOp>(out, std::vector<std::size_t>{0, 3, 1, 2})
                ->getOutput(0); // NHWC -> NCHW
  if (out->getType().isQuantized())
    new_out->setQuantization(out->getType().getQuantization());
  return new_out;
}

mir::Operation::Output *DataFormatSwitcher::insertTransposeAfter(mir::Operation::Output *out)
{
  mir::Operation::Output *new_out;
  if (_target_format == mir::DataFormat::NHWC)
    new_out = _graph->create<mir::ops::TransposeOp>(out, std::vector<std::size_t>{0, 3, 1, 2})
                ->getOutput(0); // NHWC -> NCHW
  else
    new_out = _graph->create<mir::ops::TransposeOp>(out, std::vector<std::size_t>{0, 2, 3, 1})
                ->getOutput(0); // NCHW -> NHWC
  if (out->getType().isQuantized())
    new_out->setQuantization(out->getType().getQuantization());
  return new_out;
}

void DataFormatSwitcher::switchAvgPool2D(mir::ops::AvgPool2DOp *op)
{
  if (op->getDataFormat() == _target_format)
    return;

  auto *input = op->getInput(0);

  mir::AvgPool2DOpAttributes attributes(op->getAttributes());
  attributes.data_format = _target_format;

  auto *trans_in = insertTransposeBefore(input);

  auto new_pool = _graph->create<mir::ops::AvgPool2DOp>(trans_in, attributes);

  auto *trans_out = insertTransposeAfter(new_pool->getOutput(0));

  _graph->replaceNode(op, trans_out->getNode());
}

void DataFormatSwitcher::switchConv2D(mir::ops::Conv2DOp *op)
{
  if (op->getDataFormat() == _target_format)
    return;

  assert(op->getNumInputs() >= 2);
  auto *input = op->getInput(0);
  auto *kernel = op->getInput(1);

  mir::Conv2DOpAttributes attributes(op->getAttributes());
  attributes.data_format = _target_format;

  auto *trans_in = insertTransposeBefore(input);

  mir::Operation *new_conv;
  if (op->getNumInputs() == 2)
    new_conv = _graph->create<mir::ops::Conv2DOp>(trans_in, kernel, attributes);
  else
  {
    auto bias = op->getInput(2);
    new_conv = _graph->create<mir::ops::Conv2DOp>(trans_in, kernel, bias, attributes);
  }

  if (op->getOutput(0)->getType().isQuantized())
    new_conv->getOutput(0)->setQuantization(op->getOutput(0)->getType().getQuantization());

  auto *trans_out = insertTransposeAfter(new_conv->getOutput(0));

  _graph->replaceNode(op, trans_out->getNode());
}

void DataFormatSwitcher::switchDeConv2D(mir::ops::DeConv2DOp *op)
{
  if (op->getDataFormat() == _target_format)
    return;

  assert(op->getNumInputs() == 2);
  auto *input = op->getInput(0);
  auto *kernel = op->getInput(1);

  auto *trans_in = insertTransposeBefore(input);

  mir::Operation *new_deconv;
  mir::Deconv2DOpAttributes attributes(op->getAttributes());
  attributes.data_format = _target_format;
  if (attributes.padding_type == mir::ops::PaddingType::Explicit)
  {
    new_deconv = _graph->create<mir::ops::DeConv2DOp>(trans_in, kernel, attributes);
  }
  else
  {
    mir::Shape output_shape = op->getOutputShape(0);
    if (_target_format == mir::DataFormat::NHWC)
      output_shape = mir::transposeShape<0, 2, 3, 1>(output_shape);
    else
      output_shape = mir::transposeShape<0, 3, 1, 2>(output_shape);
    new_deconv = _graph->create<mir::ops::DeConv2DOp>(trans_in, kernel, attributes, output_shape);
  }

  auto *trans_out = insertTransposeAfter(new_deconv->getOutput(0));

  _graph->replaceNode(op, trans_out->getNode());
}

void DataFormatSwitcher::switchDepthwiseConv2D(mir::ops::DepthwiseConv2DOp *op)
{
  if (op->getDataFormat() == _target_format)
    return;

  assert(op->getNumInputs() >= 2);
  auto *input = op->getInput(0);
  auto *kernel = op->getInput(1);

  mir::Conv2DOpAttributes attributes(op->getAttributes());
  attributes.data_format = _target_format;

  auto *trans_in = insertTransposeBefore(input);

  mir::Operation *new_dw_conv;
  if (op->getNumInputs() == 2)
    new_dw_conv = _graph->create<mir::ops::DepthwiseConv2DOp>(trans_in, kernel, attributes);
  else
  {
    auto bias = op->getInput(2);
    new_dw_conv = _graph->create<mir::ops::DepthwiseConv2DOp>(trans_in, kernel, bias, attributes);
  }

  if (op->getOutput(0)->getType().isQuantized())
    new_dw_conv->getOutput(0)->setQuantization(op->getOutput(0)->getType().getQuantization());

  auto *trans_out = insertTransposeAfter(new_dw_conv->getOutput(0));

  _graph->replaceNode(op, trans_out->getNode());
}

void DataFormatSwitcher::switchMaxPool2D(mir::ops::MaxPool2DOp *op)
{
  if (op->getDataFormat() == _target_format)
    return;

  auto *input = op->getInput(0);

  mir::MaxPool2DOpAttributes attributes(op->getAttributes());
  attributes.data_format = _target_format;

  auto *trans_in = insertTransposeBefore(input);

  auto new_pool = _graph->create<mir::ops::MaxPool2DOp>(trans_in, attributes);

  auto *trans_out = insertTransposeAfter(new_pool->getOutput(0));

  _graph->replaceNode(op, trans_out->getNode());
}

} // namespace nnc
