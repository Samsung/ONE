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

#include "TFFormattedGraph.h"

#include <moco/IR/TFDialect.h>
#include <moco/IR/TFNodes.h>

#include "LogHelper.h"

#include <pepper/str.h>
#include <locoex/Service/COpFormattedGraph.h>
#include <oops/InternalExn.h>

#include <sstream>

namespace
{

std::string opname(uint32_t opnum)
{
  static std::string prefix{"tf."};

  switch (static_cast<moco::TFOpcode>(opnum))
  {
#define TENSORFLOW_NODE(OPCODE, CLASS) \
  case moco::TFOpcode::OPCODE:         \
    return prefix + #OPCODE;
#include <moco/IR/TFNodes.lst>
#undef TENSORFLOW_NODE
    default:
      break;
  };

  return prefix + "Invalid";
}

using namespace moco;
using namespace moco::tf;

/// TFNodeSummaryBuilder with default implementation
class TFNodeSummaryBuilderBase : public locop::NodeSummaryBuilder
{
public:
  TFNodeSummaryBuilderBase(const locop::SymbolTable *tbl) : _tbl{tbl}
  {
    // DO NOTHING
  }

public:
  bool build(const loco::Node *, locop::NodeSummary &s) const final;

protected:
#define TENSORFLOW_NODE(OPCODE, CLASS)                                 \
  virtual bool summary(const CLASS *node, locop::NodeSummary &s) const \
  {                                                                    \
    s.comments().append("Emitted by Default NodeSummaryBuilder");      \
    s.state(locop::NodeSummary::State::PartiallyKnown);                \
    return true;                                                       \
  }
#include <moco/IR/TFNodes.lst>
#undef TENSORFLOW_NODE

protected:
  const locop::SymbolTable *tbl(void) const { return _tbl; }

  // Please do not use _tbl directly and use tbl().
  // This will be changed to private in near future.
protected:
  const locop::SymbolTable *_tbl;
};

class TFNodeSummaryBuilder final : public TFNodeSummaryBuilderBase
{
public:
  TFNodeSummaryBuilder(const locop::SymbolTable *tbl) : TFNodeSummaryBuilderBase(tbl)
  {
    // DO NOTHING
  }

private:
#define IMPLEMENT(CLASS) bool summary(const CLASS *, locop::NodeSummary &) const final
  IMPLEMENT(TFAdd);
  IMPLEMENT(TFAvgPool);
  IMPLEMENT(TFBiasAdd);
  IMPLEMENT(TFConcatV2);
  IMPLEMENT(TFConst);
  IMPLEMENT(TFConv2D);
  IMPLEMENT(TFConv2DBackpropInput);
  IMPLEMENT(TFDepthwiseConv2dNative);
  IMPLEMENT(TFFusedBatchNorm);
  IMPLEMENT(TFMaximum);
  IMPLEMENT(TFMaxPool);
  IMPLEMENT(TFMean);
  IMPLEMENT(TFMul);
  IMPLEMENT(TFPack);
  IMPLEMENT(TFReshape);
  IMPLEMENT(TFRsqrt);
  IMPLEMENT(TFShape);
  IMPLEMENT(TFSoftmax);
  IMPLEMENT(TFSqueeze);
  IMPLEMENT(TFStopGradient);
  IMPLEMENT(TFStridedSlice);
  IMPLEMENT(TFTanh);
  // For virtual nodes
  IMPLEMENT(TFPush);
#undef IMPLEMENT
};

bool TFNodeSummaryBuilderBase::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (node->dialect() != TFDialect::get())
    return false;

#define TENSORFLOW_NODE(OPCODE, CLASS)                    \
  if (dynamic_cast<const CLASS *>(node))                  \
  {                                                       \
    s.opname(opname(node->opnum()));                      \
    return summary(dynamic_cast<const CLASS *>(node), s); \
  }
#include <moco/IR/TFNodes.lst>
#undef TENSORFLOW_NODE

  return false;
}

bool TFNodeSummaryBuilder::summary(const TFAdd *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFAvgPool *node, locop::NodeSummary &s) const
{
  s.args().append("value", tbl()->lookup(node->value()));
  s.args().append("ksize", pepper::str(node->ksize()));
  s.args().append("strides", pepper::str(node->strides()));
  s.args().append("padding", node->padding());
  s.args().append("data_layout", node->data_layout());

  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFBiasAdd *node, locop::NodeSummary &s) const
{
  s.args().append("value", tbl()->lookup(node->value()));
  s.args().append("bias", tbl()->lookup(node->bias()));
  s.args().append("data_layout", node->data_layout());
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFConcatV2 *node, locop::NodeSummary &s) const
{
  for (uint32_t n = 0; n < node->num_values(); ++n)
  {
    std::ostringstream ss;
    ss << "values(" << n << ")";
    s.args().append(ss.str(), tbl()->lookup(node->values(n)));
  }
  s.args().append("axis", tbl()->lookup(node->axis()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFConst *node, locop::NodeSummary &s) const
{
  std::ostringstream ss;

  auto dtype = node->dtype();
  switch (dtype)
  {
    case loco::DataType::S32:
      ss << node->size<loco::DataType::S32>();
      break;
    case loco::DataType::FLOAT32:
      ss << node->size<loco::DataType::FLOAT32>();
      break;
    default:
      INTERNAL_EXN_V("Unsupported data type", node->name());
  }
  s.args().append("size", ss.str());
  s.state(locop::NodeSummary::State::PartiallyKnown);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFConv2D *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("filter", tbl()->lookup(node->filter()));
  s.args().append("padding", node->padding());
  s.args().append("data_layout", node->data_layout());
  s.args().append("strides", pepper::str(node->strides()));
  s.state(locop::NodeSummary::State::PartiallyKnown);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFConv2DBackpropInput *node, locop::NodeSummary &s) const
{
  s.args().append("input_sizes", tbl()->lookup(node->input_sizes()));
  s.args().append("filter", tbl()->lookup(node->filter()));
  s.args().append("out_backprop", tbl()->lookup(node->out_backprop()));
  s.args().append("padding", node->padding());
  s.args().append("data_layout", node->data_layout());
  s.args().append("strides", pepper::str(node->strides()));
  s.state(locop::NodeSummary::State::PartiallyKnown);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFDepthwiseConv2dNative *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("filter", tbl()->lookup(node->filter()));
  s.args().append("padding", node->padding());
  s.args().append("data_layout", node->data_layout());
  s.args().append("strides", pepper::str(node->strides()));
  s.state(locop::NodeSummary::State::PartiallyKnown);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFFusedBatchNorm *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("scale", tbl()->lookup(node->scale()));
  s.args().append("offset", tbl()->lookup(node->offset()));
  s.args().append("mean", tbl()->lookup(node->mean()));
  s.args().append("variance", tbl()->lookup(node->variance()));
  s.args().append("epsilon", pepper::str(node->epsilon()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFMaximum *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFMaxPool *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("ksize", pepper::str(node->ksize()));
  s.args().append("strides", pepper::str(node->strides()));
  s.args().append("padding", node->padding());
  s.args().append("data_layout", node->data_layout());

  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFMean *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("reduction_indices", tbl()->lookup(node->reduction_indices()));
  s.args().append("keep_dims", pepper::str(node->keep_dims()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFMul *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFPack *node, locop::NodeSummary &s) const
{
  s.args().append("N", pepper::str(node->N()));
  s.args().append("axis", pepper::str(node->axis()));
  for (uint32_t n = 0; n < node->N(); ++n)
    s.args().append("values", tbl()->lookup(node->values(n)));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFReshape *node, locop::NodeSummary &s) const
{
  s.args().append("tensor", tbl()->lookup(node->tensor()));
  s.args().append("shape", tbl()->lookup(node->shape()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFRsqrt *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFShape *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.state(locop::NodeSummary::State::PartiallyKnown);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFSoftmax *node, locop::NodeSummary &s) const
{
  s.args().append("logits", tbl()->lookup(node->logits()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFSqueeze *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("squeeze_dims", pepper::str(node->squeeze_dims()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFStopGradient *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFStridedSlice *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("begin", tbl()->lookup(node->begin()));
  s.args().append("end", tbl()->lookup(node->end()));
  if (node->strides() != nullptr)
    s.args().append("strides", tbl()->lookup(node->strides()));
  s.args().append("begin_mask", pepper::str(node->begin_mask()));
  s.args().append("end_mask", pepper::str(node->end_mask()));
  s.args().append("ellipsis_mask", pepper::str(node->ellipsis_mask()));
  s.args().append("new_axis_mask", pepper::str(node->new_axis_mask()));
  s.args().append("shrink_axis_mask", pepper::str(node->shrink_axis_mask()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFNodeSummaryBuilder::summary(const TFTanh *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

// For virtual nodes
bool TFNodeSummaryBuilder::summary(const TFPush *node, locop::NodeSummary &s) const
{
  s.args().append("index", node->indexed() ? pepper::str(node->index()) : "?");
  s.args().append("from", tbl()->lookup(node->from()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool MocoNodeSummaryBuilder::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (locop::CanonicalNodeSummaryBuilder(_tbl).build(node, s))
  {
    return true;
  }

  if (TFNodeSummaryBuilder(_tbl).build(node, s))
  {
    return true;
  }

  if (locoex::COpNodeSummaryBuilder(_tbl).build(node, s))
  {
    return true;
  }

  return false;
}

} // namespace tf
} // namespace moco
