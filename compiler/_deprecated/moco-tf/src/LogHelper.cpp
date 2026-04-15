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

#include "LogHelper.h"

namespace loco
{

std::ostream &operator<<(std::ostream &os, const loco::FeatureShape &feature_shape)
{
  os << "[" << feature_shape.count().value() << "," << feature_shape.height().value() << ","
     << feature_shape.width().value() << "," << feature_shape.depth().value() << "]";
  return os;
}

std::ostream &operator<<(std::ostream &os, const loco::FilterShape &filter_shape)
{
  os << "[" << filter_shape.height().value() << "," << filter_shape.width().value() << ","
     << filter_shape.depth().value() << "," << filter_shape.count().value() << "]";
  return os;
}

std::ostream &operator<<(std::ostream &os, const loco::TensorShape &tensor_shape)
{
  os << "[";
  for (uint32_t r = 0; r < tensor_shape.rank(); ++r)
  {
    if (r)
      os << ",";
    os << tensor_shape.dim(r).value();
  }
  os << "]";
  return os;
}

std::ostream &operator<<(std::ostream &os, const loco::Padding2D &pad)
{
  os << "[TLBR " << pad.top() << "," << pad.left() << "," << pad.bottom() << "," << pad.right()
     << "]";

  return os;
}

} // namespace loco

std::ostream &operator<<(std::ostream &os, const std::vector<int64_t> &vi64)
{
  for (auto vi : vi64)
  {
    os << vi << " ";
  }
  return os;
}

#include "TFFormattedGraph.h"

namespace moco
{
namespace tf
{

FormattedGraph fmt(loco::Graph *g)
{
  auto node_summary_builder = std::make_unique<TFNodeSummaryBuilderFactory>();
  return std::move(locop::fmt<locop::LinearV1>(g).with(std::move(node_summary_builder)));
}

} // namespace tf
} // namespace moco
