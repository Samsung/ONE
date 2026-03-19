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

#include "moco/Import/Nodes/Placeholder.h"

#include <moco/IR/Nodes/TFPlaceholder.h>

#include <moco/Names.h>
#include <plier/tf/Convert.h>

#include <cassert>
#include <stdexcept>

namespace moco
{

bool PlaceholderGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (!plier::tf::has_attrs(node, {"dtype", "shape"}))
    return false;

  loco::DataType dtype = plier::tf::as_loco_datatype(plier::tf::get_datatype_attr(node, "dtype"));
  if (dtype != loco::DataType::FLOAT32)
    return false;
  // TODO support other types

  return true;
}

void PlaceholderGraphBuilder::build(const tensorflow::NodeDef &node,
                                    GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();

  loco::DataType dtype = plier::tf::as_loco_datatype(plier::tf::get_datatype_attr(node, "dtype"));
  const auto shape = plier::tf::get_shape_attr(node, "shape");
  // TODO handle for unknown rank
  assert(!shape.unknown_rank());
  int64_t num_dims = shape.dim_size();

  // TODO support other types
  assert(dtype == loco::DataType::FLOAT32);

  // Create a "Placeholder" node as an input
  auto placeholder_node = graph->nodes()->create<moco::TFPlaceholder>();
  placeholder_node->name(node.name());
  placeholder_node->dtype(dtype);

  // Setting shape info.
  placeholder_node->rank(num_dims);
  for (int64_t d = 0; d < num_dims; d++)
  {
    assert(shape.dim(d).size() < std::numeric_limits<uint32_t>::max());
    int64_t dim_value = shape.dim(d).size();
    if (dim_value >= 0)
    {
      uint32_t dim_value32 = static_cast<uint32_t>(dim_value);
      placeholder_node->dim(d) = dim_value32;
    }
    else
    {
      placeholder_node->dim(d).unset();
      // TODO Remove assert() and do implement
      // NOTE Current implementation assumes dim is all know
      assert(false);
    }
  }

  // register string-name to node
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, placeholder_node);
}

} // namespace moco
