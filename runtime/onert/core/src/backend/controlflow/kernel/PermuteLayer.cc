/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PermuteLayer.h"

#include "exec/ShapeConverter.h"

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

void PermuteLayer::run()
{
  assert(_src_tensors.size() == _dst_tensors.size());
  // check if output is not dynamic
  for (size_t i = 0; i < _src_tensors.size(); ++i)
  {
    auto dst_tensor = _dst_tensors.at(i);
    auto src_tensor = _src_tensors.at(i);
    if (dst_tensor->is_dynamic())
    {
      // getting output shape
      auto src_shape = getShape(src_tensor.get());

      // set output shape and output buffer
      ir::Shape new_shape =
          exec::convertShape(src_shape, src_tensor->layout(), dst_tensor->layout());

      const auto dst_index = _dst_dyn_alloc_info_map.at(dst_tensor).ind;
      _dst_dyn_alloc_info_map.at(dst_tensor).dyn_tensor_manager->allocate(dst_index, new_shape);
      assert(dst_tensor->buffer() != nullptr);
      // TODO Move setShape() above allocate()
      setShape(dst_tensor.get(), new_shape);
    }
    else
    {
      assert(exec::convertShape(getShape(src_tensor.get()), src_tensor->layout(),
                                dst_tensor->layout()) == getShape(dst_tensor.get()));
    }
  }
  IPermuteFunction::run();
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
