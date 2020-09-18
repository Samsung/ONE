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
  // PermuteLayer infers dynamic shape inside itself whenever run is called for the following
  // reasons:
  // 1. PermuteLayer has to access dynamic tensor manager for input/output tensors of other backends
  // 2. Other controlflow operation(If/While) uses this layout for copying tensors of other
  // subgraphs(with other backends)
  // 3. This infering code is placed here to avoid duplicated code that can be caused by above 2
  // reasons

  // check if output is not dynamic
  for (size_t i = 0; i < _src_tensors.size(); ++i)
  {
    auto dst_tensor = _dst_tensors.at(i);
    auto src_tensor = _src_tensors.at(i);
    if (src_tensor->is_dynamic() || dst_tensor->is_dynamic())
    {
      // getting output shape
      auto src_shape = src_tensor->getShape();
      VERBOSE(Permute) << "SRC SHAPE : " << src_shape << std::endl;

      // set output shape and output buffer
      ir::Shape new_shape =
          exec::convertShape(src_shape, src_tensor->layout(), dst_tensor->layout());

      try
      {
        if (!dst_tensor->applyShape(new_shape))
          throw std::runtime_error{
              "Error: PermuteLayer: output's TensorManager does not support dynamic tensor"};
        assert(dst_tensor->buffer() != nullptr);
      }
      catch (const std::out_of_range &e)
      {
        std::cerr << "Error: out_of_range in PermuteLayer: output's TensorManager does not support "
                     "dynamic tensor"
                  << '\n';
        throw;
      }
    }
    assert(exec::convertShape(src_tensor->getShape(), src_tensor->layout(), dst_tensor->layout()) ==
           dst_tensor->getShape());
  }
  IPermuteFunction::run();
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
