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

#include "Input.h"
#include "Convert.h"

#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <cassert>

using namespace nncc::core::ADT;

using tensor::LexicalLayout;
using tensor::num_elements;

namespace caffeimport
{

void InputBuilder::build(const ::caffe::LayerParameter &layer, GraphBuilderContext *context) const
{
  coco::Module *module = context->module();
  std::map<std::string, tensor::Shape> &shape_ctx = context->shape_ctx();
  std::map<std::string, coco::Bag *> &bag_ctx = context->bag_ctx();

  assert(layer.has_input_param());
  const auto &param = layer.input_param();

  for (uint32_t n = 0; n < layer.top_size(); ++n)
  {
    const auto &name = layer.top(n);
    const auto shape = as_tensor_shape(param.shape(n));

    auto bag = module->entity()->bag()->create(num_elements(shape));
    auto input = module->entity()->input()->create(shape);

    input->bag(bag);
    input->name(name);
    input->reorder<LexicalLayout>();

    module->input()->insert(input);

    bag_ctx[name] = bag;
    shape_ctx[name] = shape;
  }
}

} // namespace caffeimport
