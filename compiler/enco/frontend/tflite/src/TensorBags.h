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

#ifndef __TENSOR_BAGS_H__
#define __TENSOR_BAGS_H__

#include "Convert.h"

#include <coco/IR/Data.h>
#include <coco/IR/Module.h>

#include <schema_generated.h>

#include <map>

using namespace nncc::core::ADT;

namespace tflimport
{

/**
 * @brief Pre-creates coco:Bags for each operands(tensors)
 */
class TensorBags
{
public:
  void prepare(const tflite::SubGraph *graph, std::unique_ptr<coco::Module> &m)
  {
    for (uint32_t tensor_id = 0; tensor_id < graph->tensors()->size(); ++tensor_id)
    {
      auto const tensor_info = graph->tensors()->Get(tensor_id);
      auto const tensor_shape = as_tensor_shape(tensor_info->shape());
      auto const tensor_bag = m->entity()->bag()->create(num_elements(tensor_shape));

      _bag_ctx[tensor_id] = tensor_bag;
    }
  }

  coco::Bag *bag(int32_t tensor_id) { return _bag_ctx[tensor_id]; }

public:
  std::map<uint32_t, coco::Bag *>::iterator begin() { return _bag_ctx.begin(); }

  std::map<uint32_t, coco::Bag *>::iterator end() { return _bag_ctx.end(); }

private:
  std::map<uint32_t, coco::Bag *> _bag_ctx;
};

} // namespace tflimport

#endif // __TENSOR_BAGS_H__
