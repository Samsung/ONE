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

#include "Frontend.h"
#include "Context.h"
#include "GraphBuilderRegistry.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <map>
#include <set>
#include <string>

#include <cassert>
#include <stdexcept>

using namespace nncc::core::ADT;

using tensor::LexicalLayout;

Frontend::Frontend() : _prototxt{new ::caffe::NetParameter}, _caffemodel{new ::caffe::NetParameter}
{
  // DO NOTHING
}

enco::Bundle Frontend::load(void) const
{
  auto module = coco::Module::create();
  auto blk = module->entity()->block()->create();
  module->block()->append(blk);

  auto data = coco::Data::create();

  // For weight access
  caffeimport::WeightContext weight_ctx(_caffemodel.get());

  // For inter-layer communication
  std::map<std::string, tensor::Shape> shape_ctx;
  std::map<std::string, coco::Bag *> bag_ctx;

  std::set<std::string> bags;
  std::map<std::string, uint32_t> def_count;
  std::map<std::string, uint32_t> use_count;

  auto def = [&bags, &def_count, &use_count](const std::string &name) {
    if (bags.find(name) == bags.end())
    {
      bags.insert(name);
      def_count[name] = 0;
      use_count[name] = 0;
    }

    def_count.at(name) += 1;
  };

  auto use = [&use_count](const std::string &name) { use_count.at(name) += 1; };

  auto outputs = [&bags, &def_count, &use_count](void) {
    std::set<std::string> res;

    for (const auto &bag : bags)
    {
      if (def_count.at(bag) > use_count.at(bag))
      {
        res.insert(bag);
      }
    }

    return res;
  };

  caffeimport::GraphBuilderContext opbuilder_context(module.get(), data.get(), blk, shape_ctx,
                                                     bag_ctx, weight_ctx);

  for (const auto &layer : _prototxt->layer())
  {
    assert(layer.has_name());
    assert(layer.has_type());

    for (uint32_t n = 0; n < layer.top().size(); ++n)
    {
      def(layer.top(n));
    }

    for (uint32_t n = 0; n < layer.bottom().size(); ++n)
    {
      use(layer.bottom(n));
    }

    if (const auto *graph_builder = caffeimport::GraphBuilderRegistry::get().lookup(layer.type()))
    {
      graph_builder->build(layer, &opbuilder_context);
    }
    else
    {
      throw std::runtime_error{"Not supported: " + layer.type()};
    }
  }

  // Finalize: Create output for each top blob
  for (const auto &name : outputs())
  {
    const auto &shape = shape_ctx.at(name);
    auto bag = bag_ctx.at(name);

    auto output = module->entity()->output()->create(shape);

    output->bag(bag);
    output->name(name);
    output->reorder<LexicalLayout>();

    module->output()->insert(output);
  }

  enco::Bundle bundle;

  bundle.module(std::move(module));
  bundle.data(std::move(data));

  return std::move(bundle);
}
