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

#include <enco/Frontend.h>
#include <cmdline/View.h>

#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <memory>

using namespace nncc::core::ADT;

namespace
{

//
// Dummy frontend for testing
//
struct Frontend final : public enco::Frontend
{
  enco::Bundle load(void) const override
  {
    auto m = coco::Module::create();
    auto d = coco::Data::create();

    // Create an input
    {
      const tensor::Shape shape{1, 3, 3, 1};

      auto bag = m->entity()->bag()->create(9);
      auto input = m->entity()->input()->create(shape);

      input->bag(bag);
      input->name("input");
      input->reorder<tensor::LexicalLayout>();

      m->input()->insert(input);
    }

    // Create an output
    {
      const tensor::Shape shape{1, 3, 3, 1};

      auto bag = m->entity()->bag()->create(9);
      auto output = m->entity()->output()->create(shape);

      output->bag(bag);
      output->name("output");
      output->reorder<tensor::LexicalLayout>();

      m->output()->insert(output);
    }

    enco::Bundle bundle;

    bundle.module(std::move(m));
    bundle.data(std::move(d));

    return std::move(bundle);
  }
};

} // namespace

extern "C" std::unique_ptr<enco::Frontend> make_frontend(const cmdline::View &cmdline)
{
  return std::make_unique<Frontend>();
}
