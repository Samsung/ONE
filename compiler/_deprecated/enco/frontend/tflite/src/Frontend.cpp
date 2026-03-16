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
#include "Convert.h"
#include "TensorBags.h"
#include "GraphBuilderRegistry.h"

#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/Shape.h>

#include <iostream>

using namespace nncc::core::ADT;

namespace tflimport
{

/**
 * @brief Set module input operands and its information
 */
void set_module_inputs(coco::Module *m, TensorContext &ctx, TensorBags &bags,
                       const IndexVector &inputs)
{
  for (uint32_t n = 0; n < inputs.size(); ++n)
  {
    auto const tensor_id = inputs.at(n);

    auto const tensor_name = ctx.name(tensor_id);
    auto const tensor_shape = ctx.shape(tensor_id);
    auto const tensor_bag = bags.bag(tensor_id);

    auto input = m->entity()->input()->create(tensor_shape);

    input->name(tensor_name);
    input->bag(tensor_bag);
    input->reorder<tensor::LexicalLayout>();

    m->input()->insert(input);
  }
}

/**
 * @brief Set module output operands and its information
 */
void set_module_outputs(coco::Module *m, TensorContext &ctx, TensorBags &bags,
                        const IndexVector &outputs)
{
  for (uint32_t n = 0; n < outputs.size(); ++n)
  {
    auto const tensor_id = outputs.at(n);

    auto const tensor_name = ctx.name(tensor_id);
    auto const tensor_shape = ctx.shape(tensor_id);
    auto const tensor_bag = bags.bag(tensor_id);

    auto output = m->entity()->output()->create(tensor_shape);

    output->name(tensor_name);
    output->bag(tensor_bag);
    output->reorder<tensor::LexicalLayout>();

    m->output()->insert(output);
  }
}

/**
 * @brief Copy values of tfl tensors into coco::Data if the data was not copied
 */
void copy_tensors(GraphBuilderContext *ctx)
{
  auto d = ctx->d();

  // for each bag, check if bag is not allocated but tflite tensor has values
  for (auto &iter : ctx->bags())
  {
    auto tfl_tensor_id = iter.first;
    auto bag = iter.second;

    auto tfl_buffer = ctx->buffer().tensor_buffer<float>(ctx->graph(), tfl_tensor_id);

    // TODO remove this line when support int32 is ready
    if (ctx->tensor().type(tfl_tensor_id) == tflite::TensorType::TensorType_INT32)
    {
      std::cout << "*** INT32 COPYING IS NOT SUPPORTED ***" << std::endl;
      continue;
    }

    assert(ctx->tensor().type(tfl_tensor_id) == tflite::TensorType::TensorType_FLOAT32);

    auto span = d->f32()->weight(bag); // TODO support other type

    if (!(span.data() == nullptr && span.size() == 0)) // already allocated
      continue;

    if (tfl_buffer.ptr == nullptr || tfl_buffer.len == 0) // no data to copy
      continue;

    d->f32()->allocate(bag);

    auto ifm_span = d->f32()->weight(bag);
    for (uint32_t idx = 0; idx < tfl_buffer.len; ++idx)
    {
      ifm_span[idx] = tfl_buffer.ptr[idx];
    }
  }
}

} // namespace tflimport

Frontend::Frontend(std::unique_ptr<RawModel> &&raw) : _raw{std::move(raw)}
{
  // DO NOTHING
}

enco::Bundle Frontend::load(void) const
{
  auto model = _raw->model();

  assert(model->version() == 3);
  assert(model->subgraphs()->size() == 1);

  auto graph = model->subgraphs()->Get(0);

  auto m = coco::Module::create();
  auto d = coco::Data::create();

  tflimport::TensorContext tensor_context;
  tflimport::TensorBags tensor_bags;

  tensor_context.prepare(graph);
  tensor_bags.prepare(graph, m);

  auto inputs = tflimport::as_index_vector(graph->inputs());
  auto outputs = tflimport::as_index_vector(graph->outputs());

  tflimport::set_module_inputs(m.get(), tensor_context, tensor_bags, inputs);
  tflimport::set_module_outputs(m.get(), tensor_context, tensor_bags, outputs);

  auto blk = m->entity()->block()->create();
  m->block()->append(blk);

  auto opcodes = model->operator_codes();

  tflimport::TflBufferContext buffer_context(model);
  tflimport::TflOpCodeContext opcode_context(opcodes);

  auto operators = graph->operators();

  tflimport::GraphBuilderContext opbuilder_context(m.get(), d.get(), blk, tensor_bags,
                                                   tensor_context, buffer_context, graph);

  for (int i = 0; i < operators->Length(); ++i)
  {
    const auto *op = operators->Get(i);
    tflite::BuiltinOperator builtincode = opcode_context.builtin_code(op);

    if (const auto *graph_builder = tflimport::GraphBuilderRegistry::get().lookup(builtincode))
    {
      if (!graph_builder->validate(op))
      {
        throw std::runtime_error{"Invalid operator"};
      }

      graph_builder->build(op, &opbuilder_context);
    }
    else
    {
      std::string opcodename = opcode_context.opcode_name(op);
      throw std::runtime_error{"Not supported: " + opcodename};
    }

    // copying unfilled tensor value
    copy_tensors(&opbuilder_context);
  }

  // Create "Bundle"
  enco::Bundle bundle;

  bundle.module(std::move(m));
  bundle.data(std::move(d));

  return std::move(bundle);
}
