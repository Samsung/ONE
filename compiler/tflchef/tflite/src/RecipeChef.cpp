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

#include <tflchef/RecipeChef.h>

#include "Convert.h"
#include "TFliteImport.h"
#include "TFliteOpChef.h"
#include "TFliteOpChefs.h"
#include "TFliteOpRegistry.h"

#include <fstream>
#include <sstream>

namespace tflchef
{

void set_inputs(TFliteImport *import, tflchef::Operation *operation, const tflite::Operator *op)
{
  auto tensors = import->tensors();
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());

  for (auto input : inputs)
  {
    if (input == -1)
    {
      operation->add_input("");
    }
    else
    {
      auto tensor = tensors->Get(input);
      std::string name = tensor_name(tensor);
      operation->add_input(name);
    }
  }
}

void set_outputs(TFliteImport *import, tflchef::Operation *operation, const tflite::Operator *op)
{
  auto tensors = import->tensors();
  const std::vector<int32_t> &outputs = as_index_vector(op->outputs());

  for (auto output : outputs)
  {
    auto tensor = tensors->Get(output);
    std::string name = tensor_name(tensor);
    operation->add_output(name);
  }
}

/**
 * @brief This will build ModelRecipe from tflite::Model
 *        First to check operand filler options by scanning all operators,
 *        then translate all operands and operators.
 *        Last will set network inputs and outputs.
 */
std::unique_ptr<ModelRecipe> generate_recipe(const tflite::Model *model)
{
  std::unique_ptr<ModelRecipe> model_recipe{new ModelRecipe()};

  TFliteImport tflite_import(model);

  assert(tflite_import.num_subgraph() == 1);
  tflite_import.select_sub_graph(0);

  auto tensors = tflite_import.tensors();
  auto buffers = tflite_import.buffers();
  auto operators = tflite_import.operators();

  // operand fillers for adding all operators
  for (uint32_t i = 0; i < operators->Length(); ++i)
  {
    const auto *op = operators->Get(i);
    tflite::BuiltinOperator builtincode = tflite_import.builtin_code(op);

    if (const auto *graph_builder = TFliteOpRegistry::get().lookup(builtincode))
    {
      graph_builder->filler(op, &tflite_import, model_recipe.get());
    }
    else
    {
      std::string opcodename = tflite_import.opcode_name(op);
      throw std::runtime_error{"Not supported: " + opcodename};
    }
  }

  // add all operands(tensors)
  for (uint32_t i = 0; i < tensors->Length(); ++i)
  {
    auto tensor = tensors->Get(i);

    // check buffer
    if (tensor->buffer() >= buffers->size())
      throw std::runtime_error{"file load failed"};

    ::tflchef::Operand *operand = model_recipe->add_operand();

    operand->set_name(tensor_name(tensor));
    operand->set_type(as_tflchef_type(tensor->type()));

    if (tensor->shape())
    {
      std::vector<int32_t> dims = as_index_vector(tensor->shape());
      ::tflchef::TensorShape *shape = operand->mutable_shape();
      for (auto dim : dims)
      {
        shape->add_dim(dim);
      }
    }

    // filler for weights, bias and so on
    std::vector<int32_t> expvalues;
    std::vector<float> expfvalues;
    if (tflite_import.get_tensor_filler(i))
    {
      tflchef::TensorFiller *filler = operand->mutable_filler();
      // Note: it is OK to use random weights for functionality validation
      filler->set_tag("gaussian");
      filler->add_arg("0.0"); // average
      filler->add_arg("0.1"); // standard deviation
    }
    else if (tflite_import.get_tensor_filler(i, expvalues))
    {
      tflchef::TensorFiller *filler = operand->mutable_filler();
      filler->set_tag("explicit");
      for (auto value : expvalues)
      {
        std::ostringstream ss;
        ss << value;
        filler->add_arg(ss.str());
      }
    }
    else if (tflite_import.get_tensor_filler(i, expfvalues))
    {
      tflchef::TensorFiller *filler = operand->mutable_filler();
      filler->set_tag("explicit");
      for (auto value : expfvalues)
      {
        std::ostringstream ss;
        ss << value;
        filler->add_arg(ss.str());
      }
    }

    auto quant = tensor->quantization();
    if (quant != nullptr)
    {
      // Note: Calling 'operand->mutable_quant()' will create empty 'quant' node
      // in the recipe file. We want this only when valid parameter exist.
      if (quant->min() != nullptr && quant->min()->size() > 0)
      {
        tflchef::TensorQuantization *chef_quant = operand->mutable_quant();
        for (uint32_t idx = 0; idx < quant->min()->size(); ++idx)
          chef_quant->add_min(quant->min()->Get(idx));
      }
      if (quant->max() != nullptr && quant->max()->size() > 0)
      {
        tflchef::TensorQuantization *chef_quant = operand->mutable_quant();
        for (uint32_t idx = 0; idx < quant->max()->size(); idx++)
          chef_quant->add_max(quant->max()->Get(idx));
      }
      if (quant->scale() != nullptr && quant->scale()->size() > 0)
      {
        tflchef::TensorQuantization *chef_quant = operand->mutable_quant();
        for (uint32_t idx = 0; idx < quant->scale()->size(); ++idx)
          chef_quant->add_scale(quant->scale()->Get(idx));
      }
      if (quant->zero_point() != nullptr && quant->zero_point()->size() > 0)
      {
        tflchef::TensorQuantization *chef_quant = operand->mutable_quant();
        for (uint32_t idx = 0; idx < quant->zero_point()->size(); ++idx)
          chef_quant->add_zero_point(quant->zero_point()->Get(idx));
      }
      tflchef::TensorQuantization *chef_quant = operand->mutable_quant();
      chef_quant->set_quantized_dimension(quant->quantized_dimension());
    }
  }

  // add all operators
  for (uint32_t i = 0; i < operators->Length(); ++i)
  {
    const auto *op = operators->Get(i);
    tflite::BuiltinOperator builtincode = tflite_import.builtin_code(op);

    if (const auto *graph_builder = TFliteOpRegistry::get().lookup(builtincode))
    {
      auto operation = graph_builder->build(op, &tflite_import, model_recipe.get());

      // common for all operators: inputs, outputs
      set_inputs(&tflite_import, operation, op);
      set_outputs(&tflite_import, operation, op);
    }
    else
    {
      std::string opcodename = tflite_import.opcode_name(op);
      throw std::runtime_error{"Not supported: " + opcodename};
    }
  }

  // network inputs/outputs
  const std::vector<int32_t> &inputs = tflite_import.inputs();
  const std::vector<int32_t> &outputs = tflite_import.outputs();

  for (const auto input : inputs)
  {
    auto tensor = tensors->Get(input);
    std::string name = tensor_name(tensor);

    model_recipe->add_input(name);
  }
  for (const auto output : outputs)
  {
    auto tensor = tensors->Get(output);
    std::string name = tensor_name(tensor);

    model_recipe->add_output(name);
  }

  return std::move(model_recipe);
}

bool write_recipe(const std::string &filename, std::unique_ptr<ModelRecipe> &recipe)
{
  std::fstream fo(filename, std::ios::binary | std::ios::out);

  if (!fo.is_open())
  {
    throw std::runtime_error{"file store failed"};
  }

  // Note: SerializeToString() or SerializeToOstream() writes in binary mode
  // DebugString() and Utf8DebugString() will print as a human readable text
  fo << recipe->Utf8DebugString();

  fo.close();

  return true;
}

} // namespace tflchef
