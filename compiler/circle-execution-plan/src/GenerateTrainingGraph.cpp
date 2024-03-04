/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <foder/FileLoader.h>

#include <luci/Importer.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include "ExecutionPlanner.h"

#include <arser/arser.h>

#include <luci/ImporterEx.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/UserSettings.h>
#include <luci/IR/CircleNodes.h>

#include "GenerateTrainingGraph.h"

constexpr float LAMBDA = 0.005;

using namespace training_graph;
using namespace loco;
using namespace luci;

void copy_nodes_params(luci::CircleNode *dest, luci::CircleNode *src)
{
  dest->name(src->name());
  dest->shape_status(src->shape_status());
  dest->dtype(src->dtype());
  dest->rank(src->rank());

  for (uint32_t i = 0; i < dest->rank(); ++i)
  {
    dest->dim(i) = src->dim(i);
  }
}

void copy_for_context(loco::GraphInput *input, CircleInput *node)
{
  input->name(node->name());
  input->dtype(node->dtype());

  auto shape = std::make_unique<TensorShape>();
  shape->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); ++i)
  {
    shape->dim(i) = node->dim(i);
  }
  input->shape(std::move(shape));
}

void copy_for_context(loco::GraphOutput *input, CircleOutput *node)
{
  input->name(node->name());
  input->dtype(node->dtype());

  auto shape = std::make_unique<TensorShape>();
  shape->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); ++i)
  {
    shape->dim(i) = node->dim(i);
  }
  input->shape(std::move(shape));
}

void visit(luci::CircleFullyConnected *fc_node, loco::Graph::InputContext *, loco::Graph::OutputContext *)
{
  // TODO: add code here
}

std::map<uint32_t, uint32_t> training_graph::GenerateTrainingGraph::createMapTensorsIndexes(const circle::Model *origin, const circle::Model *train)
{
  luci::CircleReader reader_origin;
  luci::CircleReader reader_train;

  reader_origin.parse(origin);

  reader_train.parse(train);

  reader_origin.select_subgraph(0);
  reader_train.select_subgraph(0);

  std::map<uint32_t, uint32_t> result;

  // Create and fill std::map from origin net: key is name, value is tensor index in origin model
  std::map<std::string, uint32_t> origin_name_to_tensor_index;
  auto origin_tensors = reader_origin.tensors();
  for (uint32_t i = 0; i < origin_tensors.size(); ++i)
  {
    auto tensor = origin_tensors.at(i);
    origin_name_to_tensor_index[tensor->name()->str()] = i;
  }

  auto train_tensors = reader_train.tensors();
  for (uint32_t i = 0; i < train_tensors.size(); ++i)
  {
    auto tensor = train_tensors.at(i);
    auto tensor_name = tensor->name()->str();
    if (origin_name_to_tensor_index.find(tensor_name) == origin_name_to_tensor_index.end())
      continue;

    uint32_t origin_index = origin_name_to_tensor_index[tensor_name];
    result[i] = origin_index;
  }

  return result;
}

// Let's create backpropogation calculation node:
// For this example (MSE error, SGD, lambda=L, one fc node with weight and bias) it is:
//       PRED    TARGET
//         \       /
//            SUB      FC_InputActivation
//           /   \        /
//          /     \      /
//         /       \    /
//    Bias_grad      MUL
//                    |
//                  Weight_grad
//
std::unique_ptr<loco::Graph> training_graph::GenerateTrainingGraph::createTrainingGraph()
{
  assert(_module->size() == 1); // Should be one graph
  auto original_graph = _module->graph();

  assert(original_graph->outputs()->size() == 1); // Should be one output
  auto predicted_output_values_node = dynamic_cast<luci::CircleOutput *>(output_nodes(original_graph).at(0));
  assert(predicted_output_values_node != nullptr);

  std::unique_ptr<loco::Graph> training_graph = std::make_unique<loco::Graph>();

  // Create inputs: outputs from original model and target values
  loco::Graph::InputContext *input_context = training_graph->inputs();
  // Predicted values
  auto predicted_input_values_node = training_graph->nodes()->create<luci::CircleInput>();
  auto predicted_input_values_node_context = input_context->create();
  copy_nodes_params(predicted_input_values_node, predicted_output_values_node);
  link(predicted_input_values_node_context, predicted_input_values_node);
  copy_for_context(predicted_input_values_node_context, predicted_input_values_node);
  // Target values
  auto target_input_values_node = training_graph->nodes()->create<luci::CircleInput>();
  auto target_input_values_node_context = input_context->create();
  copy_nodes_params(target_input_values_node, predicted_input_values_node); // Note: params of the node
                                                                            // the same for target as predicted
  link(target_input_values_node_context, target_input_values_node);
  target_input_values_node->name(target_input_values_node->name() + "_target");
  copy_for_context(target_input_values_node_context, predicted_input_values_node);

  luci::CircleFullyConnected *fc_node = nullptr;
#if 1
  for (auto node : loco::active_nodes(loco::output_nodes(original_graph)))
  {
    if (auto fc = dynamic_cast<luci::CircleFullyConnected *>(node))
    {
      fc_node = fc;
      break;
    }
  }
#endif // if 0

  assert(fc_node != nullptr);

  luci::CircleConst *weight = dynamic_cast<luci::CircleConst *>(fc_node->weights());
  assert(weight != nullptr);
  luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(fc_node->bias());
  assert(bias != nullptr);
  luci::CircleNode *fc_from = dynamic_cast<luci::CircleNode *>(fc_node->input());
  assert(fc_from != nullptr);

  // Create outputs: result gradients for weights and biases
  loco::Graph::OutputContext *output_context = training_graph->outputs();

  // Create weight gradient output
  auto weight_gradient_context = output_context->create();
  auto weight_gradient_node = training_graph->nodes()->create<luci::CircleOutput>();
  copy_nodes_params(weight_gradient_node, weight);
  link(weight_gradient_context, weight_gradient_node);
  copy_for_context(weight_gradient_context, weight_gradient_node);

  // Create bias gradient output
  auto bias_gradient_context = output_context->create();
  auto bias_gradient_node = training_graph->nodes()->create<luci::CircleOutput>();
  copy_nodes_params(bias_gradient_node, bias);
  link(bias_gradient_context, bias_gradient_node);
  copy_for_context(bias_gradient_context, bias_gradient_node);

  // Create Sub node
  auto sub = training_graph->nodes()->create<luci::CircleSub>();
  sub->fusedActivationFunction(luci::FusedActFunc::NONE);
  sub->x(predicted_input_values_node);
  sub->y(target_input_values_node);
  sub->name(bias->name());

  // Connect bias grad output and sub
  bias_gradient_node->from(sub);

  // Create Fc_InputActivation node
  auto fc_input_activation_node = training_graph->nodes()->create<luci::CircleInput>();
  auto fc_input_activation_node_context = input_context->create();
  copy_nodes_params(fc_input_activation_node, fc_from);
  link(fc_input_activation_node_context, fc_input_activation_node);
  copy_for_context(fc_input_activation_node_context, fc_input_activation_node);

  // Create Mul node
  auto mul = training_graph->nodes()->create<luci::CircleMul>();
  mul->fusedActivationFunction(luci::FusedActFunc::NONE);
  mul->x(sub);
  mul->y(fc_input_activation_node);
  mul->name(weight->name());

#if 0
  // Create a const for Lambda
  auto lambda_const = training_graph->nodes()->create<luci::CircleConst>();
  lambda_const->shape({}); // scalar
  lambda_const->dtype(loco::DataType::FLOAT32);
  lambda_const->rank(0);
  lambda_const->size<loco::DataType::FLOAT32>(1);
  lambda_const->at<loco::DataType::FLOAT32>(0) = LAMBDA;
  lambda_const->name("LAMBDA_CONST");

  // Create last mul operation
  auto mul_lambda = training_graph->nodes()->create<luci::CircleMul>();
  mul_lambda->fusedActivationFunction(luci::FusedActFunc::NONE);
  mul_lambda->x(mul);
  mul_lambda->y(lambda_const);
  mul_lambda->name("/mul_with_lambda");
#endif

  // Connect output weight gradient with MUL_LAMBDA
  weight_gradient_node->from(mul);

  return std::move(training_graph);
}
