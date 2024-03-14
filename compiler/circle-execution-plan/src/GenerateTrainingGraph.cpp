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
#include <luci/Profile/CircleNodeOrigin.h>
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
  std::map<std::string, uint32_t> back_name_to_tensor_index;
  auto origin_tensors = reader_origin.tensors();
  auto train_tensors = reader_train.tensors();
  for (uint32_t i = 0; i < train_tensors.size(); ++i)
  {
    auto tensor = train_tensors.at(i);
    back_name_to_tensor_index[tensor->name()->str()] = i;
  }


  for (uint32_t i = 0; i < origin_tensors.size(); ++i)
  {
    auto tensor = origin_tensors.at(i);
    auto tensor_name = tensor->name()->str();
    auto gradient_name = tensor_name + "_gradient";

    auto it_orig_name = back_name_to_tensor_index.find(tensor_name);
    auto it_grad_name = back_name_to_tensor_index.find(gradient_name);
    if (it_orig_name != back_name_to_tensor_index.end())
    {
      uint32_t origin_index = it_orig_name->second;
      result[origin_index] = i;
    }
    if (it_grad_name != back_name_to_tensor_index.end())
    {
      uint32_t origin_index = it_grad_name->second;
      result[origin_index] = i;
    }
  }

  return result;
}

#if 0
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
#endif


// Let's create backpropogation calculation graph:
// For this example (MSE error, graph: FC(no bias) -> Relu -> FC(no bias) ) it is:
//       PRED    TARGET
//          \    /
//            SUB     FC_InputActivation_2
//           /  \    /                  \
//          /     \/                     \
//         /     /  \    fc_weight_2      \
//          MUL      \    /                \
//          /         \  /            greater(> 0)
//          /          MUL                |
//          /           \                 |
//          /             \          cast (to float)
//     Weight_grad_2        \           /
//                            \        /
//                               MUL             FC_INPUT_1
//                                |                 |
//                             Transpose(1,0)     Transpose(1,0)
//                                 |                |
//                                   \            /
//                                 FullyConnected(BatchMatMul)
//                                          |
//                                         Weight_grad_1
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
  link(target_input_values_node_context, target_input_values_node);
  target_input_values_node->name(target_input_values_node->name() + "_target");
  copy_for_context(target_input_values_node_context, predicted_input_values_node);

  luci::CircleFullyConnected *fc_node_2 = nullptr;
  luci::CircleFullyConnected *fc_node_1 = nullptr;
#if 1
  for (auto node : postorder_traversal(loco::output_nodes(original_graph)))
  {
    if (dynamic_cast<luci::CircleFullyConnected *>(node) != nullptr and fc_node_1 == nullptr)
    {
      fc_node_1 = dynamic_cast<luci::CircleFullyConnected *>(node);
    } else if (dynamic_cast<luci::CircleFullyConnected *>(node) != nullptr and fc_node_1 != nullptr)
    {
      fc_node_2 = dynamic_cast<luci::CircleFullyConnected *>(node);
    }
  }
#endif // if 0

  assert(fc_node_1 != nullptr);

  // Get 1 and 2 FC node and its weights
  luci::CircleConst *weight_2 = dynamic_cast<luci::CircleConst *>(fc_node_2->weights());
  assert(weight_2 != nullptr);

  luci::CircleConst *weight_1 = dynamic_cast<luci::CircleConst *>(fc_node_1->weights());
  assert(weight_2 != nullptr);
  auto fc_1_from = must_cast<luci::CircleNode *>(fc_node_1->input());

  // Create outputs: result gradients for weights
  loco::Graph::OutputContext *output_context = training_graph->outputs();

  // Create weight 2 gradient output
  auto weight_gradient_2_context = output_context->create();
  auto weight_gradient_2_node = training_graph->nodes()->create<luci::CircleOutput>();
  copy_nodes_params(weight_gradient_2_node, weight_2);
  link(weight_gradient_2_context, weight_gradient_2_node);
  copy_for_context(weight_gradient_2_context, weight_gradient_2_node);

  // Create weight 1 gradient output
  auto weight_gradient_1_context = output_context->create();
  auto weight_gradient_1_node = training_graph->nodes()->create<luci::CircleOutput>();
  copy_nodes_params(weight_gradient_1_node, weight_1);
  link(weight_gradient_1_context, weight_gradient_1_node);
  copy_for_context(weight_gradient_1_context, weight_gradient_1_node);

  // Create Sub node
  auto sub = training_graph->nodes()->create<luci::CircleSub>();
  sub->fusedActivationFunction(luci::FusedActFunc::NONE);
  sub->x(predicted_input_values_node);
  sub->y(target_input_values_node);
  sub->name(target_input_values_node->name() + predicted_input_values_node->name());

  // Create Fc_InputActivation_2 node
  auto fc_input_activation_2_node = training_graph->nodes()->create<luci::CircleInput>();
  auto fc_input_activation_2_node_context = input_context->create();
  copy_nodes_params(fc_input_activation_2_node, fc_node_1);
  link(fc_input_activation_2_node_context, fc_input_activation_2_node);
  copy_for_context(fc_input_activation_2_node_context, fc_input_activation_2_node);

  // Create Fc_InputActivation_1 node
  auto fc_input_activation_1_node = training_graph->nodes()->create<luci::CircleInput>();
  auto fc_input_activation_1_node_context = input_context->create();
  copy_nodes_params(fc_input_activation_1_node, fc_1_from);
  link(fc_input_activation_1_node_context, fc_input_activation_1_node);
  copy_for_context(fc_input_activation_1_node_context, fc_input_activation_1_node);

  // Create Fc_Weight_2 node as const
//  auto fc_weight_2_node = training_graph->nodes()->create<luci::CircleInput>();
//  auto fc_weight_2_node_context = input_context->create();
//  copy_nodes_params(fc_weight_2_node, weight_2);
//  link(fc_weight_2_node_context, fc_weight_2_node);
//  copy_for_context(fc_weight_2_node_context, fc_weight_2_node);
  auto fc_w_2_const = training_graph->nodes()->create<luci::CircleConst>();
  copy_nodes_params(fc_w_2_const, weight_2);
  fc_w_2_const->size<loco::DataType::FLOAT32>(0);
  fc_w_2_const->shape_status(luci::ShapeStatus::VALID);

  // Create Mul 2 node
  auto mul_2 = training_graph->nodes()->create<luci::CircleMul>();
  mul_2->fusedActivationFunction(luci::FusedActFunc::NONE);
  mul_2->x(fc_input_activation_2_node);
  mul_2->y(sub);
  mul_2->name(weight_gradient_2_node->name() + "_gradient");

  // Connect grad weight 2 with mul_2
  weight_gradient_2_node->from(mul_2);

  // Create Const with one zero for greater op
  auto greater_const = training_graph->nodes()->create<luci::CircleConst>();
  greater_const->shape({1}); // scalar
  greater_const->dtype(loco::DataType::FLOAT32);
  greater_const->rank(1);
  greater_const->size<loco::DataType::FLOAT32>(1);
  greater_const->at<loco::DataType::FLOAT32>(0) = 0.f;
  greater_const->name("GREATER_CONST");

  // Create Greater op
  auto greater = training_graph->nodes()->create<luci::CircleGreater>();
  greater->x(fc_input_activation_2_node);
  greater->y(greater_const);
  greater->name(fc_input_activation_2_node->name() + greater_const->name());
  greater->dtype(loco::DataType::BOOL);

  // Create Cast op
  auto cast = training_graph->nodes()->create<luci::CircleCast>();
  cast->x(greater);
  cast->in_data_type(loco::DataType::BOOL);
  cast->out_data_type(loco::DataType::FLOAT32);
  cast->name(greater->name() + "_cast");
  cast->dtype(loco::DataType::FLOAT32);

  // Create Mul 3 node
  auto mul_3 = training_graph->nodes()->create<luci::CircleMul>();
  mul_3->fusedActivationFunction(luci::FusedActFunc::NONE);
  mul_3->x(sub);
  mul_3->y(fc_w_2_const);
  mul_3->name(sub->name() + fc_w_2_const->name());

  // Create Mul 1 node
  auto mul_1 = training_graph->nodes()->create<luci::CircleMul>();
  mul_1->fusedActivationFunction(luci::FusedActFunc::NONE);
  mul_1->x(cast);
  mul_1->y(mul_3);
  mul_1->name(cast->name() + mul_3->name());

  // Create Const for transpose
  auto transpose_const = training_graph->nodes()->create<luci::CircleConst>();
  transpose_const->shape({2}); // scalar
  transpose_const->dtype(loco::DataType::S32);
  transpose_const->rank(1);
  transpose_const->size<loco::DataType::S32>(2);
  transpose_const->at<loco::DataType::S32>(0) = 1;
  transpose_const->at<loco::DataType::S32>(1) = 0;
  transpose_const->name("TRANSPOSE_CONST");

  // Create Transpose 1 node
  auto transpose_left = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_left->a(mul_1);
  transpose_left->perm(transpose_const);
  transpose_left->name("trabnspose_left");

  // Create Transpose 2 node
  auto transpose_right= training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_right->a(fc_input_activation_1_node);
  transpose_right->perm(transpose_const);
  transpose_right->name("trabnspose_right");

  auto empty_bias = training_graph->nodes()->create<luci::CircleOutputExclude>();

  //Create FC node
  auto new_fc_node = training_graph->nodes()->create<luci::CircleFullyConnected>();
  new_fc_node->input(transpose_left);
  new_fc_node->weights(transpose_right);
  new_fc_node->fusedActivationFunction(luci::FusedActFunc::NONE);
  new_fc_node->bias(empty_bias);
  new_fc_node->name(weight_gradient_1_node->name() + "_gradient");

  weight_gradient_1_node->from(new_fc_node);

  return std::move(training_graph);
}
