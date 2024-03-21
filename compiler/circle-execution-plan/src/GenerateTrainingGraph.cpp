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
#include <queue>
#include <stack>

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

#if 0
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
#endif


#if 0
// Let's create backpropogation calculation node:
// For this example (MSE error, SGD, lambda=L, one conv2d node with weight and bias) it is:
//       PRED    TARGET
//         \       /
//            SUB      Conv2D_InputActivation
//           /   \        /
//          /     \      /
//         /       \    /
//    Bias_grad   Conv2D_Grad_Weight
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

  luci::CircleConv2D *conv_node = nullptr;

  for (auto node : loco::active_nodes(loco::output_nodes(original_graph)))
  {
    if (auto cv = dynamic_cast<luci::CircleConv2D *>(node))
    {
      conv_node = cv;
      break;
    }
  }

assert(conv_node != nullptr);

  luci::CircleConst *weight = dynamic_cast<luci::CircleConst *>(conv_node->filter());
  assert(weight != nullptr);
  luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(conv_node->bias());
  assert(bias != nullptr);
  luci::CircleNode *conv_from = dynamic_cast<luci::CircleNode *>(conv_node->input());
  assert(conv_from != nullptr);

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
  sub->name("_sub");

  // Create Const for transpose_1 and transpose_2
  auto transpose_const = training_graph->nodes()->create<luci::CircleConst>();
  transpose_const->shape({4}); // scalar
  transpose_const->dtype(loco::DataType::S32);
  transpose_const->rank(1);
  transpose_const->size<loco::DataType::S32>(4);
  transpose_const->at<loco::DataType::S32>(0) = 0;
  transpose_const->at<loco::DataType::S32>(1) = 3;
  transpose_const->at<loco::DataType::S32>(2) = 1;
  transpose_const->at<loco::DataType::S32>(3) = 2;
  transpose_const->name("TRANSPOSE_CONST");

  // Create Const for transpose_3
  auto transpose_const_2 = training_graph->nodes()->create<luci::CircleConst>();
  transpose_const_2->shape({4}); // scalar
  transpose_const_2->dtype(loco::DataType::S32);
  transpose_const_2->rank(1);
  transpose_const_2->size<loco::DataType::S32>(4);
  transpose_const_2->at<loco::DataType::S32>(0) = 0;
  transpose_const_2->at<loco::DataType::S32>(1) = 2;
  transpose_const_2->at<loco::DataType::S32>(2) = 3;
  transpose_const_2->at<loco::DataType::S32>(3) = 1;
  transpose_const_2->name("TRANSPOSE_CONST");

  // Create Const for index
  auto index_const = training_graph->nodes()->create<luci::CircleConst>();
  index_const->shape({3}); // scalar
  index_const->dtype(loco::DataType::S32);
  index_const->rank(1);
  index_const->size<loco::DataType::S32>(3);
  index_const->at<loco::DataType::S32>(0) = 0;
  index_const->at<loco::DataType::S32>(1) = 1;
  index_const->at<loco::DataType::S32>(2) = 2;
  index_const->name("index");

  // Create ReduceSum
  // Create CircleReduceMax operation
  auto reduce_sum = training_graph->nodes()->create<luci::CircleReduceSum>();
  reduce_sum->input(sub);
  reduce_sum->reduction_indices(index_const);
  reduce_sum->keep_dims(false);
  reduce_sum->name(bias_gradient_node->name());

  // Connect bias grad output and sub
  bias_gradient_node->from(reduce_sum);

  // Create conv_InputActivation node
  auto conv_input_activation_node = training_graph->nodes()->create<luci::CircleInput>();
  auto conv_input_activation_node_context = input_context->create();
  copy_nodes_params(conv_input_activation_node, conv_from);
  link(conv_input_activation_node_context, conv_input_activation_node);
  copy_for_context(conv_input_activation_node_context, conv_input_activation_node);

  // Create Transpose_input node
  auto transpose_input = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_input->a(conv_input_activation_node);
  transpose_input->perm(transpose_const);
  transpose_input->name("trabnspose_left");

  // Create Transpose_output node
  auto transpose_output = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_output->a(sub);
  transpose_output->perm(transpose_const);
  transpose_output->name("trabnspose_left");

  // Create Conv_2D_Weight_Grad node
  auto conv2d_weight_grad_node = training_graph->nodes()->create<luci::CircleConv2DWeightGrad>();
  conv2d_weight_grad_node->input_grad(transpose_output);
  conv2d_weight_grad_node->input_activation(transpose_input);
  conv2d_weight_grad_node->padding(conv_node->padding());
  auto stride = conv2d_weight_grad_node->stride();
  stride->w(conv_node->stride()->w());
  stride->h(conv_node->stride()->h());
  conv2d_weight_grad_node->name(weight->name() + "_tmp");

  // Create Transpose_output_grad node
  auto transpose_grad = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_grad->a(conv2d_weight_grad_node);
  transpose_grad->perm(transpose_const_2);
  transpose_grad->name(weight->name());

  // Connect output weight gradient with MUL
  weight_gradient_node->from(transpose_grad);

  return std::move(training_graph);
}

#endif

// Backprop for Softmax node
luci::CircleNode *backProp(luci::CircleNode *input_grad_node, luci::CircleNode *after_node, luci::CircleSoftmax *softmax_node, loco::Graph::InputContext *input_context,
                           loco::Graph::OutputContext *output_context)
{
  assert(softmax_node != nullptr);
  assert(after_node != nullptr);
  assert(input_grad_node != nullptr);
  assert(input_context != nullptr);
  assert(output_context != nullptr);

  auto training_graph = input_grad_node->graph();

  // Create Softmax_grad node
  auto softmax_grad_node = training_graph->nodes()->create<luci::CircleSoftmaxGrad>();
  softmax_grad_node->name(input_grad_node->name() + after_node->name());
  softmax_grad_node->softmax_values(after_node);

  // Create Const for Transpose op
  auto transpose_const = training_graph->nodes()->create<luci::CircleConst>();
  transpose_const->shape({2}); // scalar
  transpose_const->dtype(loco::DataType::S32);
  transpose_const->rank(1);
  transpose_const->size<loco::DataType::S32>(2);
  transpose_const->at<loco::DataType::S32>(0) = 1;
  transpose_const->at<loco::DataType::S32>(1) = 0;
  transpose_const->name(softmax_grad_node->name() + "_transpose_const");

  // Create Transpose for input_grad node node
  auto transpose_node = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_node->a(softmax_grad_node);
  transpose_node->perm(transpose_const);
  transpose_node->name(softmax_grad_node->name() + "_transpose");

  auto empty_bias = training_graph->nodes()->create<luci::CircleOutputExclude>();

  // TODO: check transpose (maybe unnecessary)
  auto fc_node = training_graph->nodes()->create<luci::CircleFullyConnected>();
  fc_node->name(transpose_node->name() + input_grad_node->name());
  fc_node->weights(softmax_grad_node);
  fc_node->input(input_grad_node);
  fc_node->bias(empty_bias);
  fc_node->keep_num_dims(true);
  fc_node->fusedActivationFunction(luci::FusedActFunc::NONE);

  return fc_node;
}

// Backprop for Relu node
luci::CircleNode *backPropRelu(luci::CircleNode *input_grad_node, luci::CircleNode *node_with_relu, loco::Graph::InputContext *input_context,
                           loco::Graph::OutputContext *output_context)
{
  assert(node_with_relu != nullptr);
  assert(input_grad_node != nullptr);
  assert(input_context != nullptr);
  assert(output_context != nullptr);

  auto training_graph = input_grad_node->graph();

  // Create Input activation node
  auto input_activation_node = training_graph->nodes()->create<luci::CircleInput>();
  auto input_activation_node_context = input_context->create();
  copy_nodes_params(input_activation_node, node_with_relu);
  link(input_activation_node_context, input_activation_node);
  copy_for_context(input_activation_node_context, input_activation_node);

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
  greater->x(input_activation_node);
  greater->y(greater_const);
  greater->name(input_activation_node->name() + greater_const->name());
  greater->dtype(loco::DataType::BOOL);

  // Create Cast op
  auto cast = training_graph->nodes()->create<luci::CircleCast>();
  cast->x(greater);
  cast->in_data_type(loco::DataType::BOOL);
  cast->out_data_type(loco::DataType::FLOAT32);
  cast->name(greater->name() + "_cast");
  cast->dtype(loco::DataType::FLOAT32);

  // Create Mul node
  auto mul = training_graph->nodes()->create<luci::CircleMul>();
  mul->fusedActivationFunction(luci::FusedActFunc::NONE);
  mul->x(cast);
  mul->y(input_grad_node);
  mul->name(cast->name() + input_grad_node->name());

  return mul;
}

// Backprop for Conv2D node
luci::CircleNode *backProp(luci::CircleNode *input_grad_node, luci::CircleConv2D *conv_node, loco::Graph::InputContext *input_context,
                           loco::Graph::OutputContext *output_context)
{
  assert(conv_node != nullptr);
  assert(input_grad_node != nullptr);
  assert(input_context != nullptr);
  assert(output_context != nullptr);

  auto training_graph = input_grad_node->graph();

  luci::CircleConst *weight = dynamic_cast<luci::CircleConst *>(conv_node->filter());
  assert(weight != nullptr);
  luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(conv_node->bias());
  assert(bias != nullptr);
  luci::CircleNode *conv_from = dynamic_cast<luci::CircleNode *>(conv_node->input());
  assert(conv_from != nullptr);

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

  // Create Input activation node
  auto conv_input_activation_node = training_graph->nodes()->create<luci::CircleInput>();
  auto conv_input_activation_node_context = input_context->create();
  copy_nodes_params(conv_input_activation_node, conv_from);
  link(conv_input_activation_node_context, conv_input_activation_node);
  copy_for_context(conv_input_activation_node_context, conv_input_activation_node);

  // Create Const for Transpose op
  auto transpose_const_before = training_graph->nodes()->create<luci::CircleConst>();
  transpose_const_before->shape({4}); // scalar
  transpose_const_before->dtype(loco::DataType::S32);
  transpose_const_before->rank(1);
  transpose_const_before->size<loco::DataType::S32>(4);
  transpose_const_before->at<loco::DataType::S32>(0) = 0;
  transpose_const_before->at<loco::DataType::S32>(1) = 3;
  transpose_const_before->at<loco::DataType::S32>(2) = 1;
  transpose_const_before->at<loco::DataType::S32>(3) = 2;
  transpose_const_before->name(weight->name() + "BEFORE_TRANSPOSE_CONST");

  // Create Const for Transpose op
  auto transpose_const_after = training_graph->nodes()->create<luci::CircleConst>();
  transpose_const_after->shape({4}); // scalar
  transpose_const_after->dtype(loco::DataType::S32);
  transpose_const_after->rank(1);
  transpose_const_after->size<loco::DataType::S32>(4);
  transpose_const_after->at<loco::DataType::S32>(0) = 0;
  transpose_const_after->at<loco::DataType::S32>(1) = 2;
  transpose_const_after->at<loco::DataType::S32>(2) = 3;
  transpose_const_after->at<loco::DataType::S32>(3) = 1;
  transpose_const_after->name(weight->name() + "AFTER_TRANSPOSE_CONST");

  // Create Const for index
  auto index_const = training_graph->nodes()->create<luci::CircleConst>();
  index_const->shape({3}); // scalar
  index_const->dtype(loco::DataType::S32);
  index_const->rank(1);
  index_const->size<loco::DataType::S32>(3);
  index_const->at<loco::DataType::S32>(0) = 0;
  index_const->at<loco::DataType::S32>(1) = 1;
  index_const->at<loco::DataType::S32>(2) = 2;
  index_const->name("index");

  // Create ReduceSum -> Output of bias for ConvNode
  // Create CircleReduceMax operation
  auto reduce_sum = training_graph->nodes()->create<luci::CircleReduceSum>();
  reduce_sum->input(input_grad_node);
  reduce_sum->reduction_indices(index_const);
  reduce_sum->keep_dims(false);
  reduce_sum->name(bias->name() + "_gradient");

  // Connect bias grad output and sub
  bias_gradient_node->from(reduce_sum);

  // Create Transpose_input node
  auto transpose_activation_node = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_activation_node->a(conv_input_activation_node);
  transpose_activation_node->perm(transpose_const_before);
  transpose_activation_node->name(conv_input_activation_node->name() + "_transpose");

  // Create Transpose_output node
  auto transpose_grad_node = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_grad_node->a(input_grad_node);
  transpose_grad_node->perm(transpose_const_before);
  transpose_grad_node->name(input_grad_node->name() + "_transpose");

  // Create Conv_2D_Weight_Grad node
  auto conv2d_weight_grad_node = training_graph->nodes()->create<luci::CircleConv2DWeightGrad>();
  conv2d_weight_grad_node->input_grad(transpose_grad_node);
  conv2d_weight_grad_node->input_activation(transpose_activation_node);
  conv2d_weight_grad_node->padding(conv_node->padding());
  auto stride = conv2d_weight_grad_node->stride();
  stride->w(conv_node->stride()->w());
  stride->h(conv_node->stride()->h());
  conv2d_weight_grad_node->name(weight->name() + "_tmp");

  // Create Weight Result transpose node
  auto transpose_weight_grad_result = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_weight_grad_result->a(conv2d_weight_grad_node);
  transpose_weight_grad_result->perm(transpose_const_after);
  transpose_weight_grad_result->name(weight->name() + "_gradient");

  // Connect output weight gradient with MUL
  weight_gradient_node->from(transpose_weight_grad_result);

  // Create output grad node
  // Create weigth_2 const
  auto weight_const_node = training_graph->nodes()->create<luci::CircleConst>();
  copy_nodes_params(weight_const_node, weight);
  weight_const_node->size<loco::DataType::FLOAT32>(0);
  weight_const_node->shape_status(luci::ShapeStatus::VALID);

  // Create Transpose for weight conv node
  auto transpose_weight_input = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_weight_input->a(weight_const_node);
  transpose_weight_input->perm(transpose_const_before);
  transpose_weight_input->name(weight_const_node->name() + "_transpose");

  // Create Conv Grad Input
  auto result_output_grad = training_graph->nodes()->create<luci::CircleConv2DInputGrad>();
  result_output_grad->input_grad(transpose_grad_node);
  result_output_grad->weight(transpose_weight_input);
  result_output_grad->padding(conv_node->padding());
  auto stride_2 = conv2d_weight_grad_node->stride();
  stride_2->w(conv_node->stride()->w());
  stride_2->h(conv_node->stride()->h());
  result_output_grad->name(transpose_grad_node->name() + transpose_weight_input->name());

  // Create Output Grad Result transpose node
  auto transpose_output_grad_result = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_output_grad_result->a(conv2d_weight_grad_node);
  transpose_output_grad_result->perm(transpose_const_after);
  transpose_output_grad_result->name(result_output_grad->name() + transpose_const_after->name());

  return transpose_output_grad_result;
}

// Backprop for MaxPool node
luci::CircleNode *backProp(luci::CircleNode *input_grad_node, luci::CircleMaxPool2D *max_pool_node, loco::Graph::InputContext *input_context,
                           loco::Graph::OutputContext *output_context)
{
  assert(max_pool_node != nullptr);
  assert(input_grad_node != nullptr);
  assert(input_context != nullptr);
  assert(output_context != nullptr);

  luci::CircleNode *max_pool_from = dynamic_cast<luci::CircleNode *>(max_pool_node->value());

  // Create Max pool input activation node
  auto max_pool_input_activation_node = input_grad_node->graph()->nodes()->create<luci::CircleInput>();
  auto max_pool_input_activation_node_context = input_context->create();
  copy_nodes_params(max_pool_input_activation_node, max_pool_from);
  link(max_pool_input_activation_node_context, max_pool_input_activation_node);
  copy_for_context(max_pool_input_activation_node_context, max_pool_input_activation_node);

  luci::CircleMaxPool2DGrad *max_pool_grad_node = input_grad_node->graph()->nodes()->create<luci::CircleMaxPool2DGrad>();
  max_pool_grad_node->input_grad(input_grad_node);
  max_pool_grad_node->input_activations(max_pool_input_activation_node);
  max_pool_grad_node->name(input_grad_node->name() + max_pool_node->name());
  max_pool_grad_node->fusedActivationFunction(max_pool_node->fusedActivationFunction());
  max_pool_grad_node->padding(max_pool_node->padding());
  auto filter = max_pool_grad_node->filter();
  filter->w(max_pool_node->filter()->w());
  filter->h(max_pool_node->filter()->h());
  auto stride = max_pool_grad_node->stride();
  stride->w(max_pool_node->stride()->w());
  stride->h(max_pool_node->stride()->h());

  return max_pool_grad_node;
}

// Backprop for Reshape node
luci::CircleNode *backProp(luci::CircleNode *input_grad_node, luci::CircleReshape *reshape_node, loco::Graph::InputContext *input_context,
                           loco::Graph::OutputContext *output_context)
{
  assert(reshape_node != nullptr);
  assert(input_grad_node != nullptr);
  assert(input_context != nullptr);
  assert(output_context != nullptr);

  luci::CircleNode *input_node = dynamic_cast<luci::CircleNode *>(reshape_node->tensor());
  luci::CircleConst *reshape_const = input_grad_node->graph()->nodes()->create<luci::CircleConst>();
  reshape_const->shape({input_node->rank()});
  reshape_const->dtype(loco::DataType::S32);
  reshape_const->rank(1);
  reshape_const->size<loco::DataType::S32>(input_node->rank());
  reshape_const->at<loco::DataType::S32>(0) = 1;
  for (int i = 1; i < input_node->rank(); ++i)
  {
    reshape_const->at<loco::DataType::S32>(i) = input_node->dim(i).value();
  }
  reshape_const->name(input_grad_node->name() + reshape_node->name() + "_const");

  luci::CircleReshape *new_reshape_node = input_grad_node->graph()->nodes()->create<luci::CircleReshape>();
  new_reshape_node->tensor(input_grad_node);
  new_reshape_node->shape(reshape_const);
  new_reshape_node->name(input_grad_node->name() + reshape_node->name());

  return new_reshape_node;
}

// Backprop for FullyConnected node
luci::CircleNode *backProp(luci::CircleNode *input_grad_node, luci::CircleFullyConnected *fc_node, loco::Graph::InputContext *input_context,
                           loco::Graph::OutputContext *output_context)
{
  assert(fc_node != nullptr);

  assert(fc_node->fusedActivationFunction() == luci::FusedActFunc::NONE);

  luci::CircleConst *weight = dynamic_cast<luci::CircleConst *>(fc_node->weights());
  assert(weight != nullptr);
  luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(fc_node->bias());
  assert(bias != nullptr);
  luci::CircleNode *fc_from = dynamic_cast<luci::CircleNode *>(fc_node->input());
  assert(fc_from != nullptr);

  auto *training_graph = input_grad_node->graph();

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

  // change name input grad node due to bias gradient
  input_grad_node->name(bias->name() + "_gradient");

  // Connect bias grad output and sub
  bias_gradient_node->from(input_grad_node);

  // Create Fc_InputActivation node
  auto fc_input_activation_node = training_graph->nodes()->create<luci::CircleInput>();
  auto fc_input_activation_node_context = input_context->create();
  copy_nodes_params(fc_input_activation_node, fc_from);
  link(fc_input_activation_node_context, fc_input_activation_node);
  copy_for_context(fc_input_activation_node_context, fc_input_activation_node);

  // Create Const for Transpose op
  auto transpose_const = training_graph->nodes()->create<luci::CircleConst>();
  transpose_const->shape({2}); // scalar
  transpose_const->dtype(loco::DataType::S32);
  transpose_const->rank(1);
  transpose_const->size<loco::DataType::S32>(2);
  transpose_const->at<loco::DataType::S32>(0) = 1;
  transpose_const->at<loco::DataType::S32>(1) = 0;
  transpose_const->name(input_grad_node->name() + "_transpose_const");

  // Create Transpose for input_grad node node
  auto transpose_grad_node = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_grad_node->a(input_grad_node);
  transpose_grad_node->perm(transpose_const);
  transpose_grad_node->name(input_grad_node->name() + "_transpose");

  // Create Mul node
  auto mul = training_graph->nodes()->create<luci::CircleMul>();
  mul->fusedActivationFunction(luci::FusedActFunc::NONE);
  mul->x(transpose_grad_node);
  mul->y(fc_input_activation_node);
  mul->name(weight->name() + "_gradient");

  // Connect output weight gradient with MUL_LAMBDA
  weight_gradient_node->from(mul);

  // Create output gradient
  // Create Fc_Weight_2 node as const
  auto fc_w_const = training_graph->nodes()->create<luci::CircleConst>();
  copy_nodes_params(fc_w_const, weight);
  fc_w_const->size<loco::DataType::FLOAT32>(0);
  fc_w_const->shape_status(luci::ShapeStatus::VALID);

  // Create Transpose for input_grad node node
  auto transpose_output_node = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_output_node->a(fc_w_const);
  transpose_output_node->perm(transpose_const);
  transpose_output_node->name(fc_w_const->name() + "_transpose");

  // Create output node
  auto empty_bias = training_graph->nodes()->create<luci::CircleOutputExclude>();
  auto fc_new_node = training_graph->nodes()->create<luci::CircleFullyConnected>();
  fc_new_node->name(input_grad_node->name() + transpose_output_node->name());
  fc_new_node->weights(transpose_output_node);
  fc_new_node->input(input_grad_node);
  fc_new_node->bias(empty_bias);
  fc_new_node->keep_num_dims(true);
  fc_new_node->fusedActivationFunction(luci::FusedActFunc::NONE);

  return fc_new_node;
}

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
  loco::Graph::OutputContext *output_context = training_graph->outputs();

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

  // it is for MSE
//  // Create Sub node
//  auto sub = training_graph->nodes()->create<luci::CircleSub>();
//  sub->fusedActivationFunction(luci::FusedActFunc::NONE);
//  sub->x(predicted_input_values_node);
//  sub->y(target_input_values_node);
//  sub->name("_sub");

  // Create Div for Cross Entropy
  auto grad_error = training_graph->nodes()->create<luci::CircleDiv>();
  grad_error->fusedActivationFunction(luci::FusedActFunc::NONE);
  grad_error->x(target_input_values_node);
  grad_error->y(predicted_input_values_node);
  grad_error->name("corss_etropy_error");

  // Create Neg
  auto neg = training_graph->nodes()->create<luci::CircleNeg>();
  neg->name("neg");
  neg->x(grad_error);

//  // Create Const for index
//  auto neg_const = training_graph->nodes()->create<luci::CircleConst>();
//  index_const->shape({3}); // scalar
//  index_const->dtype(loco::DataType::S32);
//  index_const->rank(1);
//  index_const->size<loco::DataType::S32>(3);
//  index_const->at<loco::DataType::S32>(0) = 0;
//  index_const->at<loco::DataType::S32>(1) = 1;
//  index_const->at<loco::DataType::S32>(2) = 2;
//  index_const->name("index");


  std::stack<luci::CircleNode *> nodes;

  luci::CircleNode *cur_input_grad_node = neg;

  for (auto node : postorder_traversal(loco::output_nodes(original_graph)))
  {
    auto circle_node = dynamic_cast<luci::CircleNode *>(node);
    assert(circle_node != nullptr);
    nodes.push(circle_node);
  }

  while (nodes.empty() == false)
  {
    luci::CircleNode *circle_node = nodes.top();
    nodes.pop();
    if (auto output_node = dynamic_cast<luci::CircleOutput *>(circle_node))
    {
      // Do nothing
      continue;
    } else if (auto const_node = dynamic_cast<luci::CircleConst *>(circle_node))
    {
      // Do nothing
      continue;
    } else if (auto softmax_node = dynamic_cast<luci::CircleSoftmax *>(circle_node))
    {
      // TODO: support not only last softmax node
      cur_input_grad_node = backProp(cur_input_grad_node, predicted_input_values_node, softmax_node, input_context, output_context);
    } else if (auto fc_node = dynamic_cast<luci::CircleFullyConnected *>(circle_node))
    {
      if (fc_node->fusedActivationFunction() == luci::FusedActFunc::RELU)
      {
        cur_input_grad_node =
          backPropRelu(cur_input_grad_node, fc_node, input_context, output_context);
      }
      else
      {
        assert(fc_node->fusedActivationFunction() == luci::FusedActFunc::NONE);
      }

      cur_input_grad_node = backProp(cur_input_grad_node, fc_node, input_context, output_context);
    } else if (auto reshape_node = dynamic_cast<luci::CircleReshape *>(circle_node))
    {
      cur_input_grad_node = backProp(cur_input_grad_node, reshape_node, input_context, output_context);
    } else if  (auto max_pool_node = dynamic_cast<luci::CircleMaxPool2D *>(circle_node))
    {
      cur_input_grad_node = backProp(cur_input_grad_node, max_pool_node, input_context, output_context);
    }
    else if  (auto expand_dims = dynamic_cast<luci::CircleExpandDims *>(circle_node))
    {
      continue;
    }
    else if  (auto conv_2d = dynamic_cast<luci::CircleConv2D *>(circle_node))
    {
      if (conv_2d->fusedActivationFunction() == luci::FusedActFunc::RELU)
      {
        cur_input_grad_node =
          backPropRelu(cur_input_grad_node, conv_2d, input_context, output_context);
      }
      else
      {
        assert(conv_2d->fusedActivationFunction() == luci::FusedActFunc::NONE);
      }
      cur_input_grad_node = backProp(cur_input_grad_node, conv_2d, input_context, output_context);
      break; //tmp
    }
    else
    {
      break;
    }

  }

  return std::move(training_graph);
}

#if 0
// Let's create backpropogation calculation node:
// For this example (MSE error, SGD, lambda=L,  conv2d->conv2d node with weight and bias) it is:
//       PRED    TARGET
//           \       /
//              SUB                       Conv2D_2_InputActivation
//           /      \                     /
//          /   Transpose(0,3,1,2)    Transpose (0,3,1,2)
//         /          |  \                  /
//     ReduceMean     | Conv2D_Grad_Weight
//     (indx = 0,1,2) |        |
//          |         |     Transpose(0,2,3,1)
//        Bias_2_grad |        |
//                    |    Weight_2_grad
//   -----------------
//   |
//   |                             Conv2D_weights
//   |                                |
//   |                             Transpose(0, 3, 1, 2)
//   |                                |
//    ----------------------- Conv2D_Grad_Input        Conv2D_1_InputActivation
//                            |        |                /
//                            |        |             Transpose (0,3,1,2)
//                            |         \            /
//                            |       Conv2D_Grad_Weight
//                            |                 |
//                      ReduceMean        Transpose(0,2,3,1)
//                           |                  |
//                        Bias_grad       Weight_grad_2
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

  luci::CircleConv2D *conv_node_1 = nullptr;
  luci::CircleConv2D *conv_node_2 = nullptr;

  for (auto node : postorder_traversal(loco::output_nodes(original_graph)))
  {
    if (dynamic_cast<luci::CircleConv2D *>(node) != nullptr and conv_node_1 == nullptr)
    {
      conv_node_1 = dynamic_cast<luci::CircleConv2D *>(node);
    } else if (dynamic_cast<luci::CircleConv2D *>(node) != nullptr and conv_node_1 != nullptr)
    {
      conv_node_2 = dynamic_cast<luci::CircleConv2D *>(node);
    }
  }

  assert(conv_node_1 != nullptr);
  assert(conv_node_2 != nullptr);

  luci::CircleConst *weight_1 = dynamic_cast<luci::CircleConst *>(conv_node_1->filter());
  assert(weight_1 != nullptr);
  luci::CircleConst *bias_1 = dynamic_cast<luci::CircleConst *>(conv_node_1->bias());
  assert(bias_1 != nullptr);

  luci::CircleConst *weight_2 = dynamic_cast<luci::CircleConst *>(conv_node_2->filter());
  assert(weight_2 != nullptr);
  luci::CircleConst *bias_2 = dynamic_cast<luci::CircleConst *>(conv_node_2->bias());
  assert(bias_2 != nullptr);

  luci::CircleNode *conv_from = dynamic_cast<luci::CircleNode *>(conv_node_1->input());
  assert(conv_from != nullptr);

  // Create outputs: result gradients for weights and biases
  loco::Graph::OutputContext *output_context = training_graph->outputs();

  // Create weight gradient 1 output
  auto weight_gradient_context_1 = output_context->create();
  auto weight_gradient_node_1 = training_graph->nodes()->create<luci::CircleOutput>();
  copy_nodes_params(weight_gradient_node_1, weight_1);
  link(weight_gradient_context_1, weight_gradient_node_1);
  copy_for_context(weight_gradient_context_1, weight_gradient_node_1);

  // Create weight gradient 2 output
  auto weight_gradient_context_2 = output_context->create();
  auto weight_gradient_node_2 = training_graph->nodes()->create<luci::CircleOutput>();
  copy_nodes_params(weight_gradient_node_2, weight_2);
  link(weight_gradient_context_2, weight_gradient_node_2);
  copy_for_context(weight_gradient_context_2, weight_gradient_node_2);

  // Create bias gradient 1 output
  auto bias_gradient_context_1 = output_context->create();
  auto bias_gradient_node_1 = training_graph->nodes()->create<luci::CircleOutput>();
  copy_nodes_params(bias_gradient_node_1, bias_1);
  link(bias_gradient_context_1, bias_gradient_node_1);
  copy_for_context(bias_gradient_context_1, bias_gradient_node_1);

  // Create bias gradient 2 output
  auto bias_gradient_context_2 = output_context->create();
  auto bias_gradient_node_2 = training_graph->nodes()->create<luci::CircleOutput>();
  copy_nodes_params(bias_gradient_node_2, bias_2);
  link(bias_gradient_context_2, bias_gradient_node_2);
  copy_for_context(bias_gradient_context_2, bias_gradient_node_2);

  // Create Sub node
  auto sub = training_graph->nodes()->create<luci::CircleSub>();
  sub->fusedActivationFunction(luci::FusedActFunc::NONE);
  sub->x(predicted_input_values_node);
  sub->y(target_input_values_node);
  sub->name("_sub");

  // Create Const for transpose_1 and transpose_2
  auto transpose_const = training_graph->nodes()->create<luci::CircleConst>();
  transpose_const->shape({4}); // scalar
  transpose_const->dtype(loco::DataType::S32);
  transpose_const->rank(1);
  transpose_const->size<loco::DataType::S32>(4);
  transpose_const->at<loco::DataType::S32>(0) = 0;
  transpose_const->at<loco::DataType::S32>(1) = 3;
  transpose_const->at<loco::DataType::S32>(2) = 1;
  transpose_const->at<loco::DataType::S32>(3) = 2;
  transpose_const->name("TRANSPOSE_CONST");

  // Create Const for transpose_3
  auto transpose_const_2 = training_graph->nodes()->create<luci::CircleConst>();
  transpose_const_2->shape({4}); // scalar
  transpose_const_2->dtype(loco::DataType::S32);
  transpose_const_2->rank(1);
  transpose_const_2->size<loco::DataType::S32>(4);
  transpose_const_2->at<loco::DataType::S32>(0) = 0;
  transpose_const_2->at<loco::DataType::S32>(1) = 2;
  transpose_const_2->at<loco::DataType::S32>(2) = 3;
  transpose_const_2->at<loco::DataType::S32>(3) = 1;
  transpose_const_2->name("TRANSPOSE_CONST");


  // Create Const for index
  auto index_const = training_graph->nodes()->create<luci::CircleConst>();
  index_const->shape({3}); // scalar
  index_const->dtype(loco::DataType::S32);
  index_const->rank(1);
  index_const->size<loco::DataType::S32>(3);
  index_const->at<loco::DataType::S32>(0) = 0;
  index_const->at<loco::DataType::S32>(1) = 1;
  index_const->at<loco::DataType::S32>(2) = 2;
  index_const->name("index");

  // Create ReduceSum
  // Create CircleReduceMax operation
  auto reduce_sum = training_graph->nodes()->create<luci::CircleReduceSum>();
  reduce_sum->input(sub);
  reduce_sum->reduction_indices(index_const);
  reduce_sum->keep_dims(false);
  reduce_sum->name(bias_gradient_node_2->name());

  // Connect bias grad output and sub
  bias_gradient_node_2->from(reduce_sum);

  // Create conv_InputActivation node
  auto conv_input_activation_node = training_graph->nodes()->create<luci::CircleInput>();
  auto conv_input_activation_node_context = input_context->create();
  copy_nodes_params(conv_input_activation_node, conv_node_1);
  link(conv_input_activation_node_context, conv_input_activation_node);
  copy_for_context(conv_input_activation_node_context, conv_input_activation_node);

  // Create Transpose_input node
  auto transpose_input = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_input->a(conv_input_activation_node);
  transpose_input->perm(transpose_const);
  transpose_input->name("trabnspose_left");

  // Create Transpose_output node
  auto transpose_output = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_output->a(sub);
  transpose_output->perm(transpose_const);
  transpose_output->name("trabnspose_left");

  // Create Conv_2D_Weight_Grad node
  auto conv2d_weight_grad_node = training_graph->nodes()->create<luci::CircleConv2DWeightGrad>();
  conv2d_weight_grad_node->input_grad(transpose_output);
  conv2d_weight_grad_node->input_activation(transpose_input);
  conv2d_weight_grad_node->padding(conv_node_2->padding());
  auto stride = conv2d_weight_grad_node->stride();
  stride->w(conv_node_2->stride()->w());
  stride->h(conv_node_2->stride()->h());
  auto kernel_size = conv2d_weight_grad_node->kernel_size();
  kernel_size->h(weight_2->dim(1).value());
  kernel_size->w(weight_2->dim(2).value());
  conv2d_weight_grad_node->name(weight_2->name() + "_tmp");

  // Create Transpose_output_grad node
  auto transpose_grad = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_grad->a(conv2d_weight_grad_node);
  transpose_grad->perm(transpose_const_2);
  transpose_grad->name(weight_2->name() + "_gradient");

  // Connect output weight gradient with MUL
  weight_gradient_node_2->from(transpose_grad);

  // Create weigth_2 const
  auto conv_weight_2_const = training_graph->nodes()->create<luci::CircleConst>();
  copy_nodes_params(conv_weight_2_const, weight_2);
  conv_weight_2_const->size<loco::DataType::FLOAT32>(0);
  conv_weight_2_const->shape_status(luci::ShapeStatus::VALID);

  // Create Transpose_input node
  auto transpose_weight_input = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_weight_input->a(conv_weight_2_const);
  transpose_weight_input->perm(transpose_const);
  transpose_weight_input->name("trabnspose_weight_input");

  // Create Conv Grad Input
  auto conv2d_input_grad_node = training_graph->nodes()->create<luci::CircleConv2DInputGrad>();
  conv2d_input_grad_node->input_grad(transpose_output);
  conv2d_input_grad_node->weight(transpose_weight_input);
  conv2d_input_grad_node->padding(conv_node_2->padding());
  auto stride_2 = conv2d_weight_grad_node->stride();
  stride_2->w(conv_node_2->stride()->w());
  stride_2->h(conv_node_2->stride()->h());
  conv2d_input_grad_node->name(weight_2->name() + "_grad_input");

  // Work with first conv
  //create transose for bias grad
  auto transpose_bias_grad = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_bias_grad->a(conv2d_input_grad_node);
  transpose_bias_grad->perm(transpose_const_2);
  transpose_bias_grad->name("trabnspose_weight_input");

  // Create ReduceSum
  auto reduce_sum_1 = training_graph->nodes()->create<luci::CircleReduceSum>();
  reduce_sum_1->input(transpose_bias_grad);
  reduce_sum_1->reduction_indices(index_const);
  reduce_sum_1->keep_dims(false);
  reduce_sum_1->name(bias_gradient_node_1->name());

  // Connect bias grad output and sub
  bias_gradient_node_1->from(reduce_sum_1);

  // Create conv_InputActivation_1 node
  auto conv_input_activation_node_1 = training_graph->nodes()->create<luci::CircleInput>();
  auto conv_input_activation_node_context_1 = input_context->create();
  copy_nodes_params(conv_input_activation_node_1, conv_from);
  link(conv_input_activation_node_context_1, conv_input_activation_node_1);
  copy_for_context(conv_input_activation_node_context_1, conv_input_activation_node_1);

  // Create Transpose_input node
  auto transpose_input_1 = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_input_1->a(conv_input_activation_node_1);
  transpose_input_1->perm(transpose_const);
  transpose_input_1->name("trabnspose_left_1");

  // Create Conv_2D_Weight_Grad_1 node
  auto conv2d_weight_grad_node_1 = training_graph->nodes()->create<luci::CircleConv2DWeightGrad>();
  conv2d_weight_grad_node_1->input_grad(conv2d_input_grad_node);
  conv2d_weight_grad_node_1->input_activation(transpose_input_1);
  conv2d_weight_grad_node_1->padding(conv_node_1->padding());
  auto stride_1 = conv2d_weight_grad_node_1->stride();
  stride_1->w(conv_node_1->stride()->w());
  stride_1->h(conv_node_1->stride()->h());
  auto kernel_size_1 = conv2d_weight_grad_node_1->kernel_size();
  kernel_size_1->h(weight_1->dim(1).value());
  kernel_size_1->w(weight_1->dim(2).value());
  conv2d_weight_grad_node_1->name(weight_1->name() + "_tmp");

  // Create Transpose_output_grad node
  auto transpose_grad_1 = training_graph->nodes()->create<luci::CircleTranspose>();
  transpose_grad_1->a(conv2d_weight_grad_node_1);
  transpose_grad_1->perm(transpose_const_2);
  transpose_grad_1->name(weight_1->name());

  // Connect output weight gradient with MUL
  weight_gradient_node_1->from(transpose_grad_1);

  return std::move(training_graph);
}

#endif