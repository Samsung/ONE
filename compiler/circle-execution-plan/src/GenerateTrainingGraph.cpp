/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

std::map<std::basic_string<char>, luci::CircleInput *> inputsNames(loco::Graph *graph)
{
  std::map<std::basic_string<char>, luci::CircleInput *> result;

  auto inputs = input_nodes(graph);
  for (const auto &node : inputs)
  {
    auto circle_node = dynamic_cast<luci::CircleInput *>(node);
    if (circle_node == nullptr)
      continue;

    result[circle_node->name()] = circle_node;
  }

  return result;
}

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
  fc_node->weights(transpose_node);
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

  auto node_name = node_with_relu->name();

  luci::CircleInput *input_activation_node = nullptr;

  auto inputs_names = inputsNames(training_graph);

  if (inputs_names.find(node_name) != inputs_names.end())
  {
    input_activation_node = inputs_names[node_name];
  } else
  {
    // Create Input activation node
    input_activation_node = training_graph->nodes()->create<luci::CircleInput>();
    auto input_activation_node_context = input_context->create();
    copy_nodes_params(input_activation_node, node_with_relu);
    link(input_activation_node_context, input_activation_node);
    copy_for_context(input_activation_node_context, input_activation_node);
  }

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
  conv2d_weight_grad_node->kernel_size()->h(weight->dim(1).value());
  conv2d_weight_grad_node->kernel_size()->w(weight->dim(2).value());

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
  transpose_output_grad_result->a(result_output_grad);
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
  //assert(bias != nullptr);
  luci::CircleNode *fc_from = dynamic_cast<luci::CircleNode *>(fc_node->input());
  assert(fc_from != nullptr);

  auto *training_graph = input_grad_node->graph();

  // Create weight gradient output
  auto weight_gradient_context = output_context->create();
  auto weight_gradient_node = training_graph->nodes()->create<luci::CircleOutput>();
  copy_nodes_params(weight_gradient_node, weight);
  link(weight_gradient_context, weight_gradient_node);
  copy_for_context(weight_gradient_context, weight_gradient_node);

  if (bias != nullptr)
  {
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
  }

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

luci::CircleNode *createCrossEntropyError(luci::CircleNode *target_node, luci::CircleNode *predicted_node)
{
  auto *training_graph = target_node->graph();

  // Create Div for Cross Entropy
  auto grad_error = training_graph->nodes()->create<luci::CircleDiv>();
  grad_error->fusedActivationFunction(luci::FusedActFunc::NONE);
  grad_error->x(target_node);
  grad_error->y(predicted_node);
  grad_error->name("cross_entropy_error");

  // Create Neg
  auto neg = training_graph->nodes()->create<luci::CircleNeg>();
  neg->name("cross_entropy_error_neg");
  neg->x(grad_error);

  return neg;
}

std::unique_ptr<loco::Graph> training_graph::GenerateTrainingGraph::createTrainingGraph(const std::vector<uint32_t> &nodes_ind)
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

  std::stack<luci::CircleNode *> nodes;

  // TODO: add switch to choose error from input args
  // Note: now its just CROSS_ENTROPY_ERROR
  luci::CircleNode *cur_input_grad_node = createCrossEntropyError(target_input_values_node, predicted_input_values_node);

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

    // For checking user defined backprop nodes range
    try
    {
      auto node_id = luci::get_node_id(circle_node);
      if (std::find(nodes_ind.begin(), nodes_ind.end(), node_id) == nodes_ind.end())
        continue;
    } catch (const std::runtime_error &)
    {
      continue;
    }

    switch (circle_node->opcode())
    {
      case luci::CircleOpcode::CIRCLEOUTPUT:
      case luci::CircleOpcode::CIRCLECONST:
      case luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE:
      case luci::CircleOpcode::EXPAND_DIMS:
        continue;
      case luci::CircleOpcode::SOFTMAX:
      {
        auto softmax_node = must_cast<luci::CircleSoftmax *>(circle_node);
        cur_input_grad_node = backProp(cur_input_grad_node, predicted_input_values_node, softmax_node, input_context, output_context);
        break;
      }
      case luci::CircleOpcode::RESHAPE:
      {
        auto reshape_node = must_cast<luci::CircleReshape *>(circle_node);
        cur_input_grad_node = backProp(cur_input_grad_node, reshape_node, input_context, output_context);
        break;
      }
      case luci::CircleOpcode::MAX_POOL_2D:
      {
        auto max_pool_node = must_cast<luci::CircleMaxPool2D *>(circle_node);
        cur_input_grad_node = backProp(cur_input_grad_node, max_pool_node, input_context, output_context);
        break;
      }
      case luci::CircleOpcode::FULLY_CONNECTED:
      {
        auto fc_node = must_cast<luci::CircleFullyConnected *>(circle_node);
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
        break;
      }
      case luci::CircleOpcode::CONV_2D:
      {
        auto conv_2d = must_cast<luci::CircleConv2D *>(circle_node);
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
        break;
      }
      default:
      {
        std::cout << "New node type";
        break;
      }
    }
  }

  return std::move(training_graph);
}
