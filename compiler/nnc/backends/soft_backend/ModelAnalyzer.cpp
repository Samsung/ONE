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

#include "ModelAnalyzer.h"

#include "mir/Shape.h"
#include "mir/Graph.h"
#include "mir/OpDefs.h"

#include <stack>
#include <map>

using namespace std;

namespace nnc
{

using namespace mir;
using namespace sir;

void ModelAnalyzer::appendOperationToInference(Operation *op, const string &function_name,
                                               std::vector<size_t> aux_args)
{

  vector<size_t> node_output_tensors;

  // process operation outputs
  if (op->getType() == Operation::Type::input)
  {
    // register input tensor
    const string &tensor_name = op->getOutput(0)->getName();
    const auto tensor_id = declareInputTensor(tensor_name, op->getOutputShape(0));
    node_output_tensors.push_back(tensor_id);
  }
  else if (op->getType() == Operation::Type::constant)
  {
    // register constant tensor
    // it's data is deserialized to described tensor by O(1) at runtime
    const auto tensor_id = declareTemporaryTensor();
    node_output_tensors.push_back(tensor_id);
  }
  else if (op->getType() == Operation::Type::output)
  {
    assert(!op->getInput(0)->getName().empty());
  }
  else
  {
    for (const auto &output : op->getOutputs())
    {
      const auto &tensor_name = output.getName();
      const auto tensor_id =
        tensor_name.empty() ? declareTemporaryTensor() : declarePersistentTensor(tensor_name);
      node_output_tensors.push_back(tensor_id);
    }
  }

  // process operation inputs
  vector<size_t> node_input_tensors;
  for (const Operation::Output *input : op->getInputs())
  {
    size_t idx = input->getIndex();
    const Operation *prev_op = input->getNode();
    assert(_opToDescr.find(prev_op) != _opToDescr.end());
    auto call = dynamic_cast<const CallFunction *>(_opToDescr[prev_op]);
    assert(call);
    const size_t &in_tensor_id = call->outputs[idx];
    node_input_tensors.push_back(in_tensor_id);
  }

  std::copy(aux_args.begin(), aux_args.end(), std::back_inserter(node_input_tensors));
  unique_ptr<Action> operation_call(new CallFunction(
    op, function_name, std::move(node_input_tensors), std::move(node_output_tensors)));
  _inferenceSequence.push_back(std::move(operation_call));
  _opToDescr[op] = _inferenceSequence.back().get();
}

void ModelAnalyzer::updateMaxTemporarySize(const size_t size)
{
  _max_temp_size = std::max(_max_temp_size, size);
}

size_t ModelAnalyzer::declareInputTensor(const std::string &name, const mir::Shape &shape)
{
  assert(!name.empty() && "Input tensor must have name");
  size_t id = _allocatedTensors++;
  _tensors.push_back({id, TensorDescriptor::Type::input, name, shape});
  _inputs.push_back(id);
  return id;
}

size_t ModelAnalyzer::declarePersistentTensor(const std::string &name)
{
  assert(!name.empty());
  size_t id = _allocatedTensors++;
  _tensors.push_back({id, TensorDescriptor::Type::persistent, name, {}});
  _persistent_tensors.push_back(id);
  return id;
}

size_t ModelAnalyzer::declareTemporaryTensor()
{
  size_t id = _allocatedTensors++;
  _tensors.push_back({id, TensorDescriptor::Type::temporary, "", {}});
  return id;
}

void ModelAnalyzer::gatherDefUseInfo(const vector<unique_ptr<Action>> &post_order,
                                     map<size_t, size_t> &first_def, map<size_t, size_t> &last_use)
{

  for (size_t pos = 0; pos < post_order.size(); ++pos)
  {
    const unique_ptr<Action> &action = post_order[pos];
    const CallFunction *call = dynamic_cast<CallFunction *>(action.get());
    assert(call);

    // update def info
    for (size_t output_tensor_id : call->outputs)
    {
      const TensorDescriptor &td = _tensors[output_tensor_id];
      if (td.type != TensorDescriptor::Type::temporary)
        continue;

      if (!first_def.count(output_tensor_id))
        first_def[output_tensor_id] = pos;
    }

    // update usage info
    for (size_t input_tensor_id : call->inputs)
    {
      const TensorDescriptor &td = _tensors[input_tensor_id];
      if (td.type != TensorDescriptor::Type::temporary)
        continue;

      last_use[input_tensor_id] = pos;
    }
  }
}

void ModelAnalyzer::constructInferenceSequence(const vector<Operation *> &post_order)
{
  // Run inference sequence construction over constructed list of operations
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it)
  {
    Operation *node = *it;
    node->accept(this);
  }

  // Insert temporary tensor constructors
  // map temporary tensor id to index in original sequence where it was defined/used first/last time
  map<size_t, size_t> first_def;
  map<size_t, size_t> last_use;

  // prepare use-def info
  gatherDefUseInfo(_inferenceSequence, first_def, last_use);

  // insert memory operations
  // Every iteration of loop contains three steps:
  // 1) insert constructors of temporary tensors used in current operations
  //    and not used in inference sequence before
  // 2) insert operation call
  // 3) insert destructors of temporary tensors unused after current operation
  std::vector<unique_ptr<Action>> old_inference_seq;
  old_inference_seq.swap(_inferenceSequence);
  _inferenceSequence.reserve(old_inference_seq.size());

  for (size_t pos = 0; pos < old_inference_seq.size(); ++pos)
  {
    unique_ptr<Action> &action = old_inference_seq[pos];
    const CallFunction *call = dynamic_cast<CallFunction *>(action.get());
    assert(call);

    // construct required temporary tensors
    for (size_t output_tensor_id : call->outputs)
    {
      const TensorDescriptor &td = _tensors[output_tensor_id];
      assert(td.id == output_tensor_id);
      if (td.type != TensorDescriptor::Type::temporary)
        continue;

      if (first_def[output_tensor_id] == pos)
      {
        unique_ptr<Action> tmp_constructor(new CreateTmp(output_tensor_id));
        _inferenceSequence.push_back(std::move(tmp_constructor));
      }
    }

    // Insert operation call
    _inferenceSequence.push_back(std::move(action));

    // destroy unused temporary tensors
    for (size_t input_tensor_id : call->inputs)
    {
      const TensorDescriptor &td = _tensors[input_tensor_id];
      assert(td.id == input_tensor_id);
      if (td.type != TensorDescriptor::Type::temporary)
        continue;

      if (last_use[input_tensor_id] == pos)
      {
        unique_ptr<Action> tmp_destructor(new DestroyTmp(input_tensor_id));
        _inferenceSequence.push_back(std::move(tmp_destructor));
      }
    }
  }
}

void ModelAnalyzer::collectOutputs(const mir::Graph *g)
{
  for (ops::OutputOp *out_op : g->getOutputs())
  {
    auto op_call = dynamic_cast<const CallFunction *>(_opToDescr[out_op]);
    assert(op_call->inputs.size() == 1);
    _outputs.push_back(op_call->inputs[0]);
  }
}

void ModelAnalyzer::analyze(const mir::Graph *g)
{
  // Current path through graph
  stack<pair<Operation *, size_t>> s;
  // Nodes in Reverse Post Order stored by DFS
  vector<Operation *> post_order;
  // Set contains pointer to node if it is visited by DFS
  set<Operation *> visited;

  vector<Operation *> init_ops;
  for (Operation *op : g->getNodes())
  {
    if (op->getNumInputs() == 0)
    {
      init_ops.emplace_back(op);
    }
  }

  // Register temporary tensor for im2col buffer
  _temp_tensor_id = declareTemporaryTensor();

  // Walk all network inputs
  for (Operation *in : init_ops)
  {
    if (!visited.count(in))
    {
      visited.insert(in);
      s.push({in, 0});
    }

    // main DFS loop
    while (!s.empty())
    {
      // top stores current node and current outgoing edge from it
      auto &top = s.top();
      Operation *node = top.first;
      auto edge = top.second++;
      // FIXME Refactor me.
      std::vector<Operation *> next_nodes;
      for (const auto &out : node->getOutputs())
      {
        const auto &uses = out.getUses();
        std::transform(uses.begin(), uses.end(), std::back_inserter(next_nodes),
                       [](Operation::Use use) { return use.getNode(); });
      }
      if (edge == next_nodes.size())
      {
        // this node is fully analyzed, push it into RPO and pop from stack
        post_order.push_back(node);
        s.pop();
      }
      else
      {
        // Search current outgoing edge
        Operation *successor = next_nodes[edge];
        if (!visited.count(successor))
        {
          visited.insert(successor);
          s.push({next_nodes[edge], 0});
        }
      }
    }
  }

  constructInferenceSequence(post_order);

  collectOutputs(g);
}

void ModelAnalyzer::visit(ops::ConcatOp &op) { appendOperationToInference(&op, "concat"); }

void ModelAnalyzer::visit(ops::Conv2DOp &op)
{
  assert(op.getNumGroups() == 1);
  const auto &kernel_shape = op.getInputShape(1);
  const auto &out_shape = op.getOutputShape(0);
  const int32_t tmp_size = kernel_shape.dim(1) * kernel_shape.dim(2) * kernel_shape.dim(3) *
                           out_shape.dim(0) * out_shape.dim(1) * out_shape.dim(2);
  updateMaxTemporarySize(static_cast<size_t>(tmp_size));
  appendOperationToInference(&op, "conv2d", {_temp_tensor_id});
}

void ModelAnalyzer::visit(ops::DepthwiseConv2DOp &op)
{
  appendOperationToInference(&op, "depthwiseConv2d");
}

void ModelAnalyzer::visit(ops::SoftmaxOp &op) { appendOperationToInference(&op, "softmax"); }

void ModelAnalyzer::visit(ops::AvgPool2DOp &op) { appendOperationToInference(&op, "avgPool"); }

void ModelAnalyzer::visit(ops::MaxPool2DOp &op) { appendOperationToInference(&op, "maxPool"); }

void ModelAnalyzer::visit(ops::FullyConnectedOp &op)
{
  appendOperationToInference(&op, "fullConnect");
}

void ModelAnalyzer::visit(ops::BroadcastOp &op) { appendOperationToInference(&op, "broadcast"); }

void ModelAnalyzer::visit(ops::CappedReluOp &op) { appendOperationToInference(&op, "cappedRelu"); }

void ModelAnalyzer::visit(ops::InputOp &op)
{
  assert(op.getNumInputs() == 0);
  appendOperationToInference(&op, "in");
}

void ModelAnalyzer::visit(ops::ConstantOp &op)
{
  assert(op.getNumInputs() == 0);

  // FIXME This is to work around deserializeTensors not being able to deserialize tensors of type
  // other than float32.
  const auto *output = op.getOutput(0);
  if (output->getUses().empty())
    return;

  appendOperationToInference(&op, "constant");
}

void ModelAnalyzer::visit(ops::ReluOp &op) { appendOperationToInference(&op, "relu"); }

void ModelAnalyzer::visit(ops::ReshapeOp &op) { appendOperationToInference(&op, "reshape"); }

void ModelAnalyzer::visit(mir::ops::ResizeOp &op)
{
  const auto &in_shape = op.getInputShape(0);
  const auto &out_shape = op.getOutputShape(0);

  assert(in_shape.rank() == 4);
  assert(in_shape.rank() == out_shape.rank());

  if (in_shape.dim(0) != out_shape.dim(0) || in_shape.dim(3) != out_shape.dim(3))
    throw std::runtime_error("Not supported Resize on other dims besides height and width!");

  switch (op.getMode())
  {
    case mir::ops::ResizeOp::ResizeMethod::nearestNeighbor:
      appendOperationToInference(&op, "resize");
      break;
    default:
      assert(false && "Not Implemented!");
  }
}

void ModelAnalyzer::visit(mir::ops::SliceOp &op) { appendOperationToInference(&op, "slice"); }

void ModelAnalyzer::visit(mir::ops::TanhOp &op)
{
  appendOperationToInference(&op, "tanhActivation");
}

void ModelAnalyzer::visit(mir::ops::EluOp &op) { appendOperationToInference(&op, "elu"); }

void ModelAnalyzer::visit(mir::ops::DeConv2DOp &op)
{
  const auto &kernel_shape = op.getInputShape(1);
  const auto &out_shape = op.getOutputShape(0);
  const int32_t tmp_size = kernel_shape.dim(0) * kernel_shape.dim(1) * kernel_shape.dim(3) *
                           out_shape.dim(0) * out_shape.dim(1) * out_shape.dim(2);
  updateMaxTemporarySize(static_cast<size_t>(tmp_size));
  appendOperationToInference(&op, "convTransposed2d", {_temp_tensor_id});
}

void ModelAnalyzer::visit(ops::SqueezeOp &op) { appendOperationToInference(&op, "reshape"); }

void ModelAnalyzer::visit(ops::SqrtOp &op) { appendOperationToInference(&op, "sqrtFN"); }

void ModelAnalyzer::visit(mir::ops::PadOp &op) { appendOperationToInference(&op, "pad"); }

void ModelAnalyzer::visit(mir::ops::ReduceMeanOp &op)
{
  appendOperationToInference(&op, "reduceMean");
}

void ModelAnalyzer::visit(mir::ops::TransposeOp &op)
{
  appendOperationToInference(&op, "transpose");
}

void ModelAnalyzer::visit(mir::ops::GatherOp &op) { appendOperationToInference(&op, "gather"); }

void ModelAnalyzer::visit(mir::ops::SigmoidOp &op) { appendOperationToInference(&op, "sigmoid"); }

void ModelAnalyzer::visit(mir::ops::LeakyReluOp &op)
{
  appendOperationToInference(&op, "leakyRelu");
}

void ModelAnalyzer::visit(mir::ops::OutputOp &op) { appendOperationToInference(&op, "out"); }

void ModelAnalyzer::visit(mir::ops::AbsOp &op) { appendOperationToInference(&op, "absFN"); }

void ModelAnalyzer::visit(mir::ops::AddOp &op)
{
  appendOperationToInference(&op, "ElementWise<Add>");
}

void ModelAnalyzer::visit(mir::ops::DivOp &op)
{
  appendOperationToInference(&op, "ElementWise<Div>");
}

void ModelAnalyzer::visit(mir::ops::MaxOp &op)
{
  appendOperationToInference(&op, "ElementWise<Max>");
}

void ModelAnalyzer::visit(mir::ops::MulOp &op)
{
  appendOperationToInference(&op, "ElementWise<Mul>");
}

void ModelAnalyzer::visit(mir::ops::SubOp &op)
{
  appendOperationToInference(&op, "ElementWise<Sub>");
}

void ModelAnalyzer::visit_fallback(mir::Operation &) { throw std::runtime_error("NYI operation"); }

} // namespace nnc
