/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ExtraTensorGenerator.h"

#include "ops/BackPropAccumulator.h"
#include "ops/BinaryArithmeticLayer.h"
#include "ops/ConvolutionLayer.h"
#include "ops/DepthwiseConvolutionLayer.h"
#include "ops/ElementwiseActivationLayer.h"
#include "ops/FullyConnectedLayer.h"
#include "ops/LossMeanSquaredErrorLayer.h"
#include "ops/LossCategoricalCrossentropyLayer.h"
#include "ops/MeanLayer.h"
#include "ops/GradientApplier.h"
#include "ops/PadLayer.h"
#include "ops/PoolLayer.h"
#include "ops/ReshapeLayer.h"
#include "ops/SoftMaxLayer.h"

namespace onert
{
namespace backend
{
namespace train
{

namespace
{

bool is_fwd(const ExtraTensorLifeTime lifetime) { return lifetime == ExtraTensorLifeTime::FORWARD; }

bool is_bwd(const ExtraTensorLifeTime lifetime)
{
  return lifetime == ExtraTensorLifeTime::BACKWARD;
}

bool is_fwd_or_fwd2bwd(const ExtraTensorLifeTime lifetime)
{
  return lifetime == ExtraTensorLifeTime::FORWARD ||
         lifetime == ExtraTensorLifeTime::FORWARD_TO_BACKWARD;
}

bool is_bwd_or_fwd2bwd(const ExtraTensorLifeTime lifetime)
{
  return lifetime == ExtraTensorLifeTime::BACKWARD ||
         lifetime == ExtraTensorLifeTime::FORWARD_TO_BACKWARD;
}

} // namespace

ExtraTensorGenerator::ExtraTensorGenerator(const ir::train::TrainableGraph &tgraph,
                                           std::shared_ptr<TensorBuilder> &tensor_builder,
                                           std::shared_ptr<ITensorRegistry> &tensor_registry)
  : _tgraph(tgraph), _tensor_builder(tensor_builder)
{
  _tensor_reg = std::dynamic_pointer_cast<TensorRegistry>(tensor_registry);

  for (const auto &index : _tgraph.topolSortOperations())
  {
    const auto &node = _tgraph.operation(index);
    _node_to_idx[&node] = index;
  }
};

void ExtraTensorGenerator::generate()
{
  _tgraph.operations().iterate([&](const ir::OperationIndex &, const ir::IOperation &op) {
    const auto trainable_op = dynamic_cast<const ir::train::TrainableOperation *>(&op);
    trainable_op->accept(*this);
  });

  handle_requests();
}

void ExtraTensorGenerator::notify_first_use(ir::OperationIndex op_idx,
                                            const ExtraTensorRequests &reqs,
                                            bool (*cond)(const ExtraTensorLifeTime))
{
  for (size_t i = 0; i < reqs.size(); ++i)
  {
    if (cond(reqs[i].lifetime))
      _tensor_builder->notifyFirstUse(ExtraTensorIndex(op_idx, i));
  }
  return;
}

void ExtraTensorGenerator::notify_last_use(ir::OperationIndex op_idx,
                                           const ExtraTensorRequests &reqs,
                                           bool (*cond)(const ExtraTensorLifeTime))
{
  for (size_t i = 0; i < reqs.size(); ++i)
  {
    if (cond(reqs[i].lifetime))
      _tensor_builder->notifyLastUse(ExtraTensorIndex(op_idx, i));
  }
  return;
}

void ExtraTensorGenerator::handle_requests()
{
  // register tensor
  for (const auto &pair : _idx_to_requests)
  {
    const auto &reqs = pair.second;
    for (size_t i = 0; i < reqs.size(); ++i)
    {
      ExtraTensorIndex idx(pair.first, i);
      _tensor_builder->registerExtraTensorInfo(idx, reqs[i].info, reqs[i].layout);
    }
  }

  // forward
  for (const auto &op_index : _tgraph.topolSortOperations())
  {
    if (_idx_to_requests.find(op_index) == _idx_to_requests.end())
      continue;

    const auto &reqs = _idx_to_requests[op_index];
    notify_first_use(op_index, reqs, is_fwd_or_fwd2bwd);
    notify_last_use(op_index, reqs, is_fwd);
  }

  // backward
  for (const auto &op_index : _tgraph.btopolSortOperations())
  {
    if (_idx_to_requests.find(op_index) == _idx_to_requests.end())
      continue;

    const auto &reqs = _idx_to_requests[op_index];
    notify_first_use(op_index, reqs, is_bwd);
    notify_last_use(op_index, reqs, is_bwd_or_fwd2bwd);
  }
}

void ExtraTensorGenerator::visit(const ir::train::operation::FullyConnected &node)
{
  using ir::train::operation::FullyConnected;

  const auto out_index{node.getOutputs().at(0)};
  const auto in_index{node.getInputs().at(FullyConnected::Input::INPUT)};
  const auto weights_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};

  auto in_tensor = _tensor_reg->getPortableTensor(in_index);
  auto weights_tensor = _tensor_reg->getTrainableTensor(weights_index);
  auto out_back_prop_tensor = _tensor_reg->getBackPropTensor(out_index);
  const auto activation = node.param().activation;

  auto requests = ops::FullyConnectedLayer::requestExtraTensors(weights_tensor, in_tensor,
                                                                out_back_prop_tensor, activation);

  auto op_idx = _node_to_idx[&node];
  _idx_to_requests.emplace(op_idx, requests);
  return;
}

} // namespace train
} // namespace backend
} // namespace onert
