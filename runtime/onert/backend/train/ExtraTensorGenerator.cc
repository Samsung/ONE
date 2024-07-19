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
/*
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
*/
} // namespace

ExtraTensorGenerator::ExtraTensorGenerator(const ir::train::TrainableGraph* tgraph,
                                           std::shared_ptr<TensorBuilder> &tensor_builder,
                                           std::shared_ptr<TensorRegistry> &tensor_registry)
                                           :_tgraph(tgraph), _tensor_builder(tensor_builder), _tensor_reg(tensor_registry)
                                           {};

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

void ExtraTensorGenerator::generate(ir::OperationIndex op_idx, const ExtraTensorRequests &reqs)
{
  // save request, _idx_to_reuqests used for memory planning
  _idx_to_requests[op_idx] = reqs;

  for (size_t i = 0; i < reqs.size(); i++)
  {
    // register tensor
    ExtraTensorIndex tensor_idx(op_idx, i);
    _tensor_builder->registerExtraTensorInfo(tensor_idx, reqs[i].info, reqs[i].layout);

    // return registered tensor
    auto generated_tensor = _tensor_reg->getExtraTensor(tensor_idx);
    *reqs[i].address = generated_tensor;
  }
  return;
}

} // namespace train
} // namespace backend
} // namespace onert
