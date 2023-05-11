/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAINING_OPS_FUSEDBATCHNORM_LAYER_H__
#define __ONERT_BACKEND_TRAINING_OPS_FUSEDBATCHNORM_LAYER_H__

#include <backend/IPortableTensor.h>
#include "OperationUtils.h"

#include <exec/IFunction.h>
#include <functional>
#include <memory>

namespace nnfw
{
namespace cker
{
class FusedBatchNorm;
}
} // namespace nnfw

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

class FusedBatchNormLayer : public ::onert::exec::IFunction
{
public:
  FusedBatchNormLayer();
  ~FusedBatchNormLayer();

public:
  void fusedbatchnormFloat32();

  void configure(const std::vector<const IPortableTensor *> &inputs, float epsilon,
                 bool is_training, std::string data_format, IPortableTensor *output);

  void run() override;

private:
  std::vector<const IPortableTensor *> _inputs;
  IPortableTensor *_output;
  float _epsilon;
  bool _is_training;
  std::string _data_format;

  std::unique_ptr<nnfw::cker::FusedBatchNorm> _fusedbatchnorm_kernel;
};

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_OPS_FUSEDBATCHNORM_LAYER_H__
