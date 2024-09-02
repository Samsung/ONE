/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_BUILTIN_TRAIN_KERNEL_PERMUTELAYER_H__
#define __ONERT_BACKEND_BUILTIN_TRAIN_KERNEL_PERMUTELAYER_H__

#include "../../kernel/PermuteLayer.h"

#include "exec/train/ITrainableFunction.h"

namespace onert
{
namespace backend
{
namespace builtin
{
namespace train
{
namespace kernel
{

class PermuteLayer : public builtin::kernel::PermuteLayer, public exec::train::ITrainableFunction
{
public:
  PermuteLayer(const std::vector<ITensor *> &src_tensors, const std::vector<ITensor *> &dst_tensors,
               const std::vector<ITensor *> &input_back_prop_tensors,
               const std::vector<ITensor *> &output_back_prop_tensors,
               const std::vector<ir::PermuteType> &types, bool ignore_forward_in_training,
               const std::shared_ptr<ExternalContext> &external_context);

  void optimize() override;

  void forward(bool training) override;
  void backward() override;

private:
  std::vector<ITensor *> _input_back_prop_tensors;
  std::vector<ITensor *> _output_back_prop_tensors;
  bool _ignore_forward_in_training;
};

} // namespace kernel
} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BUILTIN_TRAIN_KERNEL_PERMUTELAYER_H__
