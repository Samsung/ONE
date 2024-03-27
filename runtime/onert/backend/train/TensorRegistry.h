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

#ifndef __ONERT_BACKEND_TRAIN_TENSOR_REGISTRY__
#define __ONERT_BACKEND_TRAIN_TENSOR_REGISTRY__

#include <backend/train/ITensorRegistry.h>

#include "Tensor.h"

namespace onert
{
namespace backend
{
namespace train
{

class TensorRegistry
  : public PortableTensorRegistryTemplate<Tensor, TrainableTensor, BackPropTensor, GradientTensor>
{
public:
  BackPropTensor *getDisposableBackPropTensor(const DisposableTensorIndex &index)
  {
    auto itr = _disposable_back_prop.find(index);
    if (itr != _disposable_back_prop.end())
      return itr->second.get();

    return nullptr;
  }

  void setDisposableBackPropTensor(const DisposableTensorIndex &index,
                                   std::unique_ptr<BackPropTensor> tensor)
  {
    assert(tensor != nullptr);
    auto itr = _disposable_back_prop.find(index);
    if (itr != _disposable_back_prop.end())
      throw std::runtime_error{
        "Tried to set a disposable tensor but another disposable tensor already exists."};

    _disposable_back_prop[index] = std::move(tensor);
  }

  const std::unordered_map<DisposableTensorIndex, std::unique_ptr<BackPropTensor>> &
  disposable_back_prop_tensors()
  {
    return _disposable_back_prop;
  }

private:
  // Disposable Tensors to be accumulated to BackPropTensor
  std::unordered_map<DisposableTensorIndex, std::unique_ptr<BackPropTensor>> _disposable_back_prop;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_TENSOR_REGISTRY__
