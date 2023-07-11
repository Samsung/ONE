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

#ifndef __ONERT_BACKEND_TRAIN_ITRAINABLE_TENSOR_H__
#define __ONERT_BACKEND_TRAIN_ITRAINABLE_TENSOR_H__

#include "backend/IPortableTensor.h"

namespace onert
{
namespace backend
{
namespace train
{

/**
 * @brief A tensor class that can be trained
 *
 */
class ITrainableTensor
{
public:
  virtual ~ITrainableTensor() = default;

  // TODO Encapsulation
  virtual IPortableTensor &weightTensor() = 0;
  virtual IPortableTensor &gradTensor() = 0;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_ITRAINABLE_TENSOR_H__
