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

#ifndef __ONERT_BACKEND_IDYNAMICTENSOR_MANAGER_H__
#define __ONERT_BACKEND_IDYNAMICTENSOR_MANAGER_H__

#include "ITensorManager.h"

#include <ir/Index.h>
#include <ir/Shape.h>
#include <backend/ITensor.h>

namespace onert
{
namespace backend
{

/**
 * @brief Interface as an abstract tensor manager, providing ways to handle memory
 *        for dynamic tensors.
 */
struct IDynamicTensorManager : public ITensorManager
{
  virtual ~IDynamicTensorManager() = default;

public:
  /**
   * @brief Allocate memory for dynamic tensor
   */
  virtual void allocate(const ir::OperandIndex &, const ir::Shape &) = 0;

  /**
   * @brief Change shape of tensor, which makes the tensor "dynamic"
   * @note  This should be called before execution.
   */
  virtual void changeShape(const ir::OperandIndex &, const ir::Shape &) = 0;
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_IDYNAMICTENSOR_MANAGER_H__
