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

#ifndef __ONERT_BACKEND_I_OPTIMIZER_H__
#define __ONERT_BACKEND_I_OPTIMIZER_H__

namespace onert
{
namespace ir
{
class LoweredGraph;
}
} // namespace onert

namespace onert
{
namespace backend
{

/**
 * @brief Class for backend optimizations. This is an optional class so not all backends must have
 * it.
 *
 */
struct IOptimizer
{
  virtual ~IOptimizer() = default;
  /**
   * @brief Run optimization
   *
   */
  virtual void optimize() = 0;
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_I_OPTIMIZER_H__
