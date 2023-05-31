/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_ICONFIG_H__
#define __ONERT_BACKEND_ICONFIG_H__

#include "ir/Layout.h"
#include "ir/Operation.h"
#include "util/ITimer.h"

#include <memory>
#include <string>

namespace onert
{
namespace backend
{

struct IConfig
{
  virtual ~IConfig() = default;
  /**
   * @brief Returns ID of the backend
   *
   * @return std::string ID of this backend
   */
  virtual std::string id() = 0;
  /**
   * @brief Initialize the backend. This is called as soon as the backend is loaded.
   *
   * @return true  Initialization succeeded
   * @return false Initialization failed, so it cannot use this backend
   */
  virtual bool initialize() = 0;
  /**
   * @brief Returns supported layout for the given \p node and \p frontend_layout
   *
   * @param node Operation
   * @param frontend_layout The layout defined in the model
   * @return ir::Layout The layout that the backend kernel actually uses
   */
  virtual ir::Layout supportLayout(const ir::Operation &node, ir::Layout frontend_layout) = 0;
  /**
   * @brief The function that is called after each Operation run on profiling mode.
   *        This may be useful for profiling GPU-based or special computing units.
   */
  virtual void sync() const {}
  /**
   * @brief Returns Timer object for this backend. For some computing units, it may need its own
   * Timer implementation.
   *
   * @return std::unique_ptr<util::ITimer> Timer object for this backend
   */
  virtual std::unique_ptr<util::ITimer> timer() { return nullptr; }

  virtual bool supportPermutation() = 0;
  virtual bool supportDynamicTensor() = 0;
  virtual bool supportFP16() = 0;
  // TODO Find a way to check whether the backend supports training at compile time
  virtual bool supportTraining() { return false; }
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ICONFIG_H__
