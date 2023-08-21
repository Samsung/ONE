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

#ifndef __ONERT_ODC_QUANTIZE_MANAGER_H__
#define __ONERT_ODC_QUANTIZE_MANAGER_H__

#include <functional>
#include <memory>

namespace onert
{
namespace odc
{

class Quantize;

class QuantizeManager
{
public:
  using dlhandle_destroy_t = std::function<void(void *)>;

  static QuantizeManager &instance();

  // Non-copyable
  QuantizeManager() = default;
  QuantizeManager(QuantizeManager const &) = delete;
  QuantizeManager &operator=(QuantizeManager const &) = delete;

public:
  /**
   * @brief load plugin
   */
  int32_t loadLibrary();

  /**
   * @brief   Get Quantize
   * @note    get() should be called after loadLibrary is finished.
   * @return  Quantize instance which is created by loaded library (plug-in)
   */
  Quantize *get() const;

private:
  // Note: Keep handle to avoid svace warning of "handle lost without dlclose()"
  std::unique_ptr<void, dlhandle_destroy_t> _dlhandle;

  // Assumption there is only 1 codegen. It is created and owned by QuantizeManager after
  // loadLibrary().
  std::unique_ptr<Quantize> _default_quantize;
};

} // namespace odc
} // namespace onert

#endif // __ONERT_ODC_QUANTIZE_MANAGER_H__
