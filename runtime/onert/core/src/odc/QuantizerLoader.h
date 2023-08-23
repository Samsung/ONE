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

#ifndef __ONERT_ODC_QUANTIZER_LOADER_H__
#define __ONERT_ODC_QUANTIZER_LOADER_H__

#include "odc/IQuantizer.h"

#include <functional>
#include <memory>

namespace onert
{
namespace odc
{

/**
 * @brief Class to manage loading and unloading of dynamic library containing
 *        implementation of IQuantizer interface
 */
class QuantizerLoader
{
public:
  /**
   * @brief Typedef for function pointer to destroy loaded library handle
   */
  using dlhandle_destroy_t = std::function<void(void *)>;
  /**
   * @brief Typedef for function pointer to create instance of IQuantizer
   */
  using factory_t = IQuantizer *(*)();
  /**
   * @brief Typedef for function pointer to destroy instance of IQuantizer
   */
  using quantizer_destory_t = void (*)(IQuantizer *);

  /**
   * @brief   Get singleton instance of QuantizerLoader
   * @return  Reference to singleton instance of QuantizerLoader
   */
  static QuantizerLoader &instance();

  // Non-copyable
  QuantizerLoader() = default;
  QuantizerLoader(QuantizerLoader const &) = delete;
  QuantizerLoader &operator=(QuantizerLoader const &) = delete;

  /**
   * @brief   Load dynamic library containing implementation of IQuantizer
   * @return  0 if success, otherwise errno value
   */
  int32_t loadLibrary();
  /**
   * @brief   Get instance of IQuantizer created through factory method
   * @return  Pointer to instance of IQuantizer
   */
  IQuantizer *get() const { return _quantizer.get(); }

private:
  // Note: Keep handle to avoid svace warning of "handle lost without dlclose()"
  std::unique_ptr<void, dlhandle_destroy_t> _dlhandle;
  std::unique_ptr<IQuantizer, quantizer_destory_t> _quantizer{nullptr, nullptr};
};

} // namespace odc
} // namespace onert

#endif // __ONERT_ODC_QUANTIZER_LOADER_H__
