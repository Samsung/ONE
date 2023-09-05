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

#include "IQuantizer.h"

#include <functional>
#include <string>

namespace onert
{
namespace odc
{

class Quantize;

class QuantizeManager
{
public:
  // Non-copyable
  QuantizeManager() = delete;
  QuantizeManager(const std::string &model_path) : _model_path(model_path) {}
  QuantizeManager(QuantizeManager const &) = delete;
  QuantizeManager &operator=(QuantizeManager const &) = delete;

public:
  /**
   * @brief Set model path to export quantized model
   *
   * @param model_path  Model path to export quantized model
   */
  void exportModelPath(const std::string &model_path) { _export_model_path = model_path; }

  /**
   * @brief   Get model path to export quantized model
   *
   * @return  Model path to export quantized model
   */
  std::string &exportModelPath() { return _export_model_path; }

  /**
   * @brief Set quantize type
   *
   * @param is_q16  true if q16, false if q8
   *
   * @todo  Support more general quantize type
   */
  void quantizeType(bool is_q16) { _is_q16 = is_q16; }

  /**
   * @brief  Quantize model
   *
   * @return  true if success, otherwise false
   */
  bool quantize();

private:
  std::string _model_path = "";
  std::string _export_model_path = "";
  bool _is_q16 = false;
};

} // namespace odc
} // namespace onert

#endif // __ONERT_ODC_QUANTIZE_MANAGER_H__
