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
#include "QuantizeType.h"

#include <functional>
#include <string>

namespace onert::odc
{

class Quantize;

class QuantizeManager
{
public:
  // Non-copyable
  QuantizeManager() = default;
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
   * @param qtype quantization type
   *
   * @todo  Support more general quantize type
   */
  void quantizeType(QuantizeType qtype) { _qtype = qtype; }

  /**
   * @brief     Quantize model
   * @param[in] model_path  Model path to quantize
   * @return    @c true if success, otherwise @c false
   */
  bool quantize(const std::string &model_path);

  /**
   * @brief Set the number of minmax records enough for quantization
   * @return    @c true if success, otherwise @c false
   */
  bool setMinMaxRecordsThreshold(uint32_t value);

  /**
   * @brief     checking minmax recording count and threshold for quantization
   * @return    @c true if ready, otherwise @c false
   */
  bool readyForQuantize();

  /**
   * @brief     Delete MinMax File of on-device compiler
   * @return    Return true if file removed successfully
   */
  bool deleteMinMaxFile();

private:
  std::string _export_model_path = "";
  QuantizeType _qtype = ODC_QTYPE_NOT_SET;
};

} // namespace onert::odc

#endif // __ONERT_ODC_QUANTIZE_MANAGER_H__
