
/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_ODC_ODC_INFO_H__
#define __ONERT_ODC_ODC_INFO_H__

namespace onert
{
namespace odc
{

/**
 * @brief  Structure contains information about current state of on-device compiler
 */
class OdcInfo
{
public:
  OdcInfo() {}
  OdcInfo(const OdcInfo &) = default;
  OdcInfo(OdcInfo &&) = default;
  OdcInfo &operator=(const OdcInfo &) = default;
  OdcInfo &operator=(OdcInfo &&) = default;
  ~OdcInfo() = default;

  // getter
  int getMinMaxRecordsCount() const { return _minmax_records_count; }
  int getMinMaxCountForQuantization() const { return _min_max_count_for_quantization; }
  int isQuantizedModelLoaded() const { return _is_quantized_model_loaded; }
  int isCompiledModelLoaded() const { return _is_compiled_model_loaded; }

  // setter
  void setMinMaxRecordsCount(const int new_value) { _minmax_records_count = new_value; }
  void setMinMaxCountForQuantization(const int new_value)
  {
    _min_max_count_for_quantization = new_value;
  }
  void setQuantizedModelLoaded(bool new_value) { _is_quantized_model_loaded = new_value; }
  void setCompiledModelLoaded(bool new_value) { _is_compiled_model_loaded = new_value; }

private:
  int _minmax_records_count = 0;
  int _min_max_count_for_quantization = 0;
  bool _is_quantized_model_loaded = false;
  bool _is_compiled_model_loaded = false;
};

} // namespace odc
} // namespace onert

#endif // __ONERT_ODC_ODC_INFO_H__
