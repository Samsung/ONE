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

#ifndef __ONERT_ODC_QUANTIZE_TYPE_H__
#define __ONERT_ODC_QUANTIZE_TYPE_H__

namespace onert
{
namespace odc
{

enum QuantizeType
{
  /** default value: type not set */
  ODC_QTYPE_NOT_SET,
  /** asymmetric quantization with a scale and zero point */
  ODC_QTYPE_U8_ASYM,
  /** symmetric quantization with a scale only */
  ODC_QTYPE_I16_SYM,
  /** weight-only int8 symmetric quantization */
  ODC_QTYPE_WO_I8_SYM,
  /** weight-only int16 symmetric quantization */
  ODC_QTYPE_WO_I16_SYM,
};

} // namespace odc
} // namespace onert

#endif // __ONERT_ODC_QUANTIZE_TYPE_H__
