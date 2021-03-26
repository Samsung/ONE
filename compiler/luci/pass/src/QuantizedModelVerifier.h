/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LUCI_QUANTIZED_MODEL_VERIFIER_H__
#define __LUCI_QUANTIZED_MODEL_VERIFIER_H__

#include "luci/Pass/QuantizationParameters.h"

#include <loco.h>

namespace luci
{

/**
 * @brief  Class to verify quantized model
 *
 */
struct QuantizedModelVerifier
{

public:
  QuantizedModelVerifier(loco::DataType quantized_dtype, QuantizationGranularity granularity)
    : _quantized_dtype(quantized_dtype), _granularity(granularity)
  {
  }

  void verify(loco::Graph *g);

private:
  loco::DataType _quantized_dtype;
  QuantizationGranularity _granularity;
};

} // namespace luci

#endif // __LUCI_QUANTIZED_MODEL_VERIFIER_H__
