/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_CONVERT_TO_FAKE_QUANTIZED_MODEL_H__
#define __LUCI_CONVERT_TO_FAKE_QUANTIZED_MODEL_H__

#include <loco.h>

namespace luci
{

/**
 * @brief  Class to convert a quantized model to a fake-quantized fp32 model.
 */
struct ConvertToFakeQuantizedModel final
{
  void run(loco::Graph *g);
};

} // namespace luci

#endif // __LUCI_CONVERT_TO_FAKE_QUANTIZED_MODEL_H__
