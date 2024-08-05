/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_TRAINING_CONFIG_TOOL_TRAINING_DRIVER_HANDLER
#define ONERT_MICRO_TRAINING_CONFIG_TOOL_TRAINING_DRIVER_HANDLER

#include "TrainConfigData.h"
#include "OMConfig.h"
#include "OMStatus.h"

#include <fstream>
#include <vector>

namespace training_configure_tool
{

// To start training with the current set conditions and the current configuration and save the
// result
onert_micro::OMStatus
runTrainProcessWithCurConfig(onert_micro::OMConfig &config,
                             const training_configure_tool::TrainData &train_data,
                             TrainResult &train_result);

} // namespace training_configure_tool

#endif // ONERT_MICRO_TRAINING_CONFIG_TOOL_TRAINING_DRIVER_HANDLER
