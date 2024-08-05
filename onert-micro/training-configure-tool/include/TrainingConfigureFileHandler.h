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

#ifndef ONERT_MICRO_TRAINING_CONFIG_TOOL_TRAINING_CONFIGURE_FILE_HANDLER
#define ONERT_MICRO_TRAINING_CONFIG_TOOL_TRAINING_CONFIGURE_FILE_HANDLER

#include "OMStatus.h"
#include "TrainConfigData.h"

#include <fstream>
#include <vector>
#include <unordered_set>

namespace training_configure_tool
{

using DataBuffer = std::vector<char>;

void readDataFromFile(const std::string &filename, char *data, size_t data_size,
                      size_t start_position = 0);

void writeDataToFile(const std::string &filename, const char *data, size_t data_size);

DataBuffer readFile(const char *path);

// Save train config data into file
onert_micro::OMStatus createResultFile(const TrainConfigFileData &train_data,
                                       const char *save_path);

// Save train config data into buffer
onert_micro::OMStatus createResultData(const TrainConfigFileData &train_data,
                                       std::vector<char> &result_buffer);

} // namespace training_configure_tool

#endif // ONERT_MICRO_TRAINING_CONFIG_TOOL_TRAINING_CONFIGURE_FILE_HANDLER
