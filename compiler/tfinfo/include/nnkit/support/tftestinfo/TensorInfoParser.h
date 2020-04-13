/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file     TensorInfoParser.h
 * @brief    This file contains functions to parse test.info files in moco/test/tf
 */

#ifndef __NNKIT_SUPPORT_TFTESTINFO_TENSOR_INFO_PARSER_H__
#define __NNKIT_SUPPORT_TFTESTINFO_TENSOR_INFO_PARSER_H__

#include "ParsedTensor.h"

#include <memory>
#include <vector>

namespace nnkit
{
namespace support
{
namespace tftestinfo
{

/**
 * @brief Function to parse test.info
 */
std::vector<std::unique_ptr<ParsedTensor>> parse(const char *info_path);

} // namespace tftestinfo
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_TFTESTINFO_TENSOR_INFO_PARSER_H__
