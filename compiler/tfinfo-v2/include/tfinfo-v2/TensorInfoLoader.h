/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __TFINFO_TENSOR_INFO_LOADER_H__
#define __TFINFO_TENSOR_INFO_LOADER_H__

#include "TensorSignature.h"

#include <memory>
#include <vector>

namespace tfinfo
{
inline namespace v2
{

/**
 * @brief Function to create TensorSignatures defined in info file
 */
TensorSignatures load(const char *info_path);

/**
 * @brief Function to create TensorSignatures from stream
 */
TensorSignatures load(std::istream *stream, const char *path_for_error_msg);

} // namespace v2
} // namespace tfinfo

#endif // __TFINFO_TENSOR_INFO_LOADER_H__
