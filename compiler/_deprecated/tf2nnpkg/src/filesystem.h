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

#ifndef __TF2NNPKG_FILESYSTEM_H__
#define __TF2NNPKG_FILESYSTEM_H__

/// @file  OS-dependent filesystem functionalities

#include <string>

namespace filesystem
{

const std::string separator();

/// @brief  Normalize compatible separator in path to default separator
std::string normalize_path(const std::string &path);

bool is_dir(const std::string &path);

bool mkdir(const std::string &path);

// TODO use variadic argument
std::string join(const std::string &path1, const std::string &path2);

std::string basename(const std::string &path);

} // namespace filesystem

#endif // __TF2NNPKG_FILESYSTEM_H__
