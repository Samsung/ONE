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

#ifndef __CIRCLE_HELPER_PATH_H__
#define __CIRCLE_HELPER_PATH_H__

#include <string>
#include <cstdint>

namespace partee
{

/**
 * @brief create folder
 */
bool make_dir(const std::string &path);

/**
 * @brief get filename part of base
 */
std::string get_filename_ext(const std::string &base);

/**
 * @brief Make file path from base and backend
 */
std::string make_path(const std::string &base, const std::string &input, uint32_t idx,
                      const std::string &backend);

} // namespace partee

#endif // __CIRCLE_HELPER_PATH_H__
