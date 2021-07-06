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

#ifndef __PEPPER_CSV2VEC_H__
#define __PEPPER_CSV2VEC_H__

#include <string>
#include <vector>

namespace pepper
{

template <typename T> std::vector<T> csv_to_vector(const std::string &str);

template <typename T> bool is_one_of(const T &item, const std::vector<T> &items);

} // namespace pepper

#endif // __PEPPER_CSV2VEC_H__
