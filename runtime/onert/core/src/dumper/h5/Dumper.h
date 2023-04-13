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

#ifndef __ONERT_DUMPER_H5_DUMPER_H__
#define __ONERT_DUMPER_H5_DUMPER_H__

#include "exec/MinMaxMap.h"

#include <H5Cpp.h>
#include <string>

namespace onert
{
namespace dumper
{
namespace h5
{

class Dumper
{
public:
  /**
   * @brief Construct dumper
   *
   * @param[in] path  filepath to dump
   * @throw 	H5::FileIException on error during file open/create
   */
  Dumper(const std::string &filepath);

protected:
  H5::H5File _file;
};

} // namespace h5
} // namespace dumper
} // namespace onert

#endif // __ONERT_DUMPER_H5_DUMPER_H__
