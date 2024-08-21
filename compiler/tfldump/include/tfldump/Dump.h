/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __TFLDUMP_DUMP_H__
#define __TFLDUMP_DUMP_H__

#include <mio/tflite/schema_generated.h>

#include <ostream>

namespace tfldump
{

struct ModelEx
{
  const tflite::Model *model;
  const std::vector<char> *rawdata;
};

void dump_model(std::ostream &os, const ModelEx &model);

} // namespace tfldump

std::ostream &operator<<(std::ostream &os, const tfldump::ModelEx &model);

#endif // __TFLDUMP_DUMP_H__
