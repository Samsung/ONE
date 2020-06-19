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

#ifndef __CIRCLE_TENSORDUMP_DUMP_H__
#define __CIRCLE_TENSORDUMP_DUMP_H__

#include <mio/circle/schema_generated.h>

#include <ostream>

namespace circletensordump
{

class DumpInterface
{
public:
  virtual ~DumpInterface() = default;

public:
  virtual void run(std::ostream &os, const circle::Model *model,
                   const std::string &output_path = {}) = 0;
};

class DumpTensors final : public DumpInterface
{
public:
  DumpTensors() = default;

public:
  void run(std::ostream &os, const circle::Model *model, const std::string &) override;
};

class DumpTensorsToHdf5 final : public DumpInterface
{
public:
  DumpTensorsToHdf5() = default;

public:
  void run(std::ostream &os, const circle::Model *model, const std::string &output_path) override;
};

} // namespace circletensordump

#endif // __CIRCLE_TENSORDUMP_DUMP_H__
