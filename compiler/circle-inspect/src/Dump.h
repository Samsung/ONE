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

#ifndef __DUMP_H__
#define __DUMP_H__

#include <mio/circle/schema_generated.h>

#include <ostream>

namespace circleinspect
{

class DumpInterface
{
public:
  virtual ~DumpInterface() = default;

public:
  virtual void run(std::ostream &os, const circle::Model *model) = 0;
};

class DumpOperators final : public DumpInterface
{
public:
  DumpOperators() = default;

public:
  void run(std::ostream &os, const circle::Model *model);
};

class DumpConv2DWeight final : public DumpInterface
{
public:
  DumpConv2DWeight() = default;

public:
  void run(std::ostream &os, const circle::Model *model);
};

class DumpOperatorVersion final : public DumpInterface
{
public:
  DumpOperatorVersion() = default;

public:
  void run(std::ostream &os, const circle::Model *model);
};

} // namespace circleinspect

#endif // __DUMP_H__
