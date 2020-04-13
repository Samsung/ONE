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

#include "mlapse/CSV_report_generator.h"

#include <cassert>
#include <stdexcept>

namespace
{

std::string tag(const mlapse::Phase &phase)
{
  switch (phase)
  {
    case mlapse::Phase::Warmup:
      return "WARMUP";
    case mlapse::Phase::Record:
      return "STEADY";
    default:
      break;
  }

  throw std::invalid_argument{"phase"};
}

} // namespace

namespace mlapse
{

void CSVReportGenerator::notify(const NotificationArg<PhaseBegin> &arg)
{
  assert(_phase == uninitialized_phase());
  _phase = arg.phase;
}

void CSVReportGenerator::notify(const NotificationArg<PhaseEnd> &arg)
{
  assert(_phase != uninitialized_phase());
  _phase = uninitialized_phase();
}

void CSVReportGenerator::notify(const NotificationArg<IterationBegin> &arg)
{
  // DO NOTHING
}

void CSVReportGenerator::notify(const NotificationArg<IterationEnd> &arg)
{
  _ofs << tag(_phase) << "," << arg.latency.count() << std::endl;
}

} // namespace mlapse
