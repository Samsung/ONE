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

#ifndef __MLAPSE_CSV_REPORT_GENERATOR_H__
#define __MLAPSE_CSV_REPORT_GENERATOR_H__

#include "mlapse/benchmark_observer.h"

#include <fstream>
#include <string>

namespace mlapse
{

class CSVReportGenerator final : public BenchmarkObserver
{
public:
  CSVReportGenerator(const std::string &path) : _ofs{path, std::ofstream::out}
  {
    // DO NOTHING
  }

public:
  void notify(const NotificationArg<PhaseBegin> &arg) final;
  void notify(const NotificationArg<PhaseEnd> &arg) final;
  void notify(const NotificationArg<IterationBegin> &arg) final;
  void notify(const NotificationArg<IterationEnd> &arg);

private:
  std::ofstream _ofs;

  Phase _phase = uninitialized_phase();
};

} // namespace mlapse

#endif // __MLAPSE_MULTICAST_OBSERER_H__
