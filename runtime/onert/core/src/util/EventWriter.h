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

#ifndef __ONERT_UTIL_EVENT_WRITER_H__
#define __ONERT_UTIL_EVENT_WRITER_H__

#include "EventRecorder.h"

#include <string>
#include <ostream>

class EventWriter
{
public:
  enum class WriteFormat
  {
    CHROME_TRACING,
    SNPE_BENCHMARK,
    MD_TABLE,
  };

public:
  EventWriter(const EventRecorder &recorder);

public:
  void writeToFiles(const std::string &base_filepath);
  void writeToFile(const std::string &filepath, WriteFormat write_format);

private:
  void writeSNPEBenchmark(std::ostream &os);
  void writeChromeTrace(std::ostream &os);
  void writeMDTable(std::ostream &os);

private:
  const EventRecorder &_recorder;
};

#endif // __ONERT_UTIL_EVENT_WRITER_H__
