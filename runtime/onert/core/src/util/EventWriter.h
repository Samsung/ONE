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
#include <unordered_map>

class EventFormatWriter
{
public:
  EventFormatWriter(const EventRecorder &recorder) : _recorder(recorder)
  { /* empty */
  }
  virtual void writeToFile(std::ostream &os) = 0;
  virtual ~EventFormatWriter()
  { /* empty */
  }

protected:
  const EventRecorder &_recorder;
};

class SNPEWriter : public EventFormatWriter
{
public:
  SNPEWriter(const EventRecorder &recorder) : EventFormatWriter(recorder)
  { /* empty */
  }
  void writeToFile(std::ostream &os) override;
};

class ChromeTracingWriter : public EventFormatWriter
{
public:
  ChromeTracingWriter(const EventRecorder &recorder) : EventFormatWriter(recorder)
  { /* empty */
  }
  void writeToFile(std::ostream &os) override;
};

class MDTableWriter : public EventFormatWriter
{
public:
  MDTableWriter(const EventRecorder &recorder) : EventFormatWriter(recorder)
  { /* empty */
  }
  void writeToFile(std::ostream &os) override;
};

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
  EventWriter(const EventRecorder &recorder)
  {
    _actual_writers[WriteFormat::SNPE_BENCHMARK] = std::make_unique<SNPEWriter>(recorder);
    _actual_writers[WriteFormat::CHROME_TRACING] = std::make_unique<ChromeTracingWriter>(recorder);
    _actual_writers[WriteFormat::MD_TABLE] = std::make_unique<MDTableWriter>(recorder);
  }

public:
  void writeToFiles(const std::string &base_filepath);
  void writeToFile(const std::string &filepath, WriteFormat write_format);

private:
  std::unordered_map<WriteFormat, std::unique_ptr<EventFormatWriter>> _actual_writers;
};

#endif // __ONERT_UTIL_EVENT_WRITER_H__
