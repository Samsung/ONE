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
#include <vector>
#include <unordered_map>
#include <mutex>
#include <fstream>

class EventFormatWriter
{
public:
  EventFormatWriter(const std::string &filepath) : _os{filepath, std::ofstream::out} {}
  virtual ~EventFormatWriter()
  { /* empty */
  }

  virtual void flush(const std::vector<std::unique_ptr<EventRecorder>> &) = 0;

protected:
  std::ofstream _os;
};

class SNPEWriter : public EventFormatWriter
{
public:
  SNPEWriter(const std::string &filepath) : EventFormatWriter(filepath)
  { /* empty */
  }
  ~SNPEWriter() {}

  void flush(const std::vector<std::unique_ptr<EventRecorder>> &) override;
};

class ChromeTracingWriter : public EventFormatWriter
{
public:
  ChromeTracingWriter(const std::string &filepath) : EventFormatWriter(filepath)
  { /* empty */
  }
  ~ChromeTracingWriter() {}

  void flush(const std::vector<std::unique_ptr<EventRecorder>> &) override;

private:
  void flushOneRecord(const EventRecorder &);
};

class MDTableWriter : public EventFormatWriter
{
public:
  MDTableWriter(const std::string &filepath) : EventFormatWriter(filepath)
  { /* empty */
  }
  ~MDTableWriter() {}

  void flush(const std::vector<std::unique_ptr<EventRecorder>> &) override;
};

#include <mutex>

class EventWriter
{
public:
  enum class WriteFormat
  {
    CHROME_TRACING,
    SNPE_BENCHMARK,
    MD_TABLE,
  };

  /**
   * @brief Retuens a singleton object
   */
  static EventWriter *get(const std::string &workspace_dir)
  {
    std::unique_lock<std::mutex> lock{_mutex};

    static EventWriter singleton(workspace_dir);
    return &singleton;
  }

  /**
   * @brief Call this when observer which use EventWriter starts
   */
  void startToUse()
  {
    std::unique_lock<std::mutex> lock{_mutex};
    _ref_count++;
  }

  /**
   * @brief Call this when observer which use EventWriter finishes.
   *        After multiple observers calls this method, the reference count will eventually be 0.
   *        Then, EventWriter will write profiling result file.
   */
  void readyToFlush(std::unique_ptr<EventRecorder> &&recorder);

private:
  EventWriter(const std::string &workspace_dir) : _ref_count(0)
  {
    std::string snpe_log_name(workspace_dir + "/trace.json");
    std::string chrome_tracing_log_name(workspace_dir + "/trace.chrome.json");
    std::string md_table_log_name(workspace_dir + "/trace.table.md");

    _actual_writers[WriteFormat::SNPE_BENCHMARK] = std::make_unique<SNPEWriter>(snpe_log_name);
    _actual_writers[WriteFormat::CHROME_TRACING] =
      std::make_unique<ChromeTracingWriter>(chrome_tracing_log_name);
    _actual_writers[WriteFormat::MD_TABLE] = std::make_unique<MDTableWriter>(md_table_log_name);
  };

  void flush(WriteFormat write_format);

private:
  static std::mutex _mutex;

  // number of observer of an executor that want to write profiling data
  int32_t _ref_count;

  // one recorder object per executor
  std::vector<std::unique_ptr<EventRecorder>> _recorders;

  std::unordered_map<WriteFormat, std::unique_ptr<EventFormatWriter>> _actual_writers;
};

#endif // __ONERT_UTIL_EVENT_WRITER_H__
