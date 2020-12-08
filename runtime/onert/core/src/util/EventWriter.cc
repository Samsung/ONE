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

#include "util/EventWriter.h"

#include <cassert>

// initialization
std::mutex EventWriter::_mutex;

void EventWriter::readyToFlush(std::unique_ptr<EventRecorder> &&recorder)
{
  {
    std::unique_lock<std::mutex> lock{_mutex};

    _recorders.emplace_back(std::move(recorder));

    if (--_ref_count > 0)
      return;
  }
  // The caller of this method is the last instance that uses EventWriter.
  // Let's write log files.

  // Note. According to an internal issue, let snpe json as just file name not '.snpe.json'
  flush(WriteFormat::SNPE_BENCHMARK);
  flush(WriteFormat::CHROME_TRACING);
  flush(WriteFormat::MD_TABLE);
}

void EventWriter::flush(WriteFormat write_format)
{
  auto *writer = _actual_writers[write_format].get();
  assert(writer);

  writer->flush(_recorders);
}
