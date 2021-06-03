/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "tracer.h"
#include <chrono>
#include <iostream>

using namespace std::chrono;

JsonWriter::JsonWriter(std::string dumpfile)
  : _stream(), _root(), _dumpfile(dumpfile),
    _json_file("/tmp/output_trace_" + _dumpfile + ".json", std::ofstream::out),
    _writer(_stream.newStreamWriter())
{
}

int64_t JsonWriter::add_timed_record(std::string name, std::string ph)
{
  Json::Value rec;
  rec["name"] = name;
  rec["pid"] = 0;
  rec["tid"] = _dumpfile;
  rec["ph"] = ph;
  auto ts_val = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  rec["ts"] = ts_val;
  _root["traceEvents"].append(rec);
  return ts_val;
}

void JsonWriter::add_instance_record(std::string name)
{
  Json::Value rec;
  rec["name"] = name;
  rec["pid"] = 0;
  rec["tid"] = _dumpfile;
  rec["ph"] = "i";
  rec["s"] = "g";
  rec["ts"] = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  _root["traceEvents"].append(rec);
}

void JsonWriter::open_file(void)
{
  std::ifstream _json_input("/tmp/output_trace_" + _dumpfile + ".json", std::ifstream::binary);
  if (!_json_input)
  {
    return;
  }
  Json::CharReaderBuilder cfg;
  JSONCPP_STRING errs;
  cfg["collectComments"] = true;
  if (!parseFromStream(cfg, _json_input, &_root, &errs))
  {
    std::cout << errs << std::endl;
  }
  _json_input.close();
}

void JsonWriter::write_to_file(void) { _writer->write(_root, &_json_file); }

void JsonWriter::write_and_close_file(void)
{
  _writer->write(_root, &_json_file);
  _json_file.close();
}
