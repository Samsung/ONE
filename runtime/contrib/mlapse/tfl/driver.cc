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

#include "mlapse/benchmark_runner.h"
#include "mlapse/multicast_observer.h"
#include "mlapse/CSV_report_generator.h"

#include "mlapse/tfl/load.h"

// From 'nnfw_lib_tflite'
#include <tflite/InterpreterSession.h>
#include <tflite/NNAPISession.h>

#include <memory>

// From C++ Standard Library
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

namespace
{

using namespace mlapse;

class ConsoleReporter final : public mlapse::BenchmarkObserver
{
public:
  ConsoleReporter() = default;

public:
  void notify(const NotificationArg<PhaseBegin> &arg) final
  {
    _phase = arg.phase;
    _count = arg.count;

    std::cout << tag() << " BEGIN" << std::endl;
  }

  void notify(const NotificationArg<PhaseEnd> &arg) final
  {
    std::cout << tag() << " END" << std::endl;

    _phase = mlapse::uninitialized_phase();
    _count = 0;
  }

  void notify(const NotificationArg<IterationBegin> &arg) final { _index = arg.index; }

  void notify(const NotificationArg<IterationEnd> &arg) final
  {
    std::cout << tag() << " " << progress() << " - " << arg.latency.count() << "ms" << std::endl;
  }

private:
  std::string progress(void) const
  {
    return "[" + std::to_string(_index + 1) + "/" + std::to_string(_count) + "]";
  }

  std::string tag(void) const
  {
    switch (_phase)
    {
      case Phase::Warmup:
        return "WARMUP";
      case Phase::Record:
        return "RECORD";
      default:
        break;
    }

    return "unknown";
  }

  Phase _phase = mlapse::uninitialized_phase();
  uint32_t _count = 0;
  uint32_t _index = 0;
};

} // namespace

// Q. Is is worth to make a library for these routines?
namespace
{

enum class SessionType
{
  Interp,
  NNAPI,
};

class SessionBuilder
{
public:
  SessionBuilder(const SessionType &type) : _type{type}
  {
    // DO NOTHING
  }

public:
  std::unique_ptr<nnfw::tflite::Session> with(tflite::Interpreter *interp) const
  {
    switch (_type)
    {
      case SessionType::Interp:
        return std::make_unique<nnfw::tflite::InterpreterSession>(interp);
      case SessionType::NNAPI:
        return std::make_unique<nnfw::tflite::NNAPISession>(interp);
      default:
        break;
    }

    return nullptr;
  }

  std::unique_ptr<nnfw::tflite::Session>
  with(const std::unique_ptr<tflite::Interpreter> &interp) const
  {
    return with(interp.get());
  }

private:
  SessionType _type;
};

SessionBuilder make_session(const SessionType &type) { return SessionBuilder{type}; }

} // namespace

namespace
{

// mlapse-tfl
//  [REQUIRED] --model <path/to/tflite>
//  [OPTIONAL] --warmup-count N (default = 3)
//  [OPTIONAL] --record-count N (default = 10)
//  [OPTIONAL] --thread N or auto (default = auto)
//  [OPTIOANL] --nnapi (default = off)
//  [OPTIONAL] --pause N (default = 0)
//  [OPTIONAL] --csv-report <path/to/csv>
int entry(const int argc, char **argv)
{
  // Create an observer
  mlapse::MulticastObserver observer;

  observer.append(std::make_unique<ConsoleReporter>());

  // Set default parameters
  std::string model_path;
  bool model_path_initialized = false;

  SessionType session_type = SessionType::Interp;
  uint32_t warmup_count = 3;
  uint32_t record_count = 10;
  int num_thread = -1; // -1 means "auto"

  // Read command-line arguments
  std::map<std::string, std::function<uint32_t(const char *const *)>> opts;

  opts["--model"] = [&model_path, &model_path_initialized](const char *const *tok) {
    model_path = std::string{tok[0]};
    model_path_initialized = true;
    return 1; // # of arguments
  };

  opts["--record-count"] = [&record_count](const char *const *tok) {
    record_count = std::stoi(tok[0]);
    return 1; // # of arguments
  };

  opts["--thread"] = [](const char *const *tok) {
    assert(std::string{tok[0]} == "auto");
    return 1;
  };

  opts["--nnapi"] = [&session_type](const char *const *) {
    session_type = SessionType::NNAPI;
    return 0;
  };

  opts["--csv-report"] = [&observer](const char *const *tok) {
    observer.append(std::make_unique<mlapse::CSVReportGenerator>(tok[0]));
    return 1;
  };

  {
    uint32_t offset = 1;

    while (offset < argc)
    {
      auto opt = argv[offset];

      auto it = opts.find(opt);

      if (it == opts.end())
      {
        std::cout << "INVALID OPTION: " << opt << std::endl;
        return 255;
      }

      auto func = it->second;

      auto num_skip = func(argv + offset + 1);

      offset += 1;
      offset += num_skip;
    }
  }

  // Check arguments
  if (!model_path_initialized)
  {
    std::cerr << "ERROR: --model is missing" << std::endl;
    return 255;
  }

  // Load T/F Lite model
  auto model = mlapse::tfl::load_model(model_path);

  if (model == nullptr)
  {
    std::cerr << "ERROR: Failed to load '" << model_path << "'" << std::endl;
    return 255;
  }

  auto interp = mlapse::tfl::make_interpreter(model.get());

  if (interp == nullptr)
  {
    std::cerr << "ERROR: Failed to create a T/F Lite interpreter" << std::endl;
    return 255;
  }

  auto sess = make_session(session_type).with(interp);

  if (sess == nullptr)
  {
    std::cerr << "ERROR: Failed to create a session" << std::endl;
  }

  // Run benchmark
  mlapse::BenchmarkRunner benchmark_runner{warmup_count, record_count};

  benchmark_runner.attach(&observer);
  benchmark_runner.run(sess);

  return 0;
}

} // namespace

int main(int argc, char **argv)
{
  try
  {
    return entry(argc, argv);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 255;
}
