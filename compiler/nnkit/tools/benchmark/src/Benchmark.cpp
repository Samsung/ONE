/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <nnkit/CmdlineArguments.h>
#include <nnkit/VectorArguments.h>
#include <nnkit/BackendPlugin.h>

#include <memory>
#include <map>
#include <string>

#include <chrono>

#include <iostream>
#include <iomanip>

using std::make_unique;

using std::chrono::milliseconds;
using std::chrono::microseconds;

using milliseconds_f = std::chrono::duration<float, std::milli>;

using std::chrono::duration_cast;

namespace
{

template <class Rep, class Period> class Session
{
public:
  Session(std::chrono::duration<Rep, Period> *out) : _out{out}
  {
    // DO NOTHING
  }

public:
  template <typename Callable> void measure(Callable cb)
  {
    using namespace std::chrono;

    auto beg = steady_clock::now();
    cb();
    auto end = steady_clock::now();

    (*_out) += duration_cast<duration<Rep, Period>>(end - beg);
  }

private:
  std::chrono::duration<Rep, Period> *_out;
};

template <class Rep, class Period, typename Callable>
Session<Rep, Period> &operator<<(Session<Rep, Period> &&sess, Callable &&cb)
{
  sess.measure(std::forward<Callable>(cb));
  return sess;
}

template <class Rep, class Period>
Session<Rep, Period> measure(std::chrono::duration<Rep, Period> &out)
{
  return Session<Rep, Period>{&out};
}

class Message
{
public:
  Message(const std::string &head) { std::cout << head; }

public:
  ~Message() { std::cout << std::endl; }

public:
  std::ostream &os(void) const { return std::cout; }
};

Message info(void) { return Message{"INFO: "}; }

using OptionHook = std::function<void(const std::string &arg)>;

} // namespace

template <typename T> std::ostream &operator<<(const ::Message &m, T &&value)
{
  return m.os() << std::forward<T>(value);
}

int main(int argc, char **argv)
{
  std::unique_ptr<nnkit::BackendPlugin> backend_plugin;
  nnkit::VectorArguments backend_arguments;

  uint32_t warmup_count = 3;
  uint32_t benchmark_count = 1;

  // Simple argument parser (based on map)
  std::map<std::string, OptionHook> argparse;

  argparse["--backend"] = [&backend_plugin](const std::string &tag) {
    backend_plugin = std::move(nnkit::make_backend_plugin(tag));
  };

  argparse["--backend-arg"] = [&backend_arguments](const std::string &arg) {
    backend_arguments.append(arg);
  };

  argparse["--benchmark-count"] = [&benchmark_count](const std::string &arg) {
    benchmark_count = std::stoi(arg);
  };

  for (int n = 1; n < argc; n += 2)
  {
    const std::string tag{argv[n]};
    const std::string arg{argv[n + 1]};

    auto it = argparse.find(tag);

    if (it == argparse.end())
    {
      std::cerr << "Option '" << tag << "' is not supported" << std::endl;
      return 255;
    }

    it->second(arg);
  }

  // CHECK: Command-line arguments
  if (backend_plugin == nullptr)
  {
    std::cerr << "ERROR: --backend is mssing" << std::endl;
    return 255;
  }

  // Initialize
  auto backend = backend_plugin->create(backend_arguments);

  // Run warm-up iterations
  info() << "Start warming-up iterations(" << warmup_count << ")";

  for (uint32_t n = 0; n < warmup_count; ++n)
  {
    backend->prepare([](nnkit::TensorContext &ctx) {
      // DO NOTHING
    });

    backend->run();

    backend->teardown([](nnkit::TensorContext &ctx) {
      // DO NOTHING
    });
  }

  // Run benchmark iterations
  info() << "Start benchmark iterations(" << benchmark_count << ")";

  microseconds elapsed_min{0};
  microseconds elapsed_max{0};
  microseconds elapsed_total{0};

  for (uint32_t n = 0; n < benchmark_count; ++n)
  {
    backend->prepare([](nnkit::TensorContext &ctx) {
      // DO NOTHING
    });

    microseconds elapsed{0};

    ::measure(elapsed) << [&](void) { backend->run(); };

    info() << "#" << n + 1 << " takes " << duration_cast<milliseconds_f>(elapsed).count() << "ms";

    elapsed_min = (n == 0) ? elapsed : std::min(elapsed_min, elapsed);
    elapsed_max = (n == 0) ? elapsed : std::max(elapsed_max, elapsed);
    elapsed_total += elapsed;

    backend->teardown([](nnkit::TensorContext &ctx) {
      // DO NOTHING
    });
  }

  // Show summary
  info() << "Show statistics";

  auto min_ms = duration_cast<milliseconds_f>(elapsed_min).count();
  auto max_ms = duration_cast<milliseconds_f>(elapsed_max).count();
  auto avg_ms = duration_cast<milliseconds_f>(elapsed_total).count() / benchmark_count;

  info() << "MIN: " << min_ms << "ms"
         << ", MAX: " << max_ms << "ms, AVG: " << avg_ms << "ms";

  return 0;
}
