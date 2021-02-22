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

#include <enco/Frontend.h>
#include <enco/Backend.h>

#include <cmdline/View.h>

#include <string>
#include <vector>

#include <functional>

namespace cmdline
{

// TODO Extract this helper class
class Vector : public cmdline::View
{
public:
  uint32_t size(void) const { return _args.size(); }

public:
  const char *at(uint32_t nth) const { return _args.at(nth).c_str(); }

public:
  Vector &append(const std::string &arg)
  {
    _args.emplace_back(arg);
    return (*this);
  }

private:
  std::vector<std::string> _args;
};

} // namespace cmdline

namespace
{

class Zone
{
public:
  Zone() = default;

public:
  const cmdline::View *args(void) const { return &_args; }

public:
  void append(const std::string &arg) { _args.append(arg); }

private:
  cmdline::Vector _args;
};

} // namespace

#include <dlfcn.h>

namespace
{

class FrontendFactory
{
public:
  FrontendFactory(const std::string &path)
  {
    _handle = dlopen(path.c_str(), RTLD_LAZY);
    assert(_handle != nullptr);
  }

public:
  // Copy is not allowed to avoid double close
  FrontendFactory(const FrontendFactory &) = delete;
  FrontendFactory(FrontendFactory &&) = delete;

public:
  ~FrontendFactory() { dlclose(_handle); }

private:
  using Entry = std::unique_ptr<enco::Frontend> (*)(const cmdline::View &);

private:
  Entry entry(void) const
  {
    auto entry = reinterpret_cast<Entry>(dlsym(_handle, "make_frontend"));
    assert(entry != nullptr);
    return entry;
  }

public:
  std::unique_ptr<enco::Frontend> make(const cmdline::View *args) const
  {
    auto fn = entry();
    return fn(*args);
  }

private:
  void *_handle;
};

} // namespace

namespace
{

class FrontendZone : public Zone
{
public:
  FrontendZone(const std::string &path) : _factory{path}
  {
    // DO NOTHING
  }

public:
  const FrontendFactory *factory(void) const { return &_factory; }

private:
  FrontendFactory _factory;
};

} // namespace

#include <memory>
#include <map>

#include <iostream>
#include <stdexcept>

static int entry(int argc, char **argv)
{
  // Usage:
  //  [Command] --frontend [Frontend .so path] --frontend-arg ...
  std::unique_ptr<FrontendZone> frontend_zone;
  cmdline::Vector backend_args;

  // Simple argument parser (based on map)
  std::map<std::string, std::function<void(const std::string &arg)>> argparse;

  argparse["--frontend"] = [&](const std::string &path) {
    frontend_zone = std::make_unique<FrontendZone>(path);
  };

  argparse["--frontend-arg"] = [&](const std::string &arg) { frontend_zone->append(arg); };
  argparse["--backend-arg"] = [&](const std::string &arg) { backend_args.append(arg); };

  if (argc < 2)
  {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "[Command] --frontend [.so path]" << std::endl;
    std::cerr << "          --frontend-arg [argument] ..." << std::endl;
    std::cerr << "          --backend-arg [argument] ..." << std::endl;
    return 255;
  }

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

  assert(frontend_zone != nullptr);

  auto frontend = frontend_zone->factory()->make(frontend_zone->args());

  auto bundle = frontend->load();

  auto backend = make_backend(backend_args);

  backend->compile(bundle.module(), bundle.data());

  return 0;
}

#ifdef NDEBUG
int main(int argc, char **argv)
{
  try
  {
    return entry(argc, argv);
  }
  catch (const std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
  }

  return 255;
}
#else  // NDEBUG
int main(int argc, char **argv)
{
  // NOTE main does not catch internal exceptions for debug build to make it easy to
  //      check the stacktrace with a debugger
  return entry(argc, argv);
}
#endif // !NDEBUG
