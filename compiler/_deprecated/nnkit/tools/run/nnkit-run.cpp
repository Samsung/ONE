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

namespace
{

class Section
{
public:
  Section() = default;

public:
  const nnkit::CmdlineArguments &args(void) const { return _args; }

public:
  void append(const std::string &arg) { _args.append(arg); }

private:
  nnkit::VectorArguments _args;
};
} // namespace

namespace
{

class BackendSection : public Section
{
public:
  BackendSection(const std::string &path) : _path{path}
  {
    // DO NOTHING
  }

public:
  std::unique_ptr<nnkit::Backend> initialize(void)
  {
    _plugin = std::move(nnkit::make_backend_plugin(_path));
    return _plugin->create(args());
  }

private:
  std::string _path;
  std::unique_ptr<nnkit::BackendPlugin> _plugin;
};
} // namespace

// TODO Extract Action-related helpers
#include <nnkit/Action.h>

#include <memory>

#include <dlfcn.h>
#include <assert.h>

namespace
{

class ActionBinder
{
private:
  typedef std::unique_ptr<nnkit::Action> (*Entry)(const nnkit::CmdlineArguments &);

public:
  ActionBinder(const std::string &path)
  {
    // Q: Do we need RTLD_GLOBAL here?
    _handle = dlopen(path.c_str(), RTLD_LAZY);
    assert(_handle != nullptr);

    _entry = reinterpret_cast<Entry>(dlsym(_handle, "make_action"));
    assert(_entry != nullptr);
  }

public:
  // Copy is not allowed to avoid double close
  ActionBinder(const ActionBinder &) = delete;
  ActionBinder(ActionBinder &&binder)
  {
    // Handle is transferd from 'binder' instance into this instance.
    _handle = binder._handle;
    _entry = binder._entry;

    binder._handle = nullptr;
    binder._entry = nullptr;
  }

public:
  ~ActionBinder()
  {
    if (_handle)
    {
      dlclose(_handle);
    }
  }

public:
  std::unique_ptr<nnkit::Action> make(const nnkit::CmdlineArguments &args) const
  {
    return _entry(args);
  }

private:
  void *_handle;
  Entry _entry;
};
} // namespace

namespace
{

class ActionSection : public Section
{
public:
  ActionSection(const std::string &path) : _binder{path}
  {
    // DO NOTHING
  }

public:
  std::unique_ptr<nnkit::Action> initialize(void) const { return _binder.make(args()); }

private:
  ActionBinder _binder;
};
} // namespace

#include <memory>
#include <map>
#include <iostream>

int main(int argc, char **argv)
{
  // Usage:
  //  [Command] --backend [Backend module path] --backend-arg ... --backend-arg ...
  //            --pre [Action module path] --pre-arg ... --pre-arg ...
  //            --post [Action module path] --post-arg ... --post-arg ...

  // Argument sections
  //
  // NOTE Command-line arguments should include one backend section, and may include multiple
  //      pre/post action sections.
  struct Sections
  {
    std::unique_ptr<BackendSection> backend;
    std::vector<ActionSection> pre;
    std::vector<ActionSection> post;
  };

  Sections sections;

  // Simple argument parser (based on map)
  std::map<std::string, std::function<void(const std::string &arg)>> argparse;

  argparse["--backend"] = [&sections](const std::string &tag) {
    sections.backend = std::make_unique<BackendSection>(tag);
  };

  argparse["--backend-arg"] = [&sections](const std::string &arg) {
    sections.backend->append(arg);
  };

  argparse["--pre"] = [&sections](const std::string &tag) { sections.pre.emplace_back(tag); };

  argparse["--pre-arg"] = [&sections](const std::string &arg) { sections.pre.back().append(arg); };

  argparse["--post"] = [&sections](const std::string &tag) { sections.post.emplace_back(tag); };

  argparse["--post-arg"] = [&sections](const std::string &arg) {
    sections.post.back().append(arg);
  };

  if (argc < 2)
  {
    std::cerr << "Usage:" << std::endl
              << "[Command] --backend [Backend module path] "
              << "--backend-arg [Backend argument] ..." << std::endl
              << "          --pre [Pre-Action module path] "
              << "--pre-arg [Pre-Action argument] ..." << std::endl
              << "          --post [Post-Action module path] "
              << "--post-arg [Post-Action argument] ..." << std::endl;
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

  // we need a backend
  if (sections.backend == nullptr)
  {
    std::cerr << "Error: Backend is required. Provide with [--backend]" << std::endl;
    return 255;
  }

  // Initialize a backend
  auto backend = sections.backend->initialize();

  // Initialize pre actions
  std::vector<std::unique_ptr<nnkit::Action>> pre_actions;

  for (const auto &section : sections.pre)
  {
    pre_actions.emplace_back(section.initialize());
  }

  // Initialize post actions
  std::vector<std::unique_ptr<nnkit::Action>> post_actions;

  for (const auto &section : sections.post)
  {
    post_actions.emplace_back(section.initialize());
  }

  //
  // Run inference
  //
  backend->prepare([&pre_actions](nnkit::TensorContext &ctx) {
    // Run pre-actions on prepared tensor context
    for (auto &action : pre_actions)
    {
      action->run(ctx);
    }
  });

  backend->run();

  backend->teardown([&post_actions](nnkit::TensorContext &ctx) {
    // Run post-actions before teardown
    for (auto &action : post_actions)
    {
      action->run(ctx);
    }
  });

  return 0;
}
