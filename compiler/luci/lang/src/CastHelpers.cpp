/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/CastHelpers.h"

#include <memory>
#include <cxxabi.h>
#include <execinfo.h>

namespace luci
{

namespace
{

inline std::string _demangle(const char *name)
{
  using free_t = void (*)(void *);
  int status = 0;
  std::unique_ptr<char, free_t> res{abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
  return (status == 0) ? res.get() : name;
}

} // namespace

__attribute__((noinline)) std::string _callstacks(void)
{
  std::string msg = "\nCall stack:\n";
  const int max_frames = 5;
  void *addrlist[max_frames];

  int addrlen = backtrace(addrlist, max_frames);
  if (addrlen > 0)
  {
    char **symbollist = backtrace_symbols(addrlist, addrlen);
    if (symbollist)
    {
      for (int i = 1; i < addrlen; ++i)
      {
        msg += "  ";
        msg += std::to_string(i);
        msg += ": ";

        std::string symbol_line(symbollist[i]);
        size_t mangled_start = symbol_line.find('(');
        size_t mangled_end = symbol_line.find('+');
        if (mangled_start != std::string::npos && mangled_end != std::string::npos)
        {
          std::string mangled_name =
            symbol_line.substr(mangled_start + 1, mangled_end - mangled_start - 1);
          std::string demangled_name = _demangle(mangled_name.c_str());
          // TODO uncomment this if binary module is better to show
          // msg += symbol_line.substr(0, mangled_start);
          msg += "(";
          msg += demangled_name;
          msg += symbol_line.substr(mangled_end);
        }
        else
        {
          msg += symbol_line;
        }
        msg += "\n";
      }
      free(symbollist);
    }
    else
      msg += "  Call stack not available.\n";
  }
  else
    msg += "  Call stack not available.\n";

  return msg;
}

} // namespace luci
