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

#ifndef __ENCO_PASS_H__
#define __ENCO_PASS_H__

#include "Session.h"

#include <string>

namespace enco
{

class Pass
{
public:
  class Name
  {
  public:
    Name(const std::string &content) : _content{content}
    {
      // DO NOTHING
    }

    Name(const Name &) = default;
    Name(Name &&) = default;

    ~Name() = default;

  public:
    const std::string &content(void) const { return _content; }

  private:
    std::string _content;
  };

public:
  Pass(const Name &name) : _name{name}
  {
    // DO NOTHING
  }

  Pass(const Pass &) = delete;
  Pass(Pass &&) = delete;

  virtual ~Pass() = default;

public:
  const Name &name(void) const { return _name; }

public:
  virtual void run(const SessionID &) const = 0;

private:
  Name _name;
};

static inline Pass::Name pass_name(const std::string &name) { return Pass::Name{name}; }

} // namespace enco

#define PASS_CTOR(NAME) \
  NAME() : enco::Pass { enco::pass_name(#NAME) }

#endif // __ENCO_PASS_H__
