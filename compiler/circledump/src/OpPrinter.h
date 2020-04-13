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

#ifndef __CIRCLEDUMP_OPPRINTER_H__
#define __CIRCLEDUMP_OPPRINTER_H__

#include <mio/circle/schema_generated.h>

#include <ostream>
#include <map>

namespace circledump
{

class OpPrinter
{
public:
  virtual void options(const circle::Operator *, std::ostream &) const {};
};

class OpPrinterRegistry
{
public:
  OpPrinterRegistry();

public:
  const OpPrinter *lookup(circle::BuiltinOperator op) const
  {
    if (_op_map.find(op) == _op_map.end())
      return nullptr;

    return _op_map.at(op).get();
  }

public:
  static OpPrinterRegistry &get()
  {
    static OpPrinterRegistry me;
    return me;
  }

private:
  std::map<circle::BuiltinOperator, std::unique_ptr<OpPrinter>> _op_map;
};

} // namespace circledump

#endif // __CIRCLEDUMP_OPPRINTER_H__
