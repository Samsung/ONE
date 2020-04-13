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

#ifndef __TFLDUMP_OPPRINTER_H__
#define __TFLDUMP_OPPRINTER_H__

#include <mio/tflite/schema_generated.h>

#include <ostream>
#include <map>

namespace tfldump
{

class OpPrinter
{
public:
  virtual void options(const tflite::Operator *, std::ostream &) const {};
};

class OpPrinterRegistry
{
public:
  OpPrinterRegistry();

public:
  const OpPrinter *lookup(tflite::BuiltinOperator op) const
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
  std::map<tflite::BuiltinOperator, std::unique_ptr<OpPrinter>> _op_map;
};

} // namespace tfldump

#endif // __TFLDUMP_OPPRINTER_H__
