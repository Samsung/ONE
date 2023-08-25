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

#ifndef __CIRCLEDUMP_METADATA_PRINTER_H__
#define __CIRCLEDUMP_METADATA_PRINTER_H__

#include <ostream>
#include <string>
#include <map>
#include <memory>

namespace circledump
{

class MetadataPrinter
{
public:
  virtual void print(const uint8_t * /* buffer */, std::ostream &) const = 0;
  virtual ~MetadataPrinter() = default;
};

class MetadataPrinterRegistry
{
public:
  MetadataPrinterRegistry();

public:
  const MetadataPrinter *lookup(std::string op) const
  {
    if (_metadata_map.find(op) == _metadata_map.end())
      return nullptr;

    return _metadata_map.at(op).get();
  }

public:
  static MetadataPrinterRegistry &get()
  {
    static MetadataPrinterRegistry me;
    return me;
  }

private:
  std::map<std::string /* metadata name */, std::unique_ptr<MetadataPrinter>> _metadata_map;
};

} // namespace circledump

#endif // __CIRCLEDUMP_METADATA_PRINTER_H__
