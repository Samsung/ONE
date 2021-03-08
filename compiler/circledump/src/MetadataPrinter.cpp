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

#include "MetadataPrinter.h"

#include <cassert>
#include <string>
#include <vector>

namespace circledump
{

class SourceTablePrinter : public MetadataPrinter
{
public:
  /**
   *  source table consists of following parts
   *  - [ entry_number : uint32_t ]
   *  - [ id : uint32_t ][ length : uint32_t ][ data : 'length' Bytes ] * entry_number
   */
  virtual void print(const uint8_t *buffer, std::ostream &os) const override
  {
    if (buffer)
    {
      os << "    [node_id : node_name]" << std::endl;
      auto cur = buffer;
      // entry number
      const uint32_t num = *reinterpret_cast<const uint32_t *>(cur);
      cur += sizeof(uint32_t);
      for (uint32_t entry = 0; entry < num; entry++)
      {
        // id
        const uint32_t node_id = *reinterpret_cast<const uint32_t *>(cur);
        cur += sizeof(uint32_t);
        // length
        const uint32_t len = *reinterpret_cast<const uint32_t *>(cur);
        cur += sizeof(uint32_t);
        assert(len != 0);
        // data
        // non-empty 'data' has trailing '\0'. Let's exclude it.
        std::string node_name = std::string(cur, cur + len - 1);
        cur += len;

        // print
        os << "    [" << node_id << " : " << node_name << "]" << std::endl;
      }
    }
  }
};

class OpTablePrinter : public MetadataPrinter
{
public:
  /**
   *  op table consists of following parts
   *  - [ entry_number : uint32_t ]
   *  - [ id : uint32_t ][ length : uint32_t ][ origin_ids : length * uint32_t ] * entry_number
   */
  virtual void print(const uint8_t *buffer, std::ostream &os) const override
  {
    if (buffer)
    {
      os << "    [node_id : origin_ids]" << std::endl;
      auto cur = buffer;
      // entry number
      const uint32_t num = *reinterpret_cast<const uint32_t *>(cur);
      cur += sizeof(uint32_t);
      for (uint32_t entry = 0; entry < num; entry++)
      {
        // id
        const uint32_t node_id = *reinterpret_cast<const uint32_t *>(cur);
        cur += sizeof(uint32_t);
        // length
        const uint32_t len = *reinterpret_cast<const uint32_t *>(cur);
        cur += sizeof(uint32_t);
        assert(len != 0);
        // origin_ids
        std::vector<uint32_t> origin_ids;
        for (uint32_t o = 0; o < len; o++)
        {
          origin_ids.push_back(*reinterpret_cast<const uint32_t *>(cur));
          cur += sizeof(uint32_t);
        }

        // print
        os << "    [" << node_id << " : ";
        uint32_t i = 0;
        for (const auto &id : origin_ids)
        {
          if (i++)
            os << ", ";
          os << id;
        }
        os << "]" << std::endl;
      }
    }
  }
};

MetadataPrinterRegistry::MetadataPrinterRegistry()
{
  _metadata_map["ONE_source_table"] = std::make_unique<SourceTablePrinter>();
  _metadata_map["ONE_op_table"] = std::make_unique<OpTablePrinter>();
}

} // namespace circledump
