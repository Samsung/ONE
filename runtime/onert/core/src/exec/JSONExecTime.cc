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

#include "exec/JSONExecTime.h"
#include "backend/IConfig.h"
#include <fstream>

namespace onert
{
namespace exec
{
/**
 * @brief Helper function for reading string from stream
 *
 * @param str Output string
 * @param stream File stream
 */
void readString(std::string &str, std::ifstream &stream)
{
  str.clear();
  char buf;
  while (stream.good())
  {
    stream.get(buf);
    if (buf == '"')
      break;
    str.push_back(buf);
  }
}

/**
 * @brief Helper function for reading bool from stream
 *
 * @param quant Output bool
 * @param stream File stream
 */
void readBool(bool &quant, std::ifstream &stream)
{
  char buf;
  stream.get(buf);
  quant = (buf == '1');
  stream.get(buf);
}

void printString(const std::string &str, std::ofstream &stream) { stream << "\"" << str << "\""; }

void printBool(bool quant, std::ofstream &stream) { stream << "\"" << quant << "\""; }

void JSON::readOperation(const std::string &backend, const std::string &operation, bool quant,
                         std::ifstream &stream)
{
  uint32_t size = 0;
  int64_t time = 0;

  std::string int_buf;
  char buf;
  int number_of_closed_braces = 0;
  int number_of_commas = 0;

  while (stream.good())
  {
    stream.get(buf);

    switch (buf)
    {
      case ']':
      {
        number_of_closed_braces++;
        break;
      }
      case '[':
      {
        number_of_closed_braces--;
        break;
      }
      default:
      {
        if (std::isdigit(buf))
        {
          int_buf.push_back(buf);
        }
        break;
      }
    }

    if (number_of_closed_braces == 1)
      break;

    if ((buf == ']' && number_of_closed_braces == 0) ||
        (buf == ',' && number_of_closed_braces == -1))
    {
      switch (number_of_commas % 2)
      {
        case 0:
        {
          size = static_cast<uint32_t>(std::atoi(int_buf.c_str()));
          break;
        }
        case 1:
        {
          time = static_cast<int64_t>(std::atol(int_buf.c_str()));
          auto bf = _backends.find(backend);
          if (bf != _backends.end())
          {
            _measurements[bf->second][operation][quant][size] = time;
          } // we ignore the records for unsupported backends
          break;
        }
      }
      number_of_commas++;
      int_buf.clear();
    }
  }
}
void JSON::printOperation(const std::map<uint32_t, int64_t> &operation_info,
                          std::ofstream &stream) const
{
  for (const auto &items : operation_info)
  {
    stream << "[" << items.first << ", " << items.second << "], ";
  }
  stream.seekp(-2, std::ofstream::end);
}

void JSON::storeOperationsExecTime() const
{
  std::ofstream stream(_measurement_file);
  if (!stream.is_open())
  {
    throw std::runtime_error("Failed to save backend config file");
  }
  else
  {
    stream << "{";
    for (const auto &backend : _measurements)
    {
      printString(backend.first->config()->id(), stream);
      stream << ": {";
      for (const auto &operation : backend.second)
      {
        printString(operation.first, stream);
        stream << ": {";
        for (const auto &type : operation.second)
        {
          printBool(type.first, stream);
          stream << ": [";
          printOperation(type.second, stream);
          stream << "], ";
        }
        stream.seekp(-2, std::ofstream::end);
        stream << "}, ";
      }
      stream.seekp(-2, std::ofstream::end);
      stream << "}, ";
    }
    stream.seekp(-2, std::ofstream::end);
    stream << "}";
    stream.close();
  }
}

void JSON::loadOperationsExecTime()
{
  std::ifstream stream(_measurement_file);
  if (stream.is_open())
  {
    std::string backend;
    std::string operation;
    bool quant = false;
    char buf;
    int number_of_open_braces = 0;

    while (stream.good())
    {
      stream.get(buf);
      switch (buf)
      {
        case '{':
          number_of_open_braces++;
          break;
        case '}':
          number_of_open_braces--;
          break;
        case '"':
        {
          if (number_of_open_braces == 1)
          {
            // read backend string
            readString(backend, stream);
          }
          if (number_of_open_braces == 2)
          {
            // read operation string
            readString(operation, stream);
          }
          if (number_of_open_braces == 3)
          {
            // read operation string
            readBool(quant, stream);
          }
          break;
        }
        case '[':
        {
          // reading and creating all info for operation
          readOperation(backend, operation, quant, stream);
          break;
        }
        default:
          break;
      }
    }
    stream.close();
  }
}

} // namespace exec
} // namespace onert
