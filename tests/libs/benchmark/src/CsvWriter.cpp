/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "benchmark/CsvWriter.h"
#include <cassert>

namespace
{

const std::vector<std::string> csv_header{
#include "benchmark/CsvHeader.lst"
};

} // namespace

namespace benchmark
{

CsvWriter::CsvWriter(const std::string &csv_filename) : CsvWriter(csv_filename, csv_header)
{
  // DO NOTHING
}

CsvWriter::CsvWriter(const std::string &csv_filename, const std::vector<std::string> &header)
  : _ofs(csv_filename), _header_size(header.size()), _col_idx(0), _row_idx(0)
{
  assert(csv_filename.empty() == false);
  assert(header.size() != 0);
  assert(_ofs.is_open());

  writeHeader(header);
}

CsvWriter::~CsvWriter()
{
  if (_ofs.is_open())
    _ofs.close();
}

void CsvWriter::writeHeader(const std::vector<std::string> &header)
{
  for (const auto &col : header)
    write(col);
}

void CsvWriter::postWrite()
{
  if (++_col_idx == _header_size)
  {
    _ofs << newline;
    _row_idx += 1;
    _col_idx = 0;
  }
  else
  {
    _ofs << delimiter;
  }
}

void CsvWriter::write(const std::string &val)
{
  _ofs << val;
  postWrite();
}

void CsvWriter::write(double val)
{
  _ofs << val;
  postWrite();
}

void CsvWriter::write(uint32_t val)
{
  _ofs << val;
  postWrite();
}

void CsvWriter::write(char val)
{
  _ofs << val;
  postWrite();
}

bool CsvWriter::done() { return (_col_idx == 0) && (_row_idx == 2); }

CsvWriter &operator<<(CsvWriter &csvw, const std::string &val)
{
  csvw.write(val);
  return csvw;
}

CsvWriter &operator<<(CsvWriter &csvw, double val)
{
  csvw.write(val);
  return csvw;
}

CsvWriter &operator<<(CsvWriter &csvw, uint32_t val)
{
  csvw.write(val);
  return csvw;
}

CsvWriter &operator<<(CsvWriter &csvw, char val)
{
  csvw.write(val);
  return csvw;
}

} // namespace benchmark
