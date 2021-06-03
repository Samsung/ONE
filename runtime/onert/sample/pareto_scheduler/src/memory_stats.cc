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
#include <fstream>
#include <limits>
#include <unordered_map>
#include "memory_stats.h"

std::unordered_map<memtype_e, std::string> memtype_map = {
  {MEM_FREE, "MemFree:"}, {MEM_AVAILABLE, "MemAvailable:"}, {MEM_TOTAL, "MemTotal:"}};

void process_mem_usage(double &vm_usage, double &resident_set)
{
  using std::ios_base;
  using std::ifstream;
  using std::string;

  vm_usage = 0.0;
  resident_set = 0.0;

  // 'file' stat seems to give the most reliable results
  //
  ifstream stat_stream("/proc/self/stat", ios_base::in);

  // dummy vars for leading entries in stat that we don't care about
  //
  string pid, comm, state, ppid, pgrp, session, tty_nr;
  string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  string utime, stime, cutime, cstime, priority, nice;
  string O, itrealvalue, starttime;

  // the two fields we want
  //
  unsigned long vsize;
  long rss;

  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >> tpgid >> flags >>
    minflt >> cminflt >> majflt >> cmajflt >> utime >> stime >> cutime >> cstime >> priority >>
    nice >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

  stat_stream.close();

  //  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB
  //  pages
  vm_usage = vsize / 1024.0;
  resident_set = rss;
}

unsigned long get_meminfo(memtype_e memtype)
{
  std::string token;
  std::string mem_field;
  std::ifstream file("/proc/meminfo");
  if (memtype_map.find(memtype) == memtype_map.end())
  {
    return 0;
  }
  while (file >> token)
  {
    if (token == memtype_map[memtype])
    {
      unsigned long mem;
      if (file >> mem)
      {
        return mem;
      }
      else
      {
        return 0;
      }
    }
    // ignore rest of the line
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return 0; // nothing found
}
