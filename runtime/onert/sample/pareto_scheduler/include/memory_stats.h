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
/**
 * @file  memory_stats.h
 * @brief This file describes API for fetching memory information
 */

#ifndef _PARETO_MEMSTATS_H
#define _PARETO_MEMSTATS_H
#include <sys/time.h>
#include <sys/resource.h>
enum memtype_e
{
  MEM_FREE,
  MEM_AVAILABLE,
  MEM_TOTAL
};

/**
 * @brief     Get the current process' RSS and VM memory usage. All units are in number of pages.
 * @param[in, out] double& vm_usage. VM stat information provided here
 * @param[in, out] double& resident_set. RSS memory information provided here
 */
void process_mem_usage(double &vm_usage, double &resident_set);
/**
 * @brief     Get the system memory information in MB.
 * @param[in] memtype_e memtype. Allowed memory types are MEM_FREE, MEM_AVAILABLE, and MEM_TOTAL
 *
 * @return    unsigned long. Memory information (free, available or total) in MB.
 */
unsigned long get_meminfo(memtype_e memtype);
#endif
