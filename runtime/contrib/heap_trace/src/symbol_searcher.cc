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

#include "symbol_searcher.h"

#include <dlfcn.h>
#include <link.h>

struct SymbolDescription
{
  const char *name;
  void *address = nullptr; // address in memory where this symbol can be found

  SymbolDescription(const char *name) : name(name) {}
};

using InfoAboutLoadedLib = struct dl_phdr_info *;

static void tryToFindSymbolInLinkedLibraries(SymbolDescription &symbol);
static void tryToFindSymbolInAllLoadedLibraries(SymbolDescription &symbol);
static int checkIfLibraryContainsSymbol(InfoAboutLoadedLib library_description, size_t /* size */,
                                        void *data);
static bool isSymbolAddressNotInTheSameTranslationUnit(SymbolDescription *symbol);
void *findSymbol(const char *name)
{
  signalizeThatNextAllocationsWillBeForSymbolSearcherInternalUsage();
  SymbolDescription symbol(name);
  tryToFindSymbolInLinkedLibraries(symbol);
  if (!symbol.address)
  {
    tryToFindSymbolInAllLoadedLibraries(symbol);
  }
  signalizeThatSymbolSearcherEndedOfWork();
  return symbol.address;
}

static void tryToFindSymbolInLinkedLibraries(SymbolDescription &symbol)
{
  symbol.address = dlsym(RTLD_NEXT, symbol.name);
}

static void tryToFindSymbolInAllLoadedLibraries(SymbolDescription &symbol)
{
  dl_iterate_phdr(checkIfLibraryContainsSymbol, &symbol);
}

static int checkIfLibraryContainsSymbol(InfoAboutLoadedLib library_description, size_t /* size */,
                                        void *data)
{
  SymbolDescription *symbol = (SymbolDescription *)data;

  void *handle = dlopen(library_description->dlpi_name, RTLD_NOW);
  symbol->address = dlsym(handle, symbol->name);
  dlclose(handle);
  if (symbol->address && isSymbolAddressNotInTheSameTranslationUnit(symbol))
  {
    return 1;
  }
  return 0;
}

static bool isSymbolAddressNotInTheSameTranslationUnit(SymbolDescription *symbol)
{
  void *handle = dlopen("", RTLD_NOW);
  void *addressInTheSameTranslationUnit = dlsym(handle, symbol->name);
  dlclose(handle);

  return addressInTheSameTranslationUnit == nullptr ||
         addressInTheSameTranslationUnit != symbol->address;
}

// TODO should be thread_local (or all symbols should be resolved at the start of application as
// alternative)
static volatile bool are_next_allocations_will_be_for_symbol_searcher_internal_usage = false;

void signalizeThatNextAllocationsWillBeForSymbolSearcherInternalUsage()
{
  are_next_allocations_will_be_for_symbol_searcher_internal_usage = true;
}

void signalizeThatSymbolSearcherEndedOfWork()
{
  are_next_allocations_will_be_for_symbol_searcher_internal_usage = false;
}

bool isCurrentAllocationForSymbolSearcherInternalUsage()
{
  return are_next_allocations_will_be_for_symbol_searcher_internal_usage;
}
