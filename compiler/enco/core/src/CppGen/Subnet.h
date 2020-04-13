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

#ifndef __ENCO_CPP_GEN_SUBNET_H__
#define __ENCO_CPP_GEN_SUBNET_H__

#include "ANN/Binder.h"
#include "CppGen/MemoryContext.h"

#include <pp/MultiLineText.h>
#include <map>
#include <set>

namespace enco
{

/**
 * @brief A C++ struct that provides Android NN model & compilation
 */
struct SubnetStruct
{
  virtual ~SubnetStruct() = default;

  /// @brief Return the field name of ANeuralNetworksModel value
  virtual std::string model(void) const = 0;
  /// @brief Return the field name of ANeuralNetworksCompilatoin value
  virtual std::string compilation(void) const = 0;

  virtual const pp::MultiLineText &def(void) const = 0;
  virtual const pp::MultiLineText &ctor(void) const = 0;
  virtual const pp::MultiLineText &dtor(void) const = 0;
};

class SubnetStructBuilder
{
public:
  std::unique_ptr<SubnetStruct> build(const ANNBinder *binder) const;

public:
  void expr(const ann::Operand *oper, const std::string &base, const std::string &size)
  {
    _weighted.insert(oper);
    _base_exprs[oper] = base;
    _size_exprs[oper] = size;
  }

private:
  std::set<const ann::Operand *> _weighted;
  std::map<const ann::Operand *, std::string> _base_exprs;
  std::map<const ann::Operand *, std::string> _size_exprs;
};

/**
 * @brief Generate C++ code that invokes Android NN subnet
 */
class SubnetBlockCompiler
{
public:
  SubnetBlockCompiler(const enco::MemoryContext &mem) : _mem(mem)
  {
    // DO NOTHING
  }

public:
  /// @brief Specify how to access ANeuralNetworksCompilation value (C expression)
  void bind(const ANNBinder *binder, const std::string &exp) { _compilation_ctx[binder] = exp; }

public:
  std::unique_ptr<pp::MultiLineText> compile(const ANNBinder *binder) const;

private:
  const enco::MemoryContext &_mem;
  std::map<const ANNBinder *, std::string> _compilation_ctx;
};

} // namespace enco

#endif // __ENCO_CPP_GEN_SUBNET_H__
