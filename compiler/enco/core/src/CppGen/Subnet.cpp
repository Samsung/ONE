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

#include "CppGen/Subnet.h"

#include "Dims.h"
#include "String.h"

#include <pp/LinearDocument.h>

#include <memory>
#include <sstream>

using std::make_unique;
using enco::concat;

#define S(content) #content

namespace ann
{
static std::ostream &operator<<(std::ostream &os, const ann::OperandID &id)
{
  os << id.value();
  return os;
}
} // namespace ann

namespace
{

class SubnetStructImpl final : public enco::SubnetStruct
{
public:
  SubnetStructImpl() : _dtor{pp::LinearDocument::Direction::Reverse}
  {
    // DO NOTHING
  }

public:
  std::string model(void) const override { return "_model"; }
  std::string compilation(void) const override { return "_compilation"; }

public:
  const pp::MultiLineText &def(void) const override { return _def; }
  pp::LinearDocument *def(void) { return &_def; }

public:
  const pp::MultiLineText &ctor(void) const override { return _ctor; }
  pp::LinearDocument *ctor(void) { return &_ctor; }

public:
  const pp::MultiLineText &dtor(void) const override { return _dtor; }
  pp::LinearDocument *dtor(void) { return &_dtor; }

private:
  pp::LinearDocument _def;
  pp::LinearDocument _ctor;
  pp::LinearDocument _dtor;
};

struct CodeFragment
{
  virtual ~CodeFragment() = default;

  virtual void dump(pp::LinearDocument *) const = 0;
};

pp::LinearDocument *operator<<(pp::LinearDocument *doc, const CodeFragment &fragment)
{
  fragment.dump(doc);
  return doc;
}

const char *scalar_operand_code(const ann::DType &dtype)
{
  switch (dtype)
  {
    case ann::DType::S32:
      return "ANEURALNETWORKS_INT32";
    default:
      break;
  };

  throw std::invalid_argument("dtype");
}

const char *tensor_operand_code(const ann::DType &dtype)
{
  switch (dtype)
  {
    case ann::DType::S32:
      return "ANEURALNETWORKS_TENSOR_INT32";
    case ann::DType::F32:
      return "ANEURALNETWORKS_TENSOR_FLOAT32";
    default:
      break;
  };

  throw std::invalid_argument("dtype");
}

class ScalarOperandDecl final : public CodeFragment
{
public:
  ScalarOperandDecl(const std::string &model, const ann::DType &dtype)
    : _model{model}, _dtype{dtype}
  {
    // DO NOTHING
  }

public:
  void dump(pp::LinearDocument *doc) const override
  {
    doc->append("{");
    doc->indent();
    doc->append("ANeuralNetworksOperandType t;");
    doc->append();
    doc->append("t.type = ", scalar_operand_code(_dtype), ";");
    doc->append("t.dimensionCount = 0;");
    doc->append("t.dimensions = nullptr;");
    doc->append("t.scale = 1.0f;");
    doc->append("t.zeroPoint = 0;");
    doc->append();
    doc->append("ANeuralNetworksModel_addOperand(", _model, ", &t);");
    doc->unindent();
    doc->append("}");
  }

private:
  std::string _model;
  ann::DType _dtype;
};

class TensorOperandDecl final : public CodeFragment
{
public:
  TensorOperandDecl(const std::string &model, const ann::DType &dtype,
                    const nncc::core::ADT::tensor::Shape &shape)
    : _model{model}, _dtype{dtype}, _shape{shape}
  {
    // DO NOTHING
  }

public:
  void dump(pp::LinearDocument *doc) const override
  {
    const auto rank = _shape.rank();
    const auto dims = as_dims(_shape);

    assert(rank == dims.size());

    doc->append("{");
    doc->indent();
    doc->append("uint32_t d[", rank, "] = { ", concat(", ", dims.begin(), dims.end()), " };");
    doc->append();
    doc->append("ANeuralNetworksOperandType t;");
    doc->append();
    doc->append("t.type = ", tensor_operand_code(_dtype), ";");
    doc->append("t.dimensionCount = ", rank, ";");
    doc->append("t.dimensions = d;");
    doc->append("t.scale = 1.0f;");
    doc->append("t.zeroPoint = 0;");
    doc->append();
    doc->append("ANeuralNetworksModel_addOperand(", _model, ", &t);");
    doc->unindent();
    doc->append("}");
  }

private:
  std::string _model;
  ann::DType _dtype;
  nncc::core::ADT::tensor::Shape _shape;
};

/**
 * @brief Code fragment that calls ANeuralNetworksModel_setOperandValue
 */
class WeightDecl final : public CodeFragment
{
public:
  WeightDecl(const std::string &model, const ann::OperandID &id, const std::string &base,
             const std::string &size)
    : _model{model}, _id{id}, _base{base}, _size{size}
  {
    // DO NOTHING
  }

public:
  void dump(pp::LinearDocument *doc) const override
  {
    doc->append("ANeuralNetworksModel_setOperandValue(", _model, ", ", _id.value(), ", ", _base,
                ", ", _size, ");");
  }

private:
  std::string _model;
  ann::OperandID _id;
  std::string _base;
  std::string _size;
};

/**
 * @brief Code fragment that calls ANeuralNetworksModel_addOperation
 */
class OperationDecl final : public CodeFragment
{
public:
  OperationDecl(const std::string &model, const ann::Operation *op) : _model{model}, _op{op}
  {
    // DO NOTHING
  }

private:
  static std::string opcode(const ann::Operation::Code &code)
  {
    switch (code)
    {
#define ANN_OPERATION(TAG, ENUM)  \
  case ann::Operation::Code::TAG: \
    return #ENUM;
#include "ANN/IR/Operation.def"
#undef ANN_OPERATION
      default:
        throw std::invalid_argument{"code"};
    };
  }

public:
  void dump(pp::LinearDocument *doc) const override
  {
    const auto in_count = _op->inputs().size();
    auto in_beg = _op->inputs().begin();
    auto in_end = _op->inputs().end();

    const auto out_count = _op->outputs().size();
    auto out_beg = _op->outputs().begin();
    auto out_end = _op->outputs().end();

    auto op = opcode(_op->code());

    doc->append("{");
    doc->indent();
    doc->append("uint32_t inputs[", in_count, "] = { ", concat(", ", in_beg, in_end), " };");
    doc->append("uint32_t outputs[", out_count, "] = { ", concat(", ", out_beg, out_end), " };");
    doc->append();
    doc->append("ANeuralNetworksModel_addOperation(", _model, ", ", op, ", ", in_count,
                ", inputs, ", out_count, ", outputs);");
    doc->unindent();
    doc->append("}");
  }

private:
  std::string _model;
  const ann::Operation *_op;
};

/**
 * @brief Code fragment that calls ANeuralNetworksModel_identifyInputsAndOutputs
 */
class ArgumentDecl final : public CodeFragment
{
public:
  ArgumentDecl(const std::string &mname, const ANNBinder *binder) : _mname{mname}, _binder{binder}
  {
    // DO NOTHING
  }

public:
  void dump(pp::LinearDocument *doc) const override
  {
    doc->append("{");
    doc->indent();

    auto module = _binder->module();
    const uint32_t input_count = module->input()->size();

    doc->append("uint32_t inputs[", input_count, "];");
    for (uint32_t n = 0; n < input_count; ++n)
    {
      doc->append("inputs[", n, "] = ", module->input()->at(n), ";");
    }

    const uint32_t output_count = module->output()->size();

    doc->append("uint32_t outputs[", output_count, "];");
    for (uint32_t n = 0; n < output_count; ++n)
    {
      doc->append("outputs[", n, "] = ", module->output()->at(n), ";");
    }

    doc->append("ANeuralNetworksModel_identifyInputsAndOutputs(", _mname, ", ", input_count,
                ", inputs, ", output_count, ", outputs);");
    doc->unindent();
    doc->append("}");
  }

private:
  std::string _mname;
  const ANNBinder *_binder;
};

} // namespace

namespace enco
{

std::unique_ptr<SubnetStruct> SubnetStructBuilder::build(const ANNBinder *binder) const
{
  auto res = make_unique<SubnetStructImpl>();

  auto mname = res->model();
  auto cname = res->compilation();

  res->def()->append("ANeuralNetworksModel *", mname, ";");
  res->def()->append("ANeuralNetworksCompilation *", cname, ";");

  res->ctor()->append("ANeuralNetworksModel_create(&", mname, ");");
  res->dtor()->append("ANeuralNetworksModel_free(", mname, ");");

  binder->module()->operand()->each([&](const ann::OperandID &id, const ann::Operand *info) {
    // TODO Remove dynamic cast
    if (auto scalar = dynamic_cast<const ann::ScalarOperand *>(info))
    {
      res->ctor() << ScalarOperandDecl{mname, scalar->dtype()};
    }
    else if (auto tensor = dynamic_cast<const ann::TensorOperand *>(info))
    {
      res->ctor() << TensorOperandDecl{mname, tensor->dtype(), tensor->shape()};
    }
    else
    {
      throw std::runtime_error{"Unsupported"};
    }

    if (_weighted.find(info) != _weighted.end())
    {
      const auto &base_exp = _base_exprs.at(info);
      const auto &size_exp = _size_exprs.at(info);

      res->ctor() << WeightDecl{mname, id, base_exp, size_exp};
    }
  });

  for (unsigned n = 0; n < binder->module()->operation()->count(); ++n)
  {
    auto op = binder->module()->operation()->at(n);
    res->ctor() << OperationDecl{mname, op};
  }

  // Emit ANeuralNetworksModel_identifyInputsAndOutputs call
  res->ctor() << ArgumentDecl{mname, binder};

  // Emit ANeuralNetworksModel_finish call
  res->ctor()->append("ANeuralNetworksModel_finish(", mname, ");");

  // Create compilation
  res->ctor()->append("ANeuralNetworksCompilation_create(", mname, ", &", cname, ");");
  res->dtor()->append("ANeuralNetworksCompilation_free(", cname, ");");

  // Finalize compilation
  res->ctor()->append("ANeuralNetworksCompilation_finish(", cname, ");");

  return std::move(res);
}

std::unique_ptr<pp::MultiLineText> SubnetBlockCompiler::compile(const ANNBinder *binder) const
{
  auto res = make_unique<pp::LinearDocument>();

  const auto compilation = _compilation_ctx.at(binder);

  res->append("ANeuralNetworksExecution *execution;");
  res->append("ANeuralNetworksEvent *event;");
  res->append();
  res->append("ANeuralNetworksExecution_create(", compilation, ", &execution);");

  // Emit ANeuralNetworksExecution_setInput call(s)
  for (uint32_t n = 0; n < binder->module()->input()->size(); ++n)
  {
    auto bag = binder->input(n);
    auto base = _mem.base(bag);
    auto size = _mem.size(bag);

    res->append("ANeuralNetworksExecution_setInput(execution, ", n, ", nullptr, ", base, ", ", size,
                ");");
  }

  // Emit ANeuralNetworksExecution_setOutput call(s)
  for (uint32_t n = 0; n < binder->module()->output()->size(); ++n)
  {
    auto bag = binder->output(n);
    auto base = _mem.base(bag);
    auto size = _mem.size(bag);

    res->append("ANeuralNetworksExecution_setOutput(execution, ", n, ", nullptr, ", base, ", ",
                size, ");");
  }

  res->append("ANeuralNetworksExecution_startCompute(execution, &event);");
  res->append("ANeuralNetworksEvent_wait(event);");
  res->append("ANeuralNetworksEvent_free(event);");

  res->append("ANeuralNetworksExecution_free(execution);");

  return std::move(res);
}

} // namespace enco
