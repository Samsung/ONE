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

#include "CppCode.h"

#include "Transforms/GlobalDataGeneration.h"
#include "Transforms/Split.h"

#include "CppGen/MemoryContext.h"

#include "CppGen/Host.h"
#include "CppGen/Subnet.h"

#include "Dims.h"

#include <pp/LinearDocument.h>
#include <pp/MultiLineTextUtils.h>

#include <map>
#include <set>
#include <string>
#include <stdexcept>

namespace
{

struct SubnetInfo
{
  std::string struct_name;
  /// @brief The field name (in this subnet struct) of ANeuralNetworksCompilation value
  std::string compilation_field;

  /// @brief The field name (in Network struct) for this subnet
  std::string field_name;
};

struct NetworkStruct
{
  pp::LinearDocument def;
};

struct InvokeFunction
{
  pp::LinearDocument head;
  pp::LinearDocument body;
  pp::LinearDocument tail{pp::LinearDocument::Direction::Reverse};

public:
  /** @brief Create a (fresh) local variable */
  std::string local(void) { return pp::fmt("v_", ++_var_count); }

private:
  uint32_t _var_count = 0;
};

/**
 * @brief Enumerate a set of Bag accessed by a given instruction
 *
 * Supported instruction:
 *    "Shuffle"
 */
class AccessedBagAccumulator : public coco::Instr::Visitor<void>
{
public:
  AccessedBagAccumulator(std::set<coco::Bag *> *out) : _out{out}
  {
    // Validate "out"
    assert(_out != nullptr);
  }

public:
  void visit(const coco::Shuffle *shuffle) override
  {
    assert(shuffle->from() != nullptr);
    assert(shuffle->into() != nullptr);

    _out->insert(shuffle->from());
    _out->insert(shuffle->into());
  }

private:
  std::set<coco::Bag *> *_out;
};

/**
 * @brief Return a set of bags that SHOULD have a host allocation
 */
std::set<coco::Bag *> hosted(const enco::Code *code)
{
  std::set<coco::Bag *> res;

  auto m = code->module();
  auto ann_ctx = enco::SubnetManager::context(m);

  for (auto blk = m->block()->head(); blk; blk = blk->next())
  {
    if (auto ann_binder = ann_ctx->find(blk))
    {
      // Case: The current block is ANN-compatible

      // Each ANN input SHOULD have a corresponding host allocation
      for (uint32_t n = 0; n < ann_binder->module()->input()->size(); ++n)
      {
        res.insert(ann_binder->input(n));
      }

      // Each ANN output SHOULD have a corresponding host allocation
      for (uint32_t n = 0; n < ann_binder->module()->output()->size(); ++n)
      {
        res.insert(ann_binder->output(n));
      }
    }
    else
    {
      // Every bag that ANN-incompatible block accesses SHOULD have a corresponding host allocation
      AccessedBagAccumulator acc{&res};

      for (auto ins = blk->instr()->head(); ins; ins = ins->next())
      {
        ins->accept(acc);
      }
    }
  }

  return res;
}
} // namespace

namespace enco
{

void CppCode::dump(std::ostream &os) const
{
  auto m = _code->module();
  auto d = _code->data();
  auto ann_ctx = enco::SubnetManager::context(m);

  NetworkStruct network;
  InvokeFunction invoke;
  pp::LinearDocument internal;

  auto data_exp = [this](const GlobalOffset &off) { return pp::fmt(_varname, " + ", off); };

  // Record the subnet information
  std::map<const ANNBinder *, SubnetInfo> subnet_ctx;

  /**
   * Create a struct for each android NN network of the following form:
   *
   * struct [Name]
   * {
   *   ...
   *
   *   [Name]() // constructor
   *   {
   *     ...
   *   }
   *
   *   ~[Name]() // destructor
   *   {
   *     ...
   *   }
   * };
   *
   */
  for (uint32_t n = 0; n < ann_ctx->count(); ++n)
  {
    SubnetStructBuilder builder;

    auto subnet_binder = ann_ctx->nth(n);
    auto subnet_struct_name = pp::fmt("Subnet_", subnet_ctx.size());
    auto subnet_field_name = pp::fmt("_subnet_", subnet_ctx.size());

    // Create global data variable
    auto emit_weight = [&](const ann::OperandID &, const ann::Operand *info) {
      if (info->weight())
      {
        auto size = info->weight()->size();
        auto off = enco::GlobalData::data_offset(info);
        auto base_exp = pp::fmt("reinterpret_cast<const void *>(", data_exp(off), ")");
        auto size_exp = pp::fmt(size);

        builder.expr(info, base_exp, size_exp);
      }
    };
    subnet_binder->module()->operand()->each(emit_weight);

    auto subnet_struct_content = builder.build(subnet_binder);

    // Emit C++ declaration
    internal.append("struct ", subnet_struct_name);
    internal.append("{");
    internal.indent();

    internal.append(subnet_struct_content->def());

    internal.append(subnet_struct_name, "()");
    internal.append("{");
    internal.indent();
    internal.append(subnet_struct_content->ctor());
    internal.unindent();
    internal.append("}");

    internal.append("~", subnet_struct_name, "()");
    internal.append("{");
    internal.indent();
    internal.append(subnet_struct_content->dtor());
    internal.unindent();
    internal.append("}");

    internal.unindent();
    internal.append("};");

    // Declare subnet field
    network.def.append(subnet_struct_name, " ", subnet_field_name, ";");

    // Update subnet context
    SubnetInfo subnet_info;

    subnet_info.struct_name = subnet_struct_name;
    subnet_info.compilation_field = subnet_struct_content->compilation();
    subnet_info.field_name = subnet_field_name;

    assert(subnet_ctx.find(subnet_binder) == subnet_ctx.end());
    subnet_ctx[subnet_binder] = subnet_info;
  }

  MemoryContext mem;

  // Set dedicated memory region for network inputs
  for (uint32_t n = 0; n < m->input()->size(); ++n)
  {
    mem.base(m->input()->at(n)->bag(), pp::fmt("net->inputs[", n, "].ptr"));
    mem.size(m->input()->at(n)->bag(), pp::fmt("net->inputs[", n, "].len"));
  }

  // Set dedicated memory region for network outputs
  for (uint32_t n = 0; n < m->output()->size(); ++n)
  {
    mem.base(m->output()->at(n)->bag(), pp::fmt("net->outputs[", n, "].ptr"));
    mem.size(m->output()->at(n)->bag(), pp::fmt("net->outputs[", n, "].len"));
  }

  // Set dedicated memory region for constant weight values
  // TODO Support non-constant bags with initial values
  for (uint32_t n = 0; n < m->entity()->bag()->size(); ++n)
  {
    auto bag = m->entity()->bag()->at(n);

    if (!d->allocated(bag))
    {
      // Skip if no weight exists
      continue;
    }

    // TODO Support non-float(fp32) weight
    auto offset = enco::GlobalData::data_offset(bag);

    auto base_expr = data_exp(offset);
    auto size_expr = pp::fmt(bag->size() * sizeof(float));

    mem.base(bag, base_expr);
    mem.size(bag, size_expr);
  }

  // Set dedicated memory reigion for intermediate buffer(s)
  for (const auto &bag : hosted(_code))
  {
    // Skip if a bag is already allocated
    if (mem.member(bag))
    {
      continue;
    }

    auto name = invoke.local();

    invoke.head.append("auto ", name, " = new uint8_t[", bag->size() * sizeof(float), "];");
    invoke.tail.append("delete[] ", name, ";");

    mem.base(bag, name);
    mem.size(bag, pp::fmt(bag->size() * sizeof(float)));
  }

  // Create Code Block Builder
  SubnetBlockCompiler subnet_compiler{mem};

  for (auto it = subnet_ctx.begin(); it != subnet_ctx.end(); ++it)
  {
    // Specify how to access ANeuralNetworksCompilation
    const auto &info = it->second;
    subnet_compiler.bind(it->first, pp::fmt("net->", info.field_name, ".", info.compilation_field));
  }

  HostBlockCompiler host_compiler{mem};

  for (auto blk = m->block()->head(); blk; blk = blk->next())
  {
    invoke.body.append("{");
    invoke.body.indent();

    if (auto binder = ann_ctx->find(blk))
    {
      // Generate code that invokes Android NN sub-network
      auto lines = subnet_compiler.compile(binder);
      invoke.body.append(*lines);
    }
    else
    {
      // Generate code on-the-fly for Android NN-incompatible blocks
      auto lines = host_compiler.compile(blk);
      invoke.body.append(*lines);
    }

    invoke.body.unindent();
    invoke.body.append("}");
  }

  //
  // Generate full C++ source code with code snippet
  //
  const std::string name{"Network"};

  pp::LinearDocument includes;
  {
    // Include Android NN API header
    includes.append("#include <NeuralNetworks.h>");
    includes.append();

    includes.append("#include <cstdint>");
    includes.append("#include <cassert>");
    includes.append("#include <array>");
  }

  pp::LinearDocument net_def;
  {
    net_def.append("struct ", name, " {");
    net_def.indent();
    net_def.append("struct Shape { uint32_t rank; const uint32_t *dims; };");
    net_def.append("struct Input {");
    net_def.indent();
    net_def.append("const char *name;");
    net_def.append("const uint8_t *ptr;");
    net_def.append("unsigned len;");
    net_def.append("Shape shape;");
    net_def.unindent();
    net_def.append("};");
    net_def.append("struct Output {");
    net_def.indent();
    net_def.append("const char *name;");
    net_def.append("uint8_t *ptr;");
    net_def.append("unsigned len;");
    net_def.append("Shape shape;");
    net_def.unindent();
    net_def.append("};");
    net_def.append();
    net_def.append(name, "();");
    net_def.append("~", name, "();");

    net_def.append();
    net_def.append(network.def);
    net_def.append();

    net_def.append("std::array<Input, ", m->input()->size(), "> inputs;");
    net_def.append("std::array<Output, ", m->output()->size(), "> outputs;");

    net_def.unindent();
    net_def.append("};");
  }

  pp::LinearDocument net_ctor;
  {
    net_ctor.append("Network::Network() {");
    net_ctor.indent();

    // Initialize input metadata
    for (uint32_t n = 0; n < m->input()->size(); ++n)
    {
      auto input = m->input()->at(n);
      auto dims = as_dims(input->shape());

      auto name_off = enco::GlobalData::name_offset(input);
      auto name_exp = pp::fmt("reinterpret_cast<const char *>(", data_exp(name_off), ")");
      auto dims_off = enco::GlobalData::dims_offset(input);
      auto dims_exp = pp::fmt("reinterpret_cast<const unsigned *>(", data_exp(dims_off), ")");

      net_ctor.append("inputs.at(", n, ").name = ", name_exp, ";");
      net_ctor.append("inputs.at(", n, ").shape.rank = ", dims.size(), ";");
      net_ctor.append("inputs.at(", n, ").shape.dims = ", dims_exp, ";");
    }

    // Initialize output metadata
    for (uint32_t n = 0; n < m->output()->size(); ++n)
    {
      auto output = m->output()->at(n);
      auto dims = as_dims(output->shape());

      auto name_off = enco::GlobalData::name_offset(output);
      auto name_exp = pp::fmt("reinterpret_cast<const char *>(", data_exp(name_off), ")");
      auto dims_off = enco::GlobalData::dims_offset(output);
      auto dims_exp = pp::fmt("reinterpret_cast<const unsigned *>(", data_exp(dims_off), ")");

      net_ctor.append("outputs.at(", n, ").name = ", name_exp, ";");
      net_ctor.append("outputs.at(", n, ").shape.rank = ", dims.size(), ";");
      net_ctor.append("outputs.at(", n, ").shape.dims = ", dims_exp, ";");
    }

    // TODO Implement this
    net_ctor.unindent();
    net_ctor.append("}");
  }

  pp::LinearDocument net_dtor;
  {
    net_dtor.append("Network::~Network() {");
    net_dtor.indent();
    // TODO Implement this
    net_dtor.unindent();
    net_dtor.append("}");
  }

  pp::LinearDocument source;

  source.append(includes);
  source.append();
  source.append("extern uint8_t ", _varname, "[];");
  source.append();

  source.append("namespace");
  source.append("{");
  source.append(internal);
  source.append("} // namespace");
  source.append();
  source.append(net_def);
  source.append();
  source.append(net_ctor);
  source.append();
  source.append(net_dtor);

  source.append();
  source.append(name, " *", name, "_construct() { return new ", name, "{}; }");
  source.append("void ", name, "_destruct(", name, " *net) { delete net; }");

  source.append();

  // Emit Network_input_count function
  source.append("unsigned ", name, "_input_count(const ", name, " *net) {");
  source.indent();
  source.append("return net->inputs.size();");
  source.unindent();
  source.append("}");

  source.append();

  // Emit Network_input_name function
  source.append("const char *", name, "_input_name(const ", name, " *net, unsigned n) {");
  source.indent();
  source.append("return net->inputs.at(n).name;");
  source.unindent();
  source.append("}");

  // Emit Network_input_rank function
  source.append("unsigned ", name, "_input_rank(const ", name, " *net, unsigned n) {");
  source.indent();
  source.append("return net->inputs.at(n).shape.rank;");
  source.unindent();
  source.append("}");

  // Emit Network_input_dim function
  source.append("unsigned ", name, "_input_dim(const ", name, " *net, unsigned n, unsigned axe)");
  source.append("{");
  source.indent();
  source.append("return net->inputs.at(n).shape.dims[axe];");
  source.unindent();
  source.append("}");

  // Emit Network_input_bind function
  source.append("void ", name, "_input_bind(", name,
                " *net, unsigned n, const void *ptr, unsigned len) {");
  source.indent();
  source.append("net->inputs.at(n).ptr = reinterpret_cast<const uint8_t *>(ptr);");
  source.append("net->inputs.at(n).len = len;");
  source.unindent();
  source.append("}");

  source.append();

  // Emit Network_output_count function
  source.append("unsigned ", name, "_output_count(const ", name, " *net) {");
  source.indent();
  source.append("return net->outputs.size();");
  source.unindent();
  source.append("}");

  source.append();

  // Emit Network_output_name function
  source.append("const char *", name, "_output_name(const ", name, " *net, unsigned n) {");
  source.indent();
  source.append("return net->outputs.at(n).name;");
  source.unindent();
  source.append("}");

  // Emit Network_output_rank function
  source.append("unsigned ", name, "_output_rank(const ", name, " *net, unsigned n) {");
  source.indent();
  source.append("return net->outputs.at(n).shape.rank;");
  source.unindent();
  source.append("}");

  // Emit Network_output_dim function
  source.append("unsigned ", name, "_output_dim(const ", name, " *net, unsigned n, unsigned axe)");
  source.append("{");
  source.indent();
  source.append("return net->outputs.at(n).shape.dims[axe];");
  source.unindent();
  source.append("}");

  // Emit Network_output_bind function
  source.append("void ", name, "_output_bind(", name,
                " *net, unsigned n, void *ptr, unsigned len) {");
  source.indent();
  source.append("net->outputs.at(n).ptr = reinterpret_cast<uint8_t *>(ptr);");
  source.append("net->outputs.at(n).len = len;");
  source.unindent();
  source.append("}");

  source.append();

  source.append("void ", name, "_invoke(", name, " *net) {");
  source.indent();
  source.append(invoke.head);
  source.append(invoke.body);
  source.append(invoke.tail);
  source.unindent();
  source.append("}");

  os << source;
}

} // namespace enco
