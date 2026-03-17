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

#include "GlobalDataGeneration.h"
#include "Split.h"
#include "Dims.h"

#include <memory>
#include <map>

using std::make_unique;

namespace
{

/**
 * @brief Manage global variable declarations
 */
class Global
{
public:
  Global(std::ostream &os) : _os(os)
  {
    // DO NOTHING
  }

public:
  /// @brief Create a global constant string (const char *) literal, and return variable name
  enco::GlobalOffset constant(const std::string &value);

  /// @brief Create a global constant array variable of type T
  template <typename T> enco::GlobalOffset constant(const std::vector<T> &values);

  /// @brief Create a global constant array variable of byte (uint8_t) type
  enco::GlobalOffset constant(const uint8_t *base, uint32_t size);

private:
  uint32_t _offset = 0;
  std::ostream &_os;
};

enco::GlobalOffset Global::constant(const std::string &s)
{
  auto const base = reinterpret_cast<const uint8_t *>(s.c_str());
  auto const size = s.size() + 1 /* NUL */;
  return constant(base, size);
}

template <> enco::GlobalOffset Global::constant(const std::vector<uint32_t> &values)
{
  auto const base = reinterpret_cast<const uint8_t *>(values.data());
  auto const size = sizeof(uint32_t) * values.size();
  return constant(base, size);
}

enco::GlobalOffset Global::constant(const uint8_t *base, uint32_t size)
{
  auto pos = _os.tellp();
  assert(pos != -1);

  _os.write(reinterpret_cast<const char *>(base), size);

  return static_cast<enco::GlobalOffset>(pos);
}

} // namespace

namespace
{

std::map<const ann::Operand *, enco::GlobalOffset> data_offset_ctx;
std::map<const coco::Bag *, enco::GlobalOffset> bag_data_offset_ctx;

std::map<const coco::Arg *, enco::GlobalOffset> name_offset_ctx;
std::map<const coco::Arg *, enco::GlobalOffset> dims_offset_ctx;

} // namespace

namespace enco
{

GlobalOffset GlobalData::data_offset(const ann::Operand *o) { return data_offset_ctx.at(o); }

GlobalOffset GlobalData::data_offset(const coco::Bag *bag)
{
  assert(bag_data_offset_ctx.find(bag) != bag_data_offset_ctx.end());
  return bag_data_offset_ctx.at(bag);
}

GlobalOffset GlobalData::name_offset(const coco::Input *in) { return name_offset_ctx.at(in); }
GlobalOffset GlobalData::dims_offset(const coco::Input *in) { return dims_offset_ctx.at(in); }

GlobalOffset GlobalData::name_offset(const coco::Output *out) { return name_offset_ctx.at(out); }
GlobalOffset GlobalData::dims_offset(const coco::Output *out) { return dims_offset_ctx.at(out); }

void generate_global_data(std::ostream &os, enco::Code *code)
{
  auto m = code->module();
  auto d = code->data();

  auto ann_ctx = enco::SubnetManager::context(m);

  auto global = make_unique<Global>(os);

  //
  // Emit Bag's weight
  //
  for (uint32_t n = 0; n < m->entity()->bag()->size(); ++n)
  {
    auto bag = m->entity()->bag()->at(n);

    if (!d->allocated(bag))
    {
      // Skip if the weight value does not exist for a given bag
      continue;
    }

    // NOTE The current implementation assumes that all the values are of float(fp32) type
    // TODO Support non-float values
    auto span = d->f32()->weight(bag);

    assert(span.data() != nullptr);
    assert(span.size() > 0);

    auto const base = reinterpret_cast<const uint8_t *>(span.data());
    uint32_t const size = span.size() * sizeof(float);

    assert(bag_data_offset_ctx.find(bag) == bag_data_offset_ctx.end());
    bag_data_offset_ctx[bag] = global->constant(base, size);
  }

  for (uint32_t n = 0; n < ann_ctx->count(); ++n)
  {
    auto binder = ann_ctx->nth(n);

    auto emit = [&](const ann::OperandID & /*id*/, const ann::Operand *info) {
      if (info->weight())
      {
        auto base = info->weight()->base();
        auto size = info->weight()->size();

        data_offset_ctx[info] = global->constant(base, size);
      }
    };
    binder->module()->operand()->each(emit);
  }

  for (uint32_t n = 0; n < m->input()->size(); ++n)
  {
    auto input = m->input()->at(n);
    auto dims = as_dims(input->shape());

    name_offset_ctx[input] = global->constant(input->name());
    dims_offset_ctx[input] = global->constant<uint32_t>(dims);
  }

  for (uint32_t n = 0; n < m->output()->size(); ++n)
  {
    auto output = m->output()->at(n);
    auto dims = as_dims(output->shape());

    name_offset_ctx[output] = global->constant(output->name());
    dims_offset_ctx[output] = global->constant<uint32_t>(dims);
  }
}

} // namespace enco
