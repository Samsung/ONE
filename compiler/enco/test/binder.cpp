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

//
// Generated API
//
struct Network;

Network *Network_construct();
void Network_destruct(Network *net);

unsigned Network_input_count(const Network *);
const char *Network_input_name(const Network *, unsigned n);
unsigned Network_input_rank(const Network *, unsigned n);
unsigned Network_input_dim(const Network *, unsigned n, unsigned axis);
void Network_input_bind(Network *net, unsigned n, const void *ptr, unsigned len);

unsigned Network_output_count(const Network *net);
const char *Network_output_name(const Network *, unsigned n);
unsigned Network_output_rank(const Network *, unsigned n);
unsigned Network_output_dim(const Network *, unsigned n, unsigned axis);
void Network_output_bind(Network *net, unsigned n, void *ptr, unsigned len);

void Network_invoke(Network *net);

//
// nnkit backend
//
#include <nnkit/Backend.h>
#include <nnkit/TensorContext.h>
#include <nnkit/CmdlineArguments.h>

#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/Overlay.h>

#include <memory>

using std::make_unique;
using namespace nncc::core::ADT;

namespace
{

class TensorContext final : public nnkit::TensorContext
{
public:
  TensorContext() = default;

public:
  void allocate(const std::string &name, const tensor::Shape &shape)
  {
    using nncc::core::ADT::tensor::num_elements;

    auto blob = make_unique<std::vector<uint8_t>>();
    blob->resize(num_elements(shape) * sizeof(float));

    _names.emplace_back(name);
    _shapes.emplace_back(shape);
    _blobs.emplace_back(std::move(blob));
  }

public:
  uint8_t *base(uint32_t n) const { return _blobs.at(n)->data(); }

public:
  uint32_t size(void) const override { return _blobs.size(); }

public:
  std::string name(uint32_t n) const override { return _names.at(n); }

public:
  tensor::Shape shape(uint32_t n) const override { return _shapes.at(n); }

public:
  uint32_t size(uint32_t n) const { return _blobs.at(n)->size(); }

public:
  // Float (fp32) tensor support
  bool isFloatTensor(uint32_t n) const override { return true; }
  void getMutableFloatTensor(uint32_t n, const TensorContext::TypedAccessor<float> &f) override
  {
    using nncc::core::ADT::tensor::LexicalLayout;
    using nncc::core::ADT::tensor::make_overlay;

    auto base = reinterpret_cast<float *>(this->base(n));
    auto view = make_overlay<float, LexicalLayout>(shape(n), base);

    f(*this, n, view);
  }

  void getConstFloatTensor(uint32_t n, const TensorContext::TypedReader<float> &f) const override
  {
    using nncc::core::ADT::tensor::LexicalLayout;
    using nncc::core::ADT::tensor::make_overlay;

    auto base = reinterpret_cast<float *>(this->base(n));
    auto view = make_overlay<float, LexicalLayout>(shape(n), base);

    f(*this, n, view);
  }

private:
  std::vector<std::string> _names;
  std::vector<tensor::Shape> _shapes;
  std::vector<std::unique_ptr<std::vector<uint8_t>>> _blobs;
};

class Backend final : public nnkit::Backend
{
public:
  Backend()
  {
    _net = Network_construct();

    // Allocate and bind inputs
    for (uint32_t n = 0; n < Network_input_count(_net); ++n)
    {
      const uint32_t rank = Network_input_rank(_net, n);
      const std::string name = Network_input_name(_net, n);

      tensor::Shape shape;

      shape.resize(rank);
      for (uint32_t axis = 0; axis < rank; ++axis)
      {
        shape.dim(axis) = Network_input_dim(_net, n, axis);
      }

      _inputs.allocate(name, shape);

      Network_input_bind(_net, n, reinterpret_cast<const void *>(_inputs.base(n)), _inputs.size(n));
    }

    // Allocate and bind outputs
    for (uint32_t n = 0; n < Network_output_count(_net); ++n)
    {
      const uint32_t rank = Network_output_rank(_net, n);
      const std::string name = Network_output_name(_net, n);

      tensor::Shape shape;

      shape.resize(rank);
      for (uint32_t axis = 0; axis < rank; ++axis)
      {
        shape.dim(axis) = Network_output_dim(_net, n, axis);
      }

      _outputs.allocate(name, shape);

      Network_output_bind(_net, n, reinterpret_cast<void *>(_outputs.base(n)), _outputs.size(n));
    }
  }

public:
  ~Backend() { Network_destruct(_net); }

public:
  void prepare(const std::function<void(nnkit::TensorContext &)> &f) override { f(_inputs); }
  void run(void) override { Network_invoke(_net); }
  void teardown(const std::function<void(nnkit::TensorContext &)> &f) override { f(_outputs); }

private:
  Network *_net;

private:
  TensorContext _inputs;
  TensorContext _outputs;
};

} // namespace

extern "C" std::unique_ptr<nnkit::Backend> make_backend(const nnkit::CmdlineArguments &args)
{
  return make_unique<::Backend>();
}
