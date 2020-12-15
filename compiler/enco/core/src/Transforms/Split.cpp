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

#include "Split.h"
#include "Usage.h"
#include "Session.h"
#include "coex/IR.h"

#include <coco/IR.h>

#include <nncc/core/ADT/kernel/NHWCLayout.h>
#include <stdex/Memory.h>

#include <map>
#include <stdexcept>
#include <functional>

using stdex::make_unique;

namespace
{

std::map<const coco::Module *, std::unique_ptr<ANNContext>> _subnet_contexts;

} // namespace

namespace enco
{

const ANNContext *SubnetManager::context(const coco::Module *m)
{
  return _subnet_contexts.at(m).get();
}

} // namespace enco

namespace
{

using Appender = std::function<void(ANNBinder *binder)>;

struct ANNOpAppender
{
  virtual ~ANNOpAppender() = default;

  virtual void append(ANNBinder *binder) const = 0;
};

class ANNAddAppender final : public ANNOpAppender
{
public:
  void left(coco::FeatureObject *o) { _left = o; }
  void right(coco::FeatureObject *o) { _right = o; }
  void out(coco::FeatureObject *o) { _out = o; }

public:
  void append(ANNBinder *binder) const override
  {
    auto left = binder->addOperand<float>(_left);
    auto right = binder->addOperand<float>(_right);
    auto fuse = binder->addOperand<int32_t>();
    binder->setOperand(fuse, 0);

    auto out = binder->addOperand<float>(_out);

    binder->addOperation(ann::Operation::Code::ADD, {left, right, fuse}, {out});
  }

private:
  coco::FeatureObject *_left = nullptr;
  coco::FeatureObject *_right = nullptr;
  coco::FeatureObject *_out = nullptr;
};

class ANNMulAppender final : public ANNOpAppender
{
public:
  void left(coco::FeatureObject *o) { _left = o; }
  void right(coco::FeatureObject *o) { _right = o; }
  void out(coco::FeatureObject *o) { _out = o; }

public:
  void append(ANNBinder *binder) const override
  {
    auto left = binder->addOperand<float>(_left);
    auto right = binder->addOperand<float>(_right);
    auto fuse = binder->addOperand<int32_t>();
    binder->setOperand(fuse, 0);

    auto out = binder->addOperand<float>(_out);

    binder->addOperation(ann::Operation::Code::MUL, {left, right, fuse}, {out});
  }

private:
  coco::FeatureObject *_left = nullptr;
  coco::FeatureObject *_right = nullptr;
  coco::FeatureObject *_out = nullptr;
};

/**
 * WARN The current implementation supports concatenation along depth only
 */
class ANNConcatAppender final : public ANNOpAppender
{
public:
  void left(coco::FeatureObject *o) { _left = o; }
  void right(coco::FeatureObject *o) { _right = o; }
  void out(coco::FeatureObject *o) { _out = o; }

public:
  void append(ANNBinder *binder) const override
  {
    auto left = binder->addOperand<float>(_left);
    auto right = binder->addOperand<float>(_right);
    auto axis = binder->addOperand<int32_t>();
    binder->setOperand(axis, 3 /* DEPTH */);

    auto out = binder->addOperand<float>(_out);

    binder->addOperation(ann::Operation::Code::CONCAT, {left, right, axis}, {out});
  }

private:
  coco::FeatureObject *_left = nullptr;
  coco::FeatureObject *_right = nullptr;
  coco::FeatureObject *_out = nullptr;
};

class ANNConv2DAppender final : public ANNOpAppender
{
public:
  void session(const enco::SessionID &sess) { _sess = sess; }

  void pad(const coco::Padding2D *pad) { _pad = *pad; }
  void stride(const coco::Stride2D *stride) { _stride = *stride; }

  void ifm(coco::FeatureObject *ifm) { _ifm = ifm; }
  void ker(coco::KernelObject *ker) { _ker = ker; }
  // Q: Should we take a bias as a feature object?
  // NOTE This interface is subject to change
  void bias(coco::FeatureObject *bias) { _bias = bias; }
  void ofm(coco::FeatureObject *ofm) { _ofm = ofm; }

public:
  void append(ANNBinder *binder) const override
  {
    auto data = enco::data(_sess);

    auto ifm = binder->addOperand<float>(_ifm);
    auto ker = binder->addOperand<float>(_ker);

    // Fill kernel data
    {
      auto ker_bag = _ker->bag();
      auto ker_weight = data->f32()->weight(ker_bag);

      assert(ker_weight.data() != nullptr);

      binder->setOperand(ker, ker_weight.data(), ker_weight.data() + ker_weight.size());
    }

    // Conv2D in coco IR has no bias, but bias is mandatory in Android NN API
    auto bias = binder->addOperand<float>(nncc::core::ADT::tensor::Shape{_ker->shape().count()});

    // Fill bias data
    if (_bias == nullptr)
    {
      // Use a fresh empty bias if "bias" is not specified
      auto length = _ker->shape().count();

      std::vector<float> values;
      values.resize(length, 0.0f);

      binder->setOperand(bias, values.begin(), values.end());
    }
    else
    {
      // Use specified "bias"
      auto bias_bag = _bias->bag();
      auto bias_weight = data->f32()->weight(bias_bag);

      assert(bias_weight.data() != nullptr);
      assert(bias_weight.size() == _ker->shape().count());

      binder->setOperand(bias, bias_weight.data(), bias_weight.data() + bias_weight.size());
    }

    auto left = binder->addOperand<int32_t>();
    binder->setOperand(left, _pad.left());
    auto right = binder->addOperand<int32_t>();
    binder->setOperand(right, _pad.right());
    auto top = binder->addOperand<int32_t>();
    binder->setOperand(top, _pad.top());
    auto bottom = binder->addOperand<int32_t>();
    binder->setOperand(bottom, _pad.bottom());
    auto hstride = binder->addOperand<int32_t>();
    binder->setOperand(hstride, _stride.horizontal());
    auto vstride = binder->addOperand<int32_t>();
    binder->setOperand(vstride, _stride.vertical());
    auto fuse = binder->addOperand<int32_t>();
    binder->setOperand(fuse, 0);

    auto ofm = binder->addOperand<float>(_ofm);

    binder->addOperation(ann::Operation::Code::CONV_2D,
                         {ifm, ker, bias, left, right, top, bottom, hstride, vstride, fuse}, {ofm});
  }

private:
  enco::SessionID _sess;

private:
  coco::Padding2D _pad;
  coco::Stride2D _stride;

private:
  coco::FeatureObject *_ifm = nullptr;
  coco::KernelObject *_ker = nullptr;
  coco::FeatureObject *_bias = nullptr;
  coco::FeatureObject *_ofm = nullptr;
};

class ANNDepthwiseConv2DAppender final : public ANNOpAppender
{
public:
  void session(const enco::SessionID &sess) { _sess = sess; }

  void multiplier(const uint32_t &multiplier) { _multiplier = multiplier; }
  void pad(const coco::Padding2D *pad) { _pad = *pad; }
  void stride(const coco::Stride2D *stride) { _stride = *stride; }

  void ifm(coco::FeatureObject *ifm) { _ifm = ifm; }
  void ker(coco::KernelObject *ker) { _ker = ker; }
  void ofm(coco::FeatureObject *ofm) { _ofm = ofm; }

public:
  void append(ANNBinder *binder) const override
  {
    using namespace nncc::core::ADT;

    auto data = enco::data(_sess);

    const uint32_t ker_N = _ker->shape().count();
    const uint32_t ker_H = _ker->shape().height();
    const uint32_t ker_W = _ker->shape().width();

    assert(ker_N % _multiplier == 0);
    const uint32_t group = ker_N / _multiplier;

    auto ifm = binder->addOperand<float>(_ifm);
    auto ker = binder->addOperand<float>(tensor::Shape{1, ker_H, ker_W, ker_N});

    // Fill kernel data
    {
      auto obj = _ker;
      auto shape = obj->shape();

      auto ovl = data->f32()->read(obj);
      assert(ovl != nullptr);

      // Flatten?
      std::vector<float> values;

      /**
       * Android NN computes DEPTHWISE_CONV_2D as follows:
       *
       * output[b, i, j, k * channel_multiplier + q] =
       *    sum_{di, dj} (
       *        input[b, strides[1] * i + di, strides[2] * j + dj, k] *
       *        filter[1, di, dj, k * channel_multiplier + q]
       *    ) + bias[k * channel_multiplier + q]
       *
       */
      for (uint32_t row = 0; row < shape.height(); ++row)
      {
        for (uint32_t col = 0; col < shape.width(); ++col)
        {
          for (uint32_t g = 0; g < group; ++g)
          {
            for (uint32_t m = 0; m < _multiplier; ++m)
            {
              const auto value = ovl->at(g * _multiplier + m, 0, row, col);
              values.emplace_back(value);
            }
          }
        }
      }

      assert(values.size() == nncc::core::ADT::kernel::num_elements(shape));
      binder->setOperand(ker, values.begin(), values.end());
    }

    // Conv2D in coco IR has no bias, but bias is mandatory in Android NN API
    auto bias = binder->addOperand<float>(nncc::core::ADT::tensor::Shape{_ker->shape().count()});

    // Fill bias data
    {
      auto length = _ker->shape().count();

      std::vector<float> values;
      values.resize(length, 0.0f);

      binder->setOperand(bias, values.begin(), values.end());
    }

    auto left = binder->addOperand<int32_t>();
    binder->setOperand(left, _pad.left());
    auto right = binder->addOperand<int32_t>();
    binder->setOperand(right, _pad.right());
    auto top = binder->addOperand<int32_t>();
    binder->setOperand(top, _pad.top());
    auto bottom = binder->addOperand<int32_t>();
    binder->setOperand(bottom, _pad.bottom());
    auto hstride = binder->addOperand<int32_t>();
    binder->setOperand(hstride, _stride.horizontal());
    auto vstride = binder->addOperand<int32_t>();
    binder->setOperand(vstride, _stride.vertical());
    auto multiplier = binder->addOperand<int32_t>();
    binder->setOperand(multiplier, _multiplier);
    auto fuse = binder->addOperand<int32_t>();
    binder->setOperand(fuse, 0);

    auto ofm = binder->addOperand<float>(_ofm);

    binder->addOperation(
      ann::Operation::Code::DEPTHWISE_CONV_2D,
      {ifm, ker, bias, left, right, top, bottom, hstride, vstride, multiplier, fuse}, {ofm});
  }

private:
  enco::SessionID _sess;

private:
  uint32_t _multiplier;
  coco::Padding2D _pad;
  coco::Stride2D _stride;

private:
  coco::FeatureObject *_ifm = nullptr;
  coco::KernelObject *_ker = nullptr;
  coco::FeatureObject *_ofm = nullptr;
};

class ANNReLUAppender final : public ANNOpAppender
{
public:
  void ifm(coco::FeatureObject *ifm) { _ifm = ifm; }
  void ofm(coco::FeatureObject *ofm) { _ofm = ofm; }

public:
  void append(ANNBinder *binder) const override
  {
    auto ifm = binder->addOperand<float>(_ifm);
    auto ofm = binder->addOperand<float>(_ofm);

    binder->addOperation(ann::Operation::Code::RELU, {ifm}, {ofm});
  }

private:
  coco::FeatureObject *_ifm = nullptr;
  coco::FeatureObject *_ofm = nullptr;
};

class ANNReLU6Appender final : public ANNOpAppender
{
public:
  void ifm(coco::FeatureObject *ifm) { _ifm = ifm; }
  void ofm(coco::FeatureObject *ofm) { _ofm = ofm; }

public:
  void append(ANNBinder *binder) const override
  {
    auto ifm = binder->addOperand<float>(_ifm);
    auto ofm = binder->addOperand<float>(_ofm);

    binder->addOperation(ann::Operation::Code::RELU6, {ifm}, {ofm});
  }

private:
  coco::FeatureObject *_ifm = nullptr;
  coco::FeatureObject *_ofm = nullptr;
};

class ANNMaxPool2DAppender final : public ANNOpAppender
{
public:
  void pad(const coco::Padding2D *pad) { _pad = *pad; }
  void stride(const coco::Stride2D *stride) { _stride = *stride; }
  void window(const coco::Window2D *window) { _window = *window; }

  void ifm(coco::FeatureObject *ifm) { _ifm = ifm; }
  void ofm(coco::FeatureObject *ofm) { _ofm = ofm; }

public:
  void append(ANNBinder *binder) const override
  {
    auto ifm = binder->addOperand<float>(_ifm);

    // Set padding
    auto left = binder->addOperand<int32_t>();
    binder->setOperand(left, _pad.left());
    auto right = binder->addOperand<int32_t>();
    binder->setOperand(right, _pad.right());
    auto top = binder->addOperand<int32_t>();
    binder->setOperand(top, _pad.top());
    auto bottom = binder->addOperand<int32_t>();
    binder->setOperand(bottom, _pad.bottom());

    // Set horizontal/vertical stride
    auto hstride = binder->addOperand<int32_t>();
    binder->setOperand(hstride, _stride.horizontal());
    auto vstride = binder->addOperand<int32_t>();
    binder->setOperand(vstride, _stride.vertical());

    // Set receptive field size
    auto width = binder->addOperand<int32_t>();
    binder->setOperand(width, _window.width());
    auto height = binder->addOperand<int32_t>();
    binder->setOperand(height, _window.height());

    // Set fuse code
    // TODO Suport operation fusion
    auto fuse = binder->addOperand<int32_t>();
    binder->setOperand(fuse, 0);

    auto ofm = binder->addOperand<float>(_ofm);

    binder->addOperation(ann::Operation::Code::MAX_POOL_2D,
                         {ifm, left, right, top, bottom, hstride, vstride, width, height, fuse},
                         {ofm});
  }

private:
  coco::Padding2D _pad;
  coco::Stride2D _stride;
  coco::Window2D _window;

private:
  coco::FeatureObject *_ifm = nullptr;
  coco::FeatureObject *_ofm = nullptr;
};

class ANNAvgPool2DAppender final : public ANNOpAppender
{
public:
  void pad(const coco::Padding2D *pad) { _pad = *pad; }
  void stride(const coco::Stride2D *stride) { _stride = *stride; }
  void window(const coco::Window2D *window) { _window = *window; }

  void ifm(coco::FeatureObject *ifm) { _ifm = ifm; }
  void ofm(coco::FeatureObject *ofm) { _ofm = ofm; }

public:
  void append(ANNBinder *binder) const override
  {
    auto ifm = binder->addOperand<float>(_ifm);

    // Set padding
    auto left = binder->addOperand<int32_t>();
    binder->setOperand(left, _pad.left());
    auto right = binder->addOperand<int32_t>();
    binder->setOperand(right, _pad.right());
    auto top = binder->addOperand<int32_t>();
    binder->setOperand(top, _pad.top());
    auto bottom = binder->addOperand<int32_t>();
    binder->setOperand(bottom, _pad.bottom());

    // Set horizontal/vertical stride
    auto hstride = binder->addOperand<int32_t>();
    binder->setOperand(hstride, _stride.horizontal());
    auto vstride = binder->addOperand<int32_t>();
    binder->setOperand(vstride, _stride.vertical());

    // Set receptive field size
    auto width = binder->addOperand<int32_t>();
    binder->setOperand(width, _window.width());
    auto height = binder->addOperand<int32_t>();
    binder->setOperand(height, _window.height());

    // Set fuse code
    // TODO Suport operation fusion
    auto fuse = binder->addOperand<int32_t>();
    binder->setOperand(fuse, 0);

    auto ofm = binder->addOperand<float>(_ofm);

    binder->addOperation(ann::Operation::Code::AVG_POOL_2D,
                         {ifm, left, right, top, bottom, hstride, vstride, width, height, fuse},
                         {ofm});
  }

private:
  coco::Padding2D _pad;
  coco::Stride2D _stride;
  coco::Window2D _window;

private:
  coco::FeatureObject *_ifm = nullptr;
  coco::FeatureObject *_ofm = nullptr;
};

class ANNPadFAppender final : public ANNOpAppender
{
public:
  void pad(const coco::Padding2D *pad) { _pad = *pad; }

public:
  void ifm(coco::FeatureObject *ifm) { _ifm = ifm; }
  void ofm(coco::FeatureObject *ofm) { _ofm = ofm; }

public:
  void append(ANNBinder *binder) const override
  {
    using nncc::core::ADT::tensor::Shape;

    auto ifm = binder->addOperand<float>(_ifm);
    auto pad = binder->addOperand<int32_t>(Shape{4, 2});
    {
      std::vector<int32_t> values;
      values.resize(8);
      // For 'N'
      values.at(0) = values.at(1) = 0;
      // For 'H'
      values.at(2) = _pad.top();
      values.at(3) = _pad.bottom();
      // For 'W'
      values.at(4) = _pad.left();
      values.at(5) = _pad.right();
      // For 'C'
      values.at(6) = values.at(7) = 0;

      binder->setOperand(pad, values.begin(), values.end());
    }

    auto ofm = binder->addOperand<float>(_ofm);

    binder->addOperation(ann::Operation::Code::PAD, {ifm, pad}, {ofm});
  }

private:
  coco::Padding2D _pad;

private:
  coco::FeatureObject *_ifm = nullptr;
  coco::FeatureObject *_ofm = nullptr;
};

class ANNOpFunctionalAppender final : public ANNOpAppender
{
public:
  ANNOpFunctionalAppender(const Appender &fun) : _fun{fun}
  {
    // DO NOTHING
  }

public:
  void append(ANNBinder *binder) const { _fun(binder); }

private:
  Appender _fun;
};

class ANNSubAppender final : public ANNOpAppender
{
public:
  void left(coco::FeatureObject *o) { _left = o; }
  void right(coco::FeatureObject *o) { _right = o; }
  void out(coco::FeatureObject *o) { _out = o; }

public:
  void append(ANNBinder *binder) const override
  {
    auto left = binder->addOperand<float>(_left);
    auto right = binder->addOperand<float>(_right);
    auto fuse = binder->addOperand<int32_t>();
    binder->setOperand(fuse, 0);

    auto out = binder->addOperand<float>(_out);

    binder->addOperation(ann::Operation::Code::SUB, {left, right, fuse}, {out});
  }

private:
  coco::FeatureObject *_left = nullptr;
  coco::FeatureObject *_right = nullptr;
  coco::FeatureObject *_out = nullptr;
};

class ANNDivAppender final : public ANNOpAppender
{
public:
  void left(coco::FeatureObject *o) { _left = o; }
  void right(coco::FeatureObject *o) { _right = o; }
  void out(coco::FeatureObject *o) { _out = o; }

public:
  void append(ANNBinder *binder) const override
  {
    auto left = binder->addOperand<float>(_left);
    auto right = binder->addOperand<float>(_right);
    auto fuse = binder->addOperand<int32_t>();
    binder->setOperand(fuse, 0);

    auto out = binder->addOperand<float>(_out);

    binder->addOperation(ann::Operation::Code::DIV, {left, right, fuse}, {out});
  }

private:
  coco::FeatureObject *_left = nullptr;
  coco::FeatureObject *_right = nullptr;
  coco::FeatureObject *_out = nullptr;
};

class ANNOpBuilder : public coco::Instr::Visitor<std::unique_ptr<ANNOpAppender>>
{
public:
  std::unique_ptr<ANNOpAppender> visit(const coco::Eval *eval)
  {
    if (auto conv = eval->op()->asConv2D())
    {
      if (auto load = conv->arg()->asLoad())
      {
        auto sess = enco::session(eval->module());

        auto ifm = load->object()->asFeature();
        auto ker = conv->ker();
        auto ofm = eval->out()->asFeature();

        const auto group = conv->group();

        if (group == 1)
        {
          auto app = make_unique<ANNConv2DAppender>();

          app->session(sess);

          app->pad(conv->pad());
          app->stride(conv->stride());

          app->ifm(ifm);
          app->ofm(ofm);
          app->ker(ker);

          return std::move(app);
        }
        else
        {
          assert(ifm->shape().depth() == group);
          assert(ker->shape().count() % group == 0);
          assert(ker->shape().depth() == 1);

          auto app = make_unique<ANNDepthwiseConv2DAppender>();

          app->session(sess);

          app->multiplier(ker->shape().count() / group);
          app->pad(conv->pad());
          app->stride(conv->stride());

          app->ifm(ifm);
          app->ofm(ofm);
          app->ker(ker);

          return std::move(app);
        }
      }
    }
    else if (auto op = eval->op()->asAdd())
    {
      auto left_load = op->left()->asLoad();
      auto right_load = op->right()->asLoad();

      if (left_load && right_load)
      {
        // Let's compile the following code fragment:
        //
        //   %ofm = eval(Add(Load(%left), Load(%right)))
        //
        auto left = left_load->object()->asFeature();
        auto right = right_load->object()->asFeature();
        assert(left != nullptr && right != nullptr);

        auto out = eval->out()->asFeature();
        assert(out != nullptr);

        auto app = make_unique<ANNAddAppender>();

        app->left(left);
        app->right(right);
        app->out(out);

        return std::move(app);
      }
    }
    else if (auto op = eval->op()->asMul())
    {
      auto left_load = op->left()->asLoad();
      auto right_load = op->right()->asLoad();

      if (left_load && right_load)
      {
        // Let's compile the following code fragment:
        //
        //   %ofm = eval(Mul(Load(%left), Load(%right)))
        //
        auto left = left_load->object()->asFeature();
        auto right = right_load->object()->asFeature();
        assert(left != nullptr && right != nullptr);

        auto out = eval->out()->asFeature();
        assert(out != nullptr);

        auto app = make_unique<ANNMulAppender>();

        app->left(left);
        app->right(right);
        app->out(out);

        return std::move(app);
      }
    }
    else if (auto op = eval->op()->asPadF())
    {
      if (auto load = op->arg()->asLoad())
      {
        // Let's compile the following code fragment:
        //
        //   %ofm = eval(PadF(Load(%ifm))
        //
        auto ifm = load->object()->asFeature();
        auto ofm = eval->out()->asFeature();

        assert(ifm != nullptr && ofm != nullptr);

        auto app = make_unique<ANNPadFAppender>();

        app->pad(op->pad());

        app->ifm(ifm);
        app->ofm(ofm);

        return std::move(app);
      }
    }
    else if (auto maxpool = eval->op()->asMaxPool2D())
    {
      if (auto load = maxpool->arg()->asLoad())
      {
        // Let's compile the following code fragment:
        //
        //   %ofm = eval(MaxPool2D(Load(%ifm))
        //
        auto ifm = load->object()->asFeature();
        auto ofm = eval->out()->asFeature();

        assert(ifm != nullptr && ofm != nullptr);

        auto app = make_unique<ANNMaxPool2DAppender>();

        app->pad(maxpool->pad());
        app->stride(maxpool->stride());
        app->window(maxpool->window());

        app->ifm(ifm);
        app->ofm(ofm);

        return std::move(app);
      }
    }
    else if (auto avgpool = eval->op()->asAvgPool2D())
    {
      if (auto load = avgpool->arg()->asLoad())
      {
        // Let's compile the following code fragment:
        //
        //   %ofm = eval(AvgPool2D(Load(%ifm))
        //
        if (avgpool->divisor() == coco::AvgPool2D::Divisor::PaddingExcluded)
        {
          // When ANN runtime computes the average of each receptive field,
          // it uses the number of valid(=non-padding) elements as a divisor.
          auto ifm = load->object()->asFeature();
          auto ofm = eval->out()->asFeature();

          assert(ifm != nullptr && ofm != nullptr);

          auto app = make_unique<ANNAvgPool2DAppender>();

          app->pad(avgpool->pad());
          app->stride(avgpool->stride());
          app->window(avgpool->window());

          app->ifm(ifm);
          app->ofm(ofm);

          return std::move(app);
        }
      }
    }
    else if (auto relu = eval->op()->asReLU())
    {
      if (auto load = relu->arg()->asLoad())
      {
        // Let's compile the following code fragment:
        //
        //   %ofm = eval(ReLU(Load(%ifm))
        //
        // TODO Support objects of other kinds, such as Tensor
        auto ifm = load->object()->asFeature();
        auto ofm = eval->out()->asFeature();

        assert(ifm != nullptr && ofm != nullptr);

        auto app = make_unique<ANNReLUAppender>();

        app->ifm(ifm);
        app->ofm(ofm);

        return std::move(app);
      }
    }
    else if (auto relu6 = eval->op()->asReLU6())
    {
      if (auto load = relu6->arg()->asLoad())
      {
        // Let's compile the following code fragment:
        //
        //   %ofm = eval(ReLU6(Load(%ifm))
        //
        // TODO Support objects of other kinds, such as Tensor
        auto ifm = load->object()->asFeature();
        auto ofm = eval->out()->asFeature();

        assert(ifm != nullptr && ofm != nullptr);

        auto app = make_unique<ANNReLU6Appender>();

        app->ifm(ifm);
        app->ofm(ofm);

        return std::move(app);
      }
    }
    else if (auto op = eval->op()->asConcatF())
    {
      auto left_load = op->left()->asLoad();
      auto right_load = op->right()->asLoad();

      if (left_load && right_load && (op->axis() == coco::ConcatF::Axis::Depth))
      {
        // Let's compile the following code fragment:
        //
        //   %ofm = eval(ConcatF(Depth, Load(%left), Load(%right)))
        //
        auto left = left_load->object()->asFeature();
        auto right = right_load->object()->asFeature();
        assert(left != nullptr && right != nullptr);

        auto out = eval->out()->asFeature();
        assert(out != nullptr);

        auto app = make_unique<ANNConcatAppender>();

        app->left(left);
        app->right(right);
        app->out(out);

        return std::move(app);
      }
    }
    else if (auto op = eval->op()->asSub())
    {
      auto left_load = op->left()->asLoad();
      auto right_load = op->right()->asLoad();

      if (left_load && right_load)
      {
        // Let's compile the following code fragment:
        //
        //   %out = eval(Sub(Load(%left), Load(%right)))
        //
        auto left = left_load->object()->asFeature();
        auto right = right_load->object()->asFeature();
        assert(left != nullptr && right != nullptr);

        auto out = eval->out()->asFeature();
        assert(out != nullptr);

        auto app = make_unique<ANNSubAppender>();

        app->left(left);
        app->right(right);
        app->out(out);

        return std::move(app);
      }
    }
    else if (auto op = eval->op()->asDiv())
    {
      auto left_load = op->left()->asLoad();
      auto right_load = op->right()->asLoad();

      if (left_load && right_load)
      {
        // Let's compile the following code fragment:
        //
        //   %out = eval(Div(Load(%left), Load(%right)))
        //
        auto left = left_load->object()->asFeature();
        auto right = right_load->object()->asFeature();
        assert(left != nullptr && right != nullptr);

        auto out = eval->out()->asFeature();
        assert(out != nullptr);

        auto app = make_unique<ANNDivAppender>();

        app->left(left);
        app->right(right);
        app->out(out);

        return std::move(app);
      }
    }

    // Return nullptr if a given Eval instruction is incompatible
    return nullptr;
  }

public:
  std::unique_ptr<ANNOpAppender> visit(const coco::Shuffle *) { return nullptr; }
};

namespace
{

std::unique_ptr<ANNOpAppender> make_appender(coco::Instr *ins)
{
  ANNOpBuilder op_builder;

  if (auto eval = coco::safe_cast<coco::Eval>(ins))
  {
    return eval->accept(op_builder);
  }

  if (auto depth_concat = coco::safe_cast<ANNDepthConcatF>(ins))
  {
    auto app = make_unique<ANNConcatAppender>();

    app->out(depth_concat->out()->asFeature());

    app->left(depth_concat->fst()->asFeature());
    app->right(depth_concat->snd()->asFeature());

    return std::move(app);
  }

  // Build ANN IR from ANNConv2D instruction
  if (auto conv2d = coco::safe_cast<ANNConv2D>(ins))
  {
    auto sess = enco::session(conv2d->module());
    auto app = make_unique<ANNConv2DAppender>();

    app->session(sess);

    app->pad(conv2d->pad());
    app->stride(conv2d->stride());

    app->ofm(conv2d->ofm()->asFeature());
    app->ifm(conv2d->ifm()->asFeature());
    app->ker(conv2d->ker()->asKernel());
    app->bias(coco::safe_cast<coco::FeatureObject>(conv2d->bias()));

    return std::move(app);
  }

  return nullptr;
}

enum Compatibility
{
  COMPATIBLE,
  INCOMPATIBLE
};

class ANNGroupBuilder
{
public:
  ANNGroupBuilder(ANNContext *ctx) : _ctx{ctx}
  {
    // DO NOTHING
  }

public:
  Compatibility kind(const coco::Block *blk) const;
  Compatibility kind(const std::unique_ptr<ANNOpAppender> &appender) const;

public:
  void build(enco::Code *code) const;

private:
  ANNContext *_ctx;
};

Compatibility ANNGroupBuilder::kind(const std::unique_ptr<ANNOpAppender> &app) const
{
  return app ? COMPATIBLE : INCOMPATIBLE;
}

Compatibility ANNGroupBuilder::kind(const coco::Block *blk) const
{
  return (_ctx->find(blk) != nullptr) ? COMPATIBLE : INCOMPATIBLE;
}

void ANNGroupBuilder::build(enco::Code *code) const
{
  auto m = code->module();

  // ANNGroupBuilder will construct a sequence of blocks from the original block sequence, and
  // a destination block (that dst_blk points to) is the tail of the generated sequence.
  coco::Block *dst_blk = nullptr;

  auto append = [&](const Compatibility &t) {
    auto blk = m->entity()->block()->create();

    if (dst_blk == nullptr)
    {
      m->block()->prepend(blk);
    }
    else
    {
      blk->insertAfter(dst_blk);
    }

    dst_blk = blk;

    if (COMPATIBLE == t)
    {
      _ctx->create(blk);
    }
  };

  for (auto blk = m->block()->head(); blk;)
  {
    // Let's move instructions from a block of interest (referred to as source block) into
    // a destination block
    auto src_blk = blk;
    blk = src_blk->next();
    src_blk->detach();

    for (auto ins = src_blk->instr()->head(); ins;)
    {
      auto cur_ins = ins;
      ins = cur_ins->next();
      cur_ins->detach();

      auto cur_append = make_appender(cur_ins);

      // Create a new compatible block and use it as a destination block if the current
      // destination block is absent or incompatible with the instruction of intereset.
      if ((dst_blk == nullptr) || (kind(cur_append) != kind(dst_blk)))
      {
        append(kind(cur_append));
      }

      assert(dst_blk != nullptr);
      assert(kind(cur_append) == kind(dst_blk));

      // Append ins to the dst_blk block
      dst_blk->instr()->append(cur_ins);

      if (cur_append)
      {
        // Update Android NN IR if the current instruction is compatible
        auto binder = _ctx->find(dst_blk);
        assert(binder != nullptr);
        cur_append->append(binder);
      }
    }

    // Destroy the source block
    assert(src_blk->instr()->empty());
    m->entity()->block()->destroy(src_blk);
  }
}

} // namespace

class ANNModuleBuilder
{
private:
  std::set<coco::Bag *> inputs(ANNBinder *binder) const;
  std::set<coco::Bag *> outputs(ANNBinder *binder) const;

public:
  void build(ANNContext *ann_ctx) const;
};

std::set<coco::Bag *> ANNModuleBuilder::inputs(ANNBinder *binder) const
{
  std::set<coco::Bag *> res;

  for (auto bag : binder->bags())
  {
    auto u = enco::updaters(bag);
    u.erase(binder->block());

    /**
     * A bag is the input of this block if
     *  1. it is an input of the whole network, or
     *  2. it is updated by preceding blocks during execution
     */
    if (bag->isInput() || (u.size() > 0))
    {
      res.insert(bag);
    }
  }

  return res;
}

std::set<coco::Bag *> ANNModuleBuilder::outputs(ANNBinder *binder) const
{
  std::set<coco::Bag *> res;

  for (auto bag : binder->bags())
  {
    auto u = enco::updaters(bag);
    auto r = enco::readers(bag);
    r.erase(binder->block());

    /**
     * Only a bag that this block updates can be the output of this block
     */
    if (u.find(binder->block()) == u.end())
    {
      continue;
    }

    /**
     * A bag is the output of this block if
     *  1. it is an output of the whole network, or
     *  2. it is read by following blocks during execution
     */
    if (bag->isOutput() || (r.size() > 0))
    {
      res.insert(bag);
    }
  }

  return res;
}

void ANNModuleBuilder::build(ANNContext *ann_ctx) const
{
  for (uint32_t n = 0; n < ann_ctx->count(); ++n)
  {
    auto binder = ann_ctx->nth(n);

    // NOTE binder->module() returns an ANN IR module (not coco IR module)
    auto m = binder->block()->module();
    auto d = enco::data(m);

    // Let's identify operands with initial values
    for (auto bag : binder->bags())
    {
      if (binder->associated(bag) && d->allocated(bag))
      {
        // TODO Support other datatype
        auto span = d->f32()->weight(bag);
        assert(span.data() != nullptr);

        binder->setOperand(binder->operand(bag), span.data(), span.data() + span.size());
      }
    }

    // Let's identify input/output bags
    binder->identifyInputs(inputs(binder));
    binder->identifyOutputs(outputs(binder));
  }
}

} // namespace

namespace
{

class SplitPass
{
public:
  void runOnCode(enco::Code *code) const;
};

void SplitPass::runOnCode(enco::Code *code) const
{
  auto ann_ctx = make_unique<ANNContext>();

  ANNGroupBuilder group_builder{ann_ctx.get()};
  group_builder.build(code);

  ANNModuleBuilder module_builder;
  module_builder.build(ann_ctx.get());

  _subnet_contexts[code->module()] = std::move(ann_ctx);
}

} // namespace

namespace enco
{

void split_into_phases(enco::Code *code)
{
  SplitPass split;
  split.runOnCode(code);
}

} // namespace enco
