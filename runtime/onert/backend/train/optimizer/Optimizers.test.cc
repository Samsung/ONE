/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Optimizers.h"

#include <backend/train/ITrainableTensor.h>

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace
{

using namespace onert;
using namespace onert::backend;
using namespace onert::ir;

class MockUpTensor : public IPortableTensor
{
public:
  MockUpTensor(const Shape &shape, const TypeInfo &type_info)
    : IPortableTensor{ir::OperandInfo{shape, type_info, MemAllocType::STATIC}}, _data{}
  {
  }
  virtual ~MockUpTensor() = default;

  template <typename T> void setData(const std::vector<T> &data)
  {
    _data.resize(data.size() * sizeof(T));
    std::copy(data.begin(), data.end(), _data.begin());
  }

  size_t total_size() const override
  {
    size_t total_size = _info.shape().num_elements();
    total_size *= sizeOfDataType(data_type());
    return total_size;
  }

  size_t calcOffset(const ir::Coordinates &coords) const override
  {
    const auto &shape = _info.shape();
    std::vector<size_t> strides(shape.rank());

    size_t stride = 1;
    for (int32_t i = shape.rank() - 1; i >= 0; --i)
    {
      strides.at(i) = stride;
      stride = stride * shape.dim(i);
    }

    size_t offset = 0;
    for (int32_t i = 0; i < shape.rank(); ++i)
    {
      offset += (strides[i] * coords[i]);
    }
    offset *= sizeOfDataType(data_type());
    return offset;
  }

  uint8_t *buffer() const override
  {
    if (_data.size() != total_size())
      return nullptr;

    return const_cast<uint8_t *>(_data.data());
  }

  template <typename T> const std::vector<T> &data() const { return _data; }

  ir::Layout layout() const override { return ir::Layout::NHWC; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }

  bool is_dynamic() const override { return false; }
  Shape getShape() const override { return _info.shape(); }

private:
  using ITensor::setShape;
  using ITensor::set_dynamic;
  using ITensor::applyShape;

private:
  std::vector<uint8_t> _data;
};

class MockUpTrainableTensor : public backend::train::ITrainableTensor
{
public:
  MockUpTrainableTensor(const Shape &shape, const TypeInfo &type_info)
    : ITrainableTensor{ir::OperandInfo{shape, type_info, MemAllocType::STATIC}}, _data{},
      _opt_vars{}
  {
  }
  virtual ~MockUpTrainableTensor() {}

  template <typename T> void setData(const std::vector<T> &data)
  {
    _data.resize(data.size() * sizeof(T));
    std::copy(data.begin(), data.end(), _data.begin());
  }

  size_t total_size() const override
  {
    size_t total_size = _info.shape().num_elements();
    total_size *= sizeOfDataType(data_type());
    return total_size;
  }

  size_t calcOffset(const ir::Coordinates &coords) const override
  {
    const auto &shape = _info.shape();
    std::vector<size_t> strides(shape.rank());

    size_t stride = 1;
    for (int32_t i = shape.rank() - 1; i >= 0; --i)
    {
      strides.at(i) = stride;
      stride = stride * shape.dim(i);
    }

    size_t offset = 0;
    for (int32_t i = 0; i < shape.rank(); ++i)
    {
      offset += (strides[i] * coords[i]);
    }
    offset *= sizeOfDataType(data_type());
    return offset;
  }

  uint8_t *buffer() const override
  {
    if (_data.size() != total_size())
      return nullptr;

    return const_cast<uint8_t *>(_data.data());
  }

  ir::Layout layout() const override { return ir::Layout::NHWC; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }

  bool is_dynamic() const override { return false; }
  Shape getShape() const override { return _info.shape(); }

public:
  std::vector<ITensor *> optVars() override
  {
    std::vector<ITensor *> ret;
    for (auto &&e : _opt_vars)
    {
      ret.emplace_back(e);
    }
    return ret;
  }
  void appendOptVar(MockUpTensor *opt_var) { _opt_vars.emplace_back(opt_var); }

private:
  using ITensor::setShape;
  using ITensor::set_dynamic;
  using ITensor::applyShape;

private:
  std::vector<uint8_t> _data;
  std::vector<MockUpTensor *> _opt_vars; //< Optimizer variables
};

template <typename T> class SGDOptimizerVerifier
{
public:
  SGDOptimizerVerifier(MockUpTrainableTensor &trainable, const MockUpTensor &gradient,
                       float learning_rate, uint32_t nums_step)
    : _sgd{backend::train::optimizer::SGD::Property{}, learning_rate}, _trainable{trainable},
      _gradient{gradient}, _learning_rate{learning_rate}, _nums_step{nums_step}
  {
    EXPECT_TRUE(trainable.total_size() == gradient.total_size());

    _expected_trainable.resize(trainable.getShape().num_elements());
    memcpy(_expected_trainable.data(), trainable.buffer(), trainable.total_size());

    _expected_gradient.resize(gradient.getShape().num_elements());
    memcpy(_expected_gradient.data(), gradient.buffer(), gradient.total_size());
  }

  SGDOptimizerVerifier(const SGDOptimizerVerifier &) = delete;
  SGDOptimizerVerifier &operator=(const SGDOptimizerVerifier &) = delete;
  SGDOptimizerVerifier(SGDOptimizerVerifier &&) = delete;
  SGDOptimizerVerifier &operator=(SGDOptimizerVerifier &&) = delete;

public:
  void verify()
  {
    for (uint32_t step = 0; step < _nums_step; ++step)
    {
      auto actual_lr = _sgd.getLearningRate(step);
      EXPECT_EQ(actual_lr, _learning_rate);

      calculateExpected();

      backend::train::optimizer::Adam::UpdateFactors factors{_gradient, _trainable, step};
      _sgd.applyGradient(factors);

      for (size_t i = 0; i < _expected_trainable.size(); ++i)
      {
        EXPECT_NEAR(reinterpret_cast<T *>(_gradient.buffer())[i], _expected_gradient[i], 1e-07f);
        EXPECT_NEAR(reinterpret_cast<T *>(_trainable.buffer())[i], _expected_trainable[i], 1e-07f);
      }
    }
  }

private:
  void calculateExpected()
  {
    EXPECT_TRUE(_expected_trainable.size() == _expected_gradient.size());

    for (size_t i = 0; i < _expected_trainable.size(); ++i)
    {
      T &g = _expected_gradient[i];
      T &t = _expected_trainable[i];

      t -= g * _learning_rate;
    }
  }

private:
  backend::train::optimizer::SGD _sgd;
  MockUpTrainableTensor &_trainable;
  const MockUpTensor &_gradient;
  float _learning_rate;
  uint32_t _nums_step;

  std::vector<T> _expected_trainable;
  std::vector<T> _expected_gradient;
};

template <typename T> class AdamOptimizerVerifier
{
public:
  AdamOptimizerVerifier(MockUpTrainableTensor &trainable, const MockUpTensor &gradient,
                        float learning_rate, float beta1, float beta2, float epsilon,
                        bool use_nesterov, uint32_t nums_step)
    : _adam{backend::train::optimizer::Adam::Property{beta1, beta2, epsilon}, learning_rate},
      _trainable{trainable}, _gradient{gradient}, _learning_rate{learning_rate}, _beta1{beta1},
      _beta2{beta2}, _epsilon{epsilon}, _use_nesterov{use_nesterov}, _nums_step{nums_step}
  {
    auto vars = trainable.optVars();
    EXPECT_TRUE(vars.size() == 2);
    const auto &m = *vars[0];
    const auto &v = *vars[1];
    EXPECT_TRUE(trainable.total_size() == gradient.total_size());
    EXPECT_TRUE(trainable.total_size() == m.total_size());
    EXPECT_TRUE(trainable.total_size() == v.total_size());

    _expected_trainable.resize(trainable.getShape().num_elements());
    memcpy(_expected_trainable.data(), trainable.buffer(), trainable.total_size());

    _expected_gradient.resize(gradient.getShape().num_elements());
    memcpy(_expected_gradient.data(), gradient.buffer(), gradient.total_size());

    _expected_m.resize(m.getShape().num_elements());
    memcpy(_expected_m.data(), m.buffer(), m.total_size());

    _expected_v.resize(v.getShape().num_elements());
    memcpy(_expected_v.data(), v.buffer(), v.total_size());
  }

  AdamOptimizerVerifier(const AdamOptimizerVerifier &) = delete;
  AdamOptimizerVerifier &operator=(const AdamOptimizerVerifier &) = delete;

public:
  void verify()
  {
    for (uint32_t step = 0; step < _nums_step; ++step)
    {
      const T beta1_power = std::pow(_beta1, step + 1);
      const T beta2_power = std::pow(_beta2, step + 1);

      calculateExpected(beta1_power, beta2_power);

      backend::train::optimizer::Adam::UpdateFactors factors{_gradient, _trainable, step};
      _adam.applyGradient(factors);

      const auto vars = _trainable.optVars();
      const auto &m = *vars[0];
      const auto &v = *vars[1];

      auto actual_lr = _adam.getLearningRate(step);
      auto expected_lr = [&]() {
        auto biasCorrection = [&](double f) { return 1.0f - std::pow(f, step + 1); };
        return _learning_rate * (std::sqrt(biasCorrection(_beta2)) / biasCorrection(_beta1));
      }();
      EXPECT_EQ(actual_lr, expected_lr);

      for (size_t i = 0; i < _expected_trainable.size(); ++i)
      {
        EXPECT_NEAR(reinterpret_cast<T *>(m.buffer())[i], _expected_m[i], 1e-07f);
        EXPECT_NEAR(reinterpret_cast<T *>(v.buffer())[i], _expected_v[i], 1e-07f);
        EXPECT_NEAR(reinterpret_cast<T *>(_gradient.buffer())[i], _expected_gradient[i], 1e-07f);
        EXPECT_NEAR(reinterpret_cast<T *>(_trainable.buffer())[i], _expected_trainable[i], 1e-07f);
      }
    }
  }

private:
  void calculateExpected(const float beta1_power, const float beta2_power)
  {
    EXPECT_TRUE(_expected_trainable.size() == _expected_gradient.size());
    EXPECT_TRUE(_expected_trainable.size() == _expected_m.size());
    EXPECT_TRUE(_expected_trainable.size() == _expected_v.size());

    const T alpha = _learning_rate * std::sqrt(static_cast<T>(1) - beta2_power) /
                    (static_cast<T>(1) - beta1_power);

    for (size_t i = 0; i < _expected_trainable.size(); ++i)
    {
      T &m = _expected_m[i];
      T &v = _expected_v[i];
      T &g = _expected_gradient[i];
      T &t = _expected_trainable[i];

      if (_use_nesterov)
      {
        m += (g - m) * (static_cast<T>(1) - _beta1);
        v += (std::pow(g, 2) - v) * (static_cast<T>(1) - _beta2);
        t -= ((g * (static_cast<T>(1) - _beta1) + _beta1 * m) * alpha) / (std::sqrt(v) + _epsilon);
      }
      else
      {
        m += (g - m) * (static_cast<T>(1) - _beta1);
        v += (std::pow(g, 2) - v) * (static_cast<T>(1) - _beta2);
        t -= (m * alpha) / (std::sqrt(v) + _epsilon);
      }
    }
  }

private:
  backend::train::optimizer::Adam _adam;
  MockUpTrainableTensor &_trainable;
  const MockUpTensor &_gradient;
  float _learning_rate;
  float _beta1;
  float _beta2;
  float _epsilon;
  bool _use_nesterov;
  uint32_t _nums_step;

  std::vector<T> _expected_trainable;
  std::vector<T> _expected_gradient;
  std::vector<T> _expected_m;
  std::vector<T> _expected_v;
};

} // namespace

TEST(Optimizer, SGDValueValidation)
{
  // 10 steps
  {
    const auto shape = ir::Shape{1, 3, 3};
    const auto type_info = ir::TypeInfo{ir::DataType::FLOAT32};
    MockUpTrainableTensor trainable{shape, type_info};
    MockUpTensor gradient{shape, type_info};

    std::vector<float> trainable_data = {-1, 2, -3, -4, 5, -6, 7, -8, 9};
    std::vector<float> gradient_data = {-1, -2, -3, 4, 5, -6, 7, 8, -9};

    trainable.setData(trainable_data);
    gradient.setData(gradient_data);

    float lr = 0.001;
    uint32_t training_steps = 10;

    SGDOptimizerVerifier<float>{trainable, gradient, lr, training_steps}.verify();
  }

  // 10000 steps
  {
    const auto shape = ir::Shape{1, 3, 3};
    const auto type_info = ir::TypeInfo{ir::DataType::FLOAT32};
    MockUpTrainableTensor trainable{shape, type_info};
    MockUpTensor gradient{shape, type_info};

    std::vector<float> trainable_data = {-1, 2, -3, -4, 5, -6, 7, -8, 9};
    std::vector<float> gradient_data = {-1, -2, -3, 4, 5, -6, 7, 8, -9};

    trainable.setData(trainable_data);
    gradient.setData(gradient_data);

    float lr = 0.001;
    uint32_t training_steps = 10000;

    SGDOptimizerVerifier<float>{trainable, gradient, lr, training_steps}.verify();
  }
}

TEST(Optimizer, SGDVarCount)
{
  backend::train::optimizer::SGD sgd{};

  EXPECT_EQ(sgd.getVarCount(), 0);
}

TEST(Optimizer, neg_SGDUnmatchedGradientShape)
{
  // Unmatched shape
  {
    const auto type_info = ir::TypeInfo{ir::DataType::FLOAT32};
    MockUpTrainableTensor trainable{{1, 3, 3}, type_info};
    MockUpTensor gradient{ir::Shape{2, 2, 2}, type_info};

    std::vector<float> trainable_data = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient_data = {-1, 2, -3, 4, 5, -6, 7, 8};

    trainable.setData(trainable_data);
    gradient.setData(gradient_data);

    float lr = 0.001;

    backend::train::optimizer::SGD sgd{lr};
    backend::train::optimizer::SGD::UpdateFactors factors{gradient, trainable, 0};

    EXPECT_ANY_THROW(sgd.applyGradient(factors));
  }
}

TEST(Optimizer, neg_SGDUnsupportedType)
{
  // Unsupported type
  {
    const auto shape = ir::Shape{1, 3, 3};
    const auto type_info = ir::TypeInfo{ir::DataType::INT32};
    MockUpTrainableTensor trainable{shape, type_info};
    MockUpTensor gradient{shape, type_info};

    std::vector<float> trainable_data = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient_data = {-1, 2, -3, 4, 5, -6, 7, 8, -9};

    trainable.setData(trainable_data);
    gradient.setData(gradient_data);

    float lr = 0.001;

    backend::train::optimizer::SGD sgd{lr};
    backend::train::optimizer::SGD::UpdateFactors factors{gradient, trainable, 0};

    EXPECT_ANY_THROW(sgd.applyGradient(factors));
  }
}

TEST(Optimizer, AdamValueValidation)
{
  // 10 steps
  {
    const auto shape = ir::Shape{1, 3, 3};
    const auto type_info = ir::TypeInfo{ir::DataType::FLOAT32};
    MockUpTrainableTensor trainable{shape, type_info};
    MockUpTensor gradient{shape, type_info};
    MockUpTensor m{shape, type_info};
    MockUpTensor v{shape, type_info};

    std::vector<float> trainable_data = {-1, 2, -3, -4, 5, -6, 7, -8, 9};
    std::vector<float> gradient_data = {-1, -2, -3, 4, 5, -6, 7, 8, -9};
    std::vector<float> m_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> v_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    trainable.setData(trainable_data);
    gradient.setData(gradient_data);
    m.setData(m_data);
    v.setData(v_data);

    trainable.appendOptVar(&m);
    trainable.appendOptVar(&v);

    float lr = 0.1;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-07;
    bool use_nesterov = false;
    uint32_t training_steps = 10;

    AdamOptimizerVerifier<float>{trainable, gradient, lr,           beta1,
                                 beta2,     epsilon,  use_nesterov, training_steps}
      .verify();
  }

  // 10000 steps
  {
    const auto shape = ir::Shape{1, 3, 3};
    const auto type_info = ir::TypeInfo{ir::DataType::FLOAT32};
    MockUpTrainableTensor trainable{shape, type_info};
    MockUpTensor gradient{shape, type_info};
    MockUpTensor m{shape, type_info};
    MockUpTensor v{shape, type_info};

    std::vector<float> trainable_data = {-1, 2, -3, -4, 5, -6, 7, -8, 9};
    std::vector<float> gradient_data = {-1, -2, -3, 4, 5, -6, 7, 8, -9};
    std::vector<float> m_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> v_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    trainable.setData(trainable_data);
    gradient.setData(gradient_data);
    m.setData(m_data);
    v.setData(v_data);

    trainable.appendOptVar(&m);
    trainable.appendOptVar(&v);

    float lr = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-07;
    bool use_nesterov = false;
    uint32_t training_steps = 10000;

    AdamOptimizerVerifier<float>{trainable, gradient, lr,           beta1,
                                 beta2,     epsilon,  use_nesterov, training_steps}
      .verify();
  }

  // TODO Add tests with use_nesterov = true
}

TEST(Optimizer, AdamVarCount)
{
  float lr = 0.001;
  backend::train::optimizer::Adam adam(lr);

  EXPECT_EQ(adam.getVarCount(), 2);
}

TEST(Optimizer, neg_AdamUnmatchedGradientShape)
{
  // Unmatched shape
  {
    const auto shape = ir::Shape{1, 3, 3};
    const auto type_info = ir::TypeInfo{ir::DataType::FLOAT32};
    MockUpTrainableTensor trainable{shape, type_info};
    MockUpTensor gradient{ir::Shape{2, 2, 2}, type_info};
    MockUpTensor m{shape, type_info};
    MockUpTensor v{shape, type_info};

    std::vector<float> trainable_data = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient_data = {-1, 2, -3, 4, 5, -6, 7, 8};
    std::vector<float> m_data = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v_data = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    trainable.setData(trainable_data);
    gradient.setData(gradient_data);
    m.setData(m_data);
    v.setData(v_data);

    trainable.appendOptVar(&m);
    trainable.appendOptVar(&v);

    backend::train::optimizer::Adam adam{};
    backend::train::optimizer::Adam::UpdateFactors factors{gradient, trainable, 0};

    EXPECT_ANY_THROW(adam.applyGradient(factors));
  }
}

TEST(Optimizer, neg_AdamUnmatchedEMAShape)
{
  // Unmatched shape
  {
    const auto shape = ir::Shape{1, 3, 3};
    const auto type_info = ir::TypeInfo{ir::DataType::FLOAT32};
    MockUpTrainableTensor trainable{shape, type_info};
    MockUpTensor gradient{shape, type_info};
    MockUpTensor m{ir::Shape{2, 2, 2}, type_info};
    MockUpTensor v{ir::Shape{2, 2, 2}, type_info};

    std::vector<float> trainable_data = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient_data = {-1, 2, -3, 4, 5, -6, 7, 8};
    std::vector<float> m_data = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v_data = {0, 0, 0, 0, 0, 0, 0, 0};

    trainable.setData(trainable_data);
    gradient.setData(gradient_data);
    m.setData(m_data);
    v.setData(v_data);

    trainable.appendOptVar(&m);
    trainable.appendOptVar(&v);

    backend::train::optimizer::Adam adam{};
    backend::train::optimizer::Adam::UpdateFactors factors{gradient, trainable, 0};

    EXPECT_ANY_THROW(adam.applyGradient(factors));
  }
}

TEST(Optimizer, neg_AdamUnsupportedType)
{
  // Unsupported type
  {
    const auto shape = ir::Shape{1, 3, 3};
    const auto type_info = ir::TypeInfo{ir::DataType::INT32};
    MockUpTrainableTensor trainable{shape, type_info};
    MockUpTensor gradient{shape, type_info};
    MockUpTensor m{shape, type_info};
    MockUpTensor v{shape, type_info};

    std::vector<float> trainable_data = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient_data = {-1, 2, -3, 4, 5, -6, 7, 8, -9};
    std::vector<float> m_data = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v_data = {0, 0, 0, 0, 0, 0, 0, 0};

    trainable.setData(trainable_data);
    gradient.setData(gradient_data);
    m.setData(m_data);
    v.setData(v_data);

    trainable.appendOptVar(&m);
    trainable.appendOptVar(&v);

    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-07;

    backend::train::optimizer::Adam adam{
      backend::train::optimizer::Adam::Property{beta1, beta2, epsilon}};
    backend::train::optimizer::Adam::UpdateFactors factors{gradient, trainable, 0};

    EXPECT_ANY_THROW(adam.applyGradient(factors));
  }
}

TEST(Optimizer, CreateOptimizer)
{
  // SGD
  {
    ir::train::OptimizerInfo optim_info;
    optim_info.optim_code = ir::train::OptimizerCode::SGD;
    optim_info.learning_rate = 0.001f;
    auto sgd = backend::train::createOptimizer(optim_info);
    EXPECT_EQ(sgd->getVarCount(), 0);
    EXPECT_EQ(sgd->name(), std::string{"SGD"});
  }

  // Adam
  {
    ir::train::OptimizerInfo optim_info;
    optim_info.optim_code = ir::train::OptimizerCode::Adam;
    optim_info.learning_rate = 0.001f;
    auto adam = backend::train::createOptimizer(optim_info);
    EXPECT_EQ(adam->getVarCount(), 2);
    EXPECT_EQ(adam->name(), std::string{"Adam"});
  }
}

TEST(Optimizer, neg_UndefinedOptimizerCode)
{
  // Undefined optimizer code
  {
    ir::train::OptimizerInfo optim_info;
    optim_info.optim_code = ir::train::OptimizerCode::Undefined;
    EXPECT_ANY_THROW(backend::train::createOptimizer(optim_info));
  }
}
