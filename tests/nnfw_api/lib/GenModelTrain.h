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

#ifndef __NNFW_API_TEST_GEN_MODEL_TRAIN_H__
#define __NNFW_API_TEST_GEN_MODEL_TRAIN_H__

#include "GenModelTest.h"
#include "CirclePlusGen.h"

#include <set>

struct SessionObjectTraining : public SessionObjectGeneric
{
  std::vector<std::vector<uint8_t>> losses;
};

struct TrainCaseData : public TestCaseData
{
  /**
   * @brief A vector of losses buffers
   */
  std::vector<std::vector<uint8_t>> losses;

  /**
   * @brief Append vector data to losses
   *
   * @tparam T Data type
   * @param data vector data array
   */
  template <typename T> TrainCaseData &addLosses(const std::vector<T> &data)
  {
    addData(losses, data);
    return *this;
  }
};

/**
 * @brief Create a TrainCaseData with a uniform type
 *
 * A helper function for generating train cases that has the same data type for model
 * inputs/expects/losses.
 *
 * @tparam T Uniform tensor type
 * @param inputs Inputs tensor buffers
 * @param expects Outputs tensor buffers
 * @param losses Losses tensor buffers
 * @return TrainCaseData Generated train case data
 */
template <typename T>
static TrainCaseData uniformTCD(const std::vector<std::vector<T>> &inputs,
                                const std::vector<std::vector<T>> &expects,
                                const std::vector<std::vector<float>> &losses)
{
  TrainCaseData ret;
  for (const auto &data : inputs)
    ret.addInput(data);
  for (const auto &data : expects)
    ret.addOutput(data);
  for (const auto &data : losses)
    ret.addLosses(data);
  return ret;
}

/**
 * @brief A train configuration class
 */
class GenModelTrainContext : public GenModelTestContext
{
public:
  GenModelTrainContext(CircleBuffers &&cbufs)
    : GenModelTestContext(std::move(cbufs.circle)), _cpbuf{std::move(cbufs.circle_plus)}, _epoch(0)
  {
    // DO NOTHING
  }

  /**
   * @brief  Return circle plus buffer
   *
   * @return CircleBuffer& the circle plus buffer
   */
  const CircleBuffer &cpbuf() const { return _cpbuf; }

  /**
   * @brief Return train cases
   *
   * @return std::vector<TrainCaseData>& the train cases
   */
  const std::vector<TrainCaseData> &train_cases() const { return _train_cases; }

  /**
   * @brief Add a train case
   *
   * @param tc the train case to be added
   */
  void addTrainCase(const TrainCaseData &tc) { _train_cases.emplace_back(tc); }

  int epoch() { return _epoch; }

  void setEpoch(int32_t epoch)
  {
    if (epoch < 2)
    {
      throw std::runtime_error{"epoch should be equal or greater than 2"};
    }
    _epoch = epoch;
  }

  void enableTrainNodeUpdate(const int op_idx) { _trainable_ops_idx.emplace(op_idx); }

  void disableTrainNodeUpdate(const int op_idx)
  {
    if (_trainable_ops_idx.find(op_idx) != std::end(_trainable_ops_idx))
    {
      _trainable_ops_idx.erase(op_idx);
    }
  }

  const std::set<int> &getTrainableOpsIdx() const { return _trainable_ops_idx; }

private:
  CircleBuffer _cpbuf;
  std::vector<TrainCaseData> _train_cases;
  int32_t _epoch;
  std::set<int> _trainable_ops_idx;
};

/**
 * @brief Generated Model test fixture for a one time training
 *
 * This fixture is for one-time training test with variety of generated models.
 * It is the test maker's responsiblity to create @c _context which contains
 * test body, which are generated circle buffer, model input data and output data and
 * backend list to be tested.
 * The rest(calling API functions for execution) is done by @c SetUp and @c TearDown .
 *
 */
class GenModelTrain : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // DO NOTHING
  }

  void TearDown() override
  {
    // NOTE: Assume that the only backend is `train` backend
    ASSERT_EQ(_context->backends().size(), 1);
    ASSERT_STREQ(_context->backends()[0].c_str(), "train");

    for (std::string backend : _context->backends())
    {
      // NOTE If we can prepare many times for one model loading on same session,
      //      we can move nnfw_create_session to SetUp and
      //      nnfw_load_circle_from_buffer to outside forloop
      NNFW_ENSURE_SUCCESS(nnfw_create_session(&_so.session));
      auto &cbuf = _context->cbuf();
      auto model_load_result =
        nnfw_load_circle_from_buffer(_so.session, cbuf.buffer(), cbuf.size());
      if (_context->expected_fail_model_load())
      {
        ASSERT_NE(model_load_result, NNFW_STATUS_NO_ERROR);
        std::cerr << "Failed model loading as expected." << std::endl;
        NNFW_ENSURE_SUCCESS(nnfw_close_session(_so.session));
        continue;
      }
      NNFW_ENSURE_SUCCESS(model_load_result);
      NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(_so.session, backend.data()));

      for (const auto &frozen_op_idx : _context->getTrainableOpsIdx())
      {
        nnfw_train_enable_node_update(_so.session, frozen_op_idx);
      }

      if (_context->expected_fail_compile())
      {
        ASSERT_NE(nnfw_train_prepare(_so.session), NNFW_STATUS_NO_ERROR);

        NNFW_ENSURE_SUCCESS(nnfw_close_session(_so.session));
        continue;
      }

      // the nnfw_input_size() and nnfw_output_size() should be called before nnfw_train_prepare
      uint32_t num_inputs;
      NNFW_ENSURE_SUCCESS(nnfw_input_size(_so.session, &num_inputs));

      uint32_t num_expecteds;
      NNFW_ENSURE_SUCCESS(nnfw_output_size(_so.session, &num_expecteds));

      // training information
      nnfw_train_info tri;

      // NOTE: This can be removed when circle schema and circle+ schema are merged and
      // then `nnfw_load_circle_from_buffer` handles traininfo metadata of circle model
      {
        // code is copied from runtime/onert/core/src/loader/traininfo_loader.cc
        auto &cpbuf = _context->cpbuf();
        const uint8_t *buffer = cpbuf.buffer();
        const size_t size = cpbuf.size();

        assert(buffer != nullptr);
        flatbuffers::Verifier v(buffer, size);
        bool verified = circle::VerifyModelTrainingBuffer(v);
        if (not verified)
          throw std::runtime_error{"TrainingInfo buffer is not accessible"};

        const circle::ModelTraining *circle_plus =
          circle::GetModelTraining(static_cast<const void *>(buffer));

        assert(circle_plus != nullptr);

        tri = LoadTrainInfo(circle_plus);
      }
      NNFW_ENSURE_SUCCESS(nnfw_train_set_traininfo(_so.session, &tri));

      // prepare for training
      NNFW_ENSURE_SUCCESS(nnfw_train_prepare(_so.session));

      // Prepare input
      _so.inputs.resize(num_inputs);
      std::vector<nnfw_tensorinfo> input_infos(num_inputs);
      for (uint32_t ind = 0; ind < num_inputs; ind++)
      {
        nnfw_tensorinfo ti;
        NNFW_ENSURE_SUCCESS(nnfw_input_tensorinfo(_so.session, ind, &ti));
        uint64_t input_elements = num_elems(&ti);
        _so.inputs[ind].resize(input_elements * sizeOfNnfwType(ti.dtype));

        // Optional inputs are not supported yet
        ASSERT_NE(_so.inputs[ind].size(), 0);

        NNFW_ENSURE_SUCCESS(nnfw_train_set_input(_so.session, ind, _so.inputs[ind].data(), &ti));

        input_infos.emplace_back(std::move(ti));
      }

      // Prepare expected output
      _so.outputs.resize(num_expecteds);
      std::vector<nnfw_tensorinfo> expected_infos(num_expecteds);
      for (uint32_t ind = 0; ind < num_expecteds; ind++)
      {
        nnfw_tensorinfo ti;
        NNFW_ENSURE_SUCCESS(nnfw_output_tensorinfo(_so.session, ind, &ti));
        uint64_t output_elements = num_elems(&ti);
        _so.outputs[ind].resize(output_elements * sizeOfNnfwType(ti.dtype));

        // Setting the output buffer size of specified output tensor is not supported yet
        ASSERT_EQ(_context->hasOutputSizes(ind), false);

        NNFW_ENSURE_SUCCESS(
          nnfw_train_set_expected(_so.session, ind, _so.outputs[ind].data(), &ti));

        expected_infos.emplace_back(std::move(ti));
      }

      // Prepare expected loss
      _so.losses.resize(num_expecteds);

      const int num_epoch = _context->epoch();
      ASSERT_GE(num_epoch, 2);
      const int num_step = num_expecteds / tri.batch_size;
      ASSERT_GE(num_step, 1);

      // Set input values & expected output values, train, and check loss
      for (const auto &train_case : _context->train_cases())
      {
        // Inputs
        const auto &ref_inputs = train_case.inputs;
        ASSERT_EQ(_so.inputs.size(), ref_inputs.size());
        for (uint32_t i = 0; i < _so.inputs.size(); i++)
        {
          // Fill the values
          ASSERT_EQ(_so.inputs[i].size(), ref_inputs[i].size());
          memcpy(_so.inputs[i].data(), ref_inputs[i].data(), ref_inputs[i].size());
        }

        // Expected outputs
        const auto &ref_outputs = train_case.outputs;
        ASSERT_EQ(_so.outputs.size(), ref_outputs.size());
        for (uint32_t i = 0; i < _so.outputs.size(); i++)
        {
          // Fill the values
          ASSERT_EQ(_so.outputs[i].size(), ref_outputs[i].size());
          memcpy(_so.outputs[i].data(), ref_outputs[i].data(), ref_outputs[i].size());
        }

        // Expected losses
        const auto &ref_losses = train_case.losses;
        ASSERT_EQ(_so.losses.size(), ref_losses.size());
        for (uint32_t i = 0; i < _so.losses.size(); i++)
        {
          // Fill the values
          _so.losses[i] = ref_losses[i];
        }

        if (train_case.expected_fail_run())
        {
          ASSERT_NE(nnfw_train(_so.session, true), NNFW_STATUS_NO_ERROR);
          continue;
        }

        // Train
        std::vector<float> actual_losses(num_expecteds, 0.f);
        for (uint32_t epoch = 0; epoch < num_epoch; ++epoch)
        {
          NNFW_ENSURE_SUCCESS(nnfw_train(_so.session, true));

          // Store loss
          for (int32_t i = 0; i < num_expecteds; ++i)
          {
            float temp = 0.f;
            NNFW_ENSURE_SUCCESS(nnfw_train_get_loss(_so.session, i, &temp));
            actual_losses[i] += temp;
          }
        }

        // Recalculate loss
        for (uint32_t i = 0; i < num_expecteds; ++i)
        {
          actual_losses[i] /= num_step;
        }

        ASSERT_EQ(actual_losses.size(), _so.losses.size());

        // TODO better way for handling FP error?
        for (uint32_t i = 0; i < actual_losses.size(); i++)
        {
          float actual = actual_losses[i];
          const auto &expected_loss = _so.losses[i];
          ASSERT_EQ(expected_loss.size(), sizeof(float));
          float expected = *(reinterpret_cast<const float *>(expected_loss.data()));
          EXPECT_NEAR(expected, actual, 0.001) << "Loss #" << i;
        }
      }

      NNFW_ENSURE_SUCCESS(nnfw_close_session(_so.session));
    }
  }

private:
  nnfw_train_info LoadTrainInfo(const circle::ModelTraining *circle_model)
  {
    nnfw_train_info tri;

    const circle::Optimizer optimizer = circle_model->optimizer();
    switch (optimizer)
    {
      case circle::Optimizer_SGD:
        tri.opt = NNFW_TRAIN_OPTIMIZER_SGD;
        tri.learning_rate = circle_model->optimizer_opt_as_SGDOptions()->learning_rate();
        break;
      case circle::Optimizer_ADAM:
        tri.opt = NNFW_TRAIN_OPTIMIZER_ADAM;
        tri.learning_rate = circle_model->optimizer_opt_as_AdamOptions()->learning_rate();
        break;
      default:
        throw std::runtime_error("unknown optimzer");
    }

    const circle::LossFn loss_type = circle_model->lossfn();
    switch (loss_type)
    {
      case circle::LossFn::LossFn_CATEGORICAL_CROSSENTROPY:
        tri.loss_info.loss = NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY;
        break;
      case circle::LossFn::LossFn_MEAN_SQUARED_ERROR:
        tri.loss_info.loss = NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR;
        break;
      case circle::LossFn::LossFn_SPARSE_CATEGORICAL_CROSSENTROPY:
        // TODO enable this conversion after core support sparse_categorial_crossentropy
        throw std::runtime_error{"'sparse_categorical_crossentropy' is not supported yet"};
      default:
        throw std::runtime_error{"unknown loss function"};
    }

    const circle::LossReductionType loss_reduction_type = circle_model->loss_reduction_type();
    switch (loss_reduction_type)
    {
      case circle::LossReductionType::LossReductionType_SumOverBatchSize:
        tri.loss_info.reduction_type = NNFW_TRAIN_LOSS_REDUCTION_SUM_OVER_BATCH_SIZE;
        break;
      case circle::LossReductionType::LossReductionType_Sum:
        tri.loss_info.reduction_type = NNFW_TRAIN_LOSS_REDUCTION_SUM;
        break;
      default:
        throw std::runtime_error{"unknown loss reduction type"};
    }

    tri.batch_size = circle_model->batch_size();

    return tri;
  }

protected:
  SessionObjectTraining _so;
  std::unique_ptr<GenModelTrainContext> _context;
};

#endif // __NNFW_API_TEST_GEN_MODEL_TRAIN_H__
