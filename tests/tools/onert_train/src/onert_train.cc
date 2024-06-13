/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "allocation.h"
#include "args.h"
#include "benchmark.h"
#include "measure.h"
#include "nnfw.h"
#include "nnfw_util.h"
#include "nnfw_internal.h"
#include "nnfw_experimental.h"
#include "randomgen.h"
#include "rawformatter.h"
#include "dataloader.h"
#include "rawdataloader.h"
#include "metrics.h"

#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <libgen.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

static const char *default_backend_cand = "train";

int main(const int argc, char **argv)
{
  using namespace onert_train;

  try
  {
    Args args(argc, argv);
    if (args.printVersion())
    {
      uint32_t version;
      NNPR_ENSURE_STATUS(nnfw_query_info_u32(NULL, NNFW_INFO_ID_VERSION, &version));
      std::cout << "onert_train (nnfw runtime: v" << (version >> 24) << "."
                << ((version & 0x0000FF00) >> 8) << "." << (version & 0xFF) << ")" << std::endl;
      exit(0);
    }

    // TODO Apply verbose level to phases
    const int verbose = args.getVerboseLevel();

    // prepare measure tool
    Measure measure(args.getMemoryPoll());

    nnfw_session *session = nullptr;
    NNPR_ENSURE_STATUS(nnfw_create_session(&session));

    // ModelLoad
    measure.run(PhaseType::MODEL_LOAD, [&]() {
      if (args.useSingleModel())
        NNPR_ENSURE_STATUS(
          nnfw_load_model_from_modelfile(session, args.getModelFilename().c_str()));
      else
        NNPR_ENSURE_STATUS(nnfw_load_model_from_file(session, args.getPackageFilename().c_str()));
    });

    // Set training backend
    NNPR_ENSURE_STATUS(nnfw_set_available_backends(session, default_backend_cand));

    uint32_t num_inputs;
    NNPR_ENSURE_STATUS(nnfw_input_size(session, &num_inputs));

    uint32_t num_expecteds;
    NNPR_ENSURE_STATUS(nnfw_output_size(session, &num_expecteds));

    // verify input and output

    auto verifyInputTypes = [session]() {
      uint32_t sz;
      NNPR_ENSURE_STATUS(nnfw_input_size(session, &sz));
      for (uint32_t i = 0; i < sz; ++i)
      {
        nnfw_tensorinfo ti;
        NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));

        if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
        {
          std::cerr << "E: not supported input type" << std::endl;
          exit(-1);
        }
      }
    };

    auto verifyOutputTypes = [session]() {
      uint32_t sz;
      NNPR_ENSURE_STATUS(nnfw_output_size(session, &sz));

      for (uint32_t i = 0; i < sz; ++i)
      {
        nnfw_tensorinfo ti;
        NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));

        if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
        {
          std::cerr << "E: not supported output type" << std::endl;
          exit(-1);
        }
      }
    };

    verifyInputTypes();
    verifyOutputTypes();

    auto getMetricTypeStr = [](int type) {
      if (type < 0)
        return "";
      // Metric type
      // 0: Categorical Accuracy
      std::vector<int> acc = {0};
      auto it = std::find(acc.begin(), acc.end(), type);
      if (it == acc.end())
        return "metric";
      return "categorical_accuracy";
    };

    // get training information
    nnfw_train_info tri;
    NNPR_ENSURE_STATUS(nnfw_train_get_traininfo(session, &tri));

    // overwrite training information using the arguments
    tri.batch_size = args.getBatchSize().value_or(tri.batch_size);
    tri.learning_rate = args.getLearningRate().value_or(tri.learning_rate);
    tri.loss_info.loss = args.getLossType().value_or(tri.loss_info.loss);
    tri.loss_info.reduction_type =
      args.getLossReductionType().value_or(tri.loss_info.reduction_type);
    tri.opt = args.getOptimizerType().value_or(tri.opt);

    size_t pos = 0;
    tri.trainble_ops_size = args.getTrainableOpsIdx().size();
    for (auto const &idx : args.getTrainableOpsIdx())
    {
      tri.trainble_ops_idx[pos++] = idx;
    }

    std::cout << "== training parameter ==" << std::endl;
    std::cout << tri;
    std::cout << "========================" << std::endl;

    // set training information
    NNPR_ENSURE_STATUS(nnfw_train_set_traininfo(session, &tri));

    // prepare execution

    // TODO When nnfw_{prepare|run} are failed, can't catch the time
    measure.run(PhaseType::PREPARE, [&]() { NNPR_ENSURE_STATUS(nnfw_train_prepare(session)); });

    // prepare input and expected tensor info lists
    std::vector<nnfw_tensorinfo> input_infos;
    std::vector<nnfw_tensorinfo> expected_infos;

    // prepare data buffers
    std::vector<Allocation> input_data(num_inputs);
    std::vector<Allocation> expected_data(num_expecteds);
    std::vector<Allocation> output_data(num_expecteds);

    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));
      input_data[i].alloc(bufsize_for(&ti));
      input_infos.emplace_back(std::move(ti));
    }

    for (uint32_t i = 0; i < num_expecteds; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));

      // For validation
      uint64_t output_size_in_bytes = bufsize_for(&ti);
      output_data[i].alloc(output_size_in_bytes);
      NNPR_ENSURE_STATUS(
        nnfw_train_set_output(session, i, ti.dtype, output_data[i].data(), output_size_in_bytes));

      expected_data[i].alloc(output_size_in_bytes);
      expected_infos.emplace_back(std::move(ti));
    }

    uint32_t tdata_length;
    Generator tdata_generator;
    uint32_t vdata_length;
    Generator vdata_generator;
    std::unique_ptr<DataLoader> dataLoader;

    if (!args.getLoadRawInputFilename().empty() && !args.getLoadRawExpectedFilename().empty())
    {
      dataLoader = std::make_unique<RawDataLoader>(args.getLoadRawInputFilename(),
                                                   args.getLoadRawExpectedFilename(), input_infos,
                                                   expected_infos);

      auto train_to = 1.0f - args.getValidationSplit();
      std::tie(tdata_generator, tdata_length) = dataLoader->loadData(tri.batch_size, 0.f, train_to);
      std::tie(vdata_generator, vdata_length) =
        dataLoader->loadData(tri.batch_size, train_to, 1.0f);
    }
    else
    {
      // TODO Use random generator
      std::cerr << "E: not supported random input and expected generator" << std::endl;
      exit(-1);
    }

    if (tdata_length < tri.batch_size)
    {
      std::cerr << "E: training data is not enough for training."
                   "Reduce batch_size or add more data"
                << std::endl;
      exit(-1);
    }

    // If the user does not give the validation_split value,
    // the vdata_length is 0 by default and it does not execute
    // validation loop.
    if (vdata_length != 0 && vdata_length < tri.batch_size)
    {
      std::cerr << "E: validation data is not enough for validation."
                   "Reduce batch_size or adjust validation_split value"
                << std::endl;
      exit(-1);
    }

    std::vector<float> losses(num_expecteds);
    std::vector<float> metrics(num_expecteds);
    measure.run(PhaseType::EXECUTE, [&]() {
      const int num_step = tdata_length / tri.batch_size;
      const int num_epoch = args.getEpoch();
      measure.set(num_epoch, num_step);
      for (uint32_t epoch = 0; epoch < num_epoch; ++epoch)
      {
        //
        // TRAINING
        //
        {
          std::fill(losses.begin(), losses.end(), 0);
          std::fill(metrics.begin(), metrics.end(), 0);
          for (uint32_t n = 0; n < num_step; ++n)
          {
            // get batchsize data
            if (!tdata_generator(n, input_data, expected_data))
              break;

            // prepare input
            for (uint32_t i = 0; i < num_inputs; ++i)
            {
              NNPR_ENSURE_STATUS(
                nnfw_train_set_input(session, i, input_data[i].data(), &input_infos[i]));
            }

            // prepare output
            for (uint32_t i = 0; i < num_expecteds; ++i)
            {
              NNPR_ENSURE_STATUS(
                nnfw_train_set_expected(session, i, expected_data[i].data(), &expected_infos[i]));
            }

            // train
            measure.run(epoch, n, [&]() { NNPR_ENSURE_STATUS(nnfw_train(session, true)); });

            // store loss
            Metrics metric(output_data, expected_data, expected_infos);
            for (int32_t i = 0; i < num_expecteds; ++i)
            {
              float temp = 0.f;
              NNPR_ENSURE_STATUS(nnfw_train_get_loss(session, i, &temp));
              losses[i] += temp;
              if (args.getMetricType() == 0)
                metrics[i] += metric.categoricalAccuracy(i);
            }
          }

          // print loss
          std::cout << std::fixed;
          std::cout << "Epoch " << epoch + 1 << "/" << num_epoch;
          measure.printTimeMs(epoch, AggregateType::AVERAGE);
          std::cout.precision(4);
          std::cout << " - loss: ";
          for (uint32_t i = 0; i < num_expecteds; ++i)
          {
            std::cout << "[" << i << "] " << losses[i] / num_step;
          }
          // TODO use init-statement in selection statements (c++17)
          std::string str;
          if ((str = getMetricTypeStr(args.getMetricType())) != "")
          {
            std::cout << " - " << str << ": ";
            for (uint32_t i = 0; i < num_expecteds; ++i)
            {
              std::cout << "[" << i << "] " << metrics[i] / num_step;
            }
          }
        }

        //
        // VALIDATION
        //
        if (vdata_length > 0)
        {
          std::fill(losses.begin(), losses.end(), 0);
          std::fill(metrics.begin(), metrics.end(), 0);
          const int num_valid_step = vdata_length / tri.batch_size;
          for (uint32_t n = 0; n < num_valid_step; ++n)
          {
            // get batchsize validation data
            if (!vdata_generator(n, input_data, expected_data))
              break;

            // prepare input
            for (uint32_t i = 0; i < num_inputs; ++i)
            {
              NNPR_ENSURE_STATUS(
                nnfw_train_set_input(session, i, input_data[i].data(), &input_infos[i]));
            }

            // prepare output
            for (uint32_t i = 0; i < num_expecteds; ++i)
            {
              NNPR_ENSURE_STATUS(
                nnfw_train_set_expected(session, i, expected_data[i].data(), &expected_infos[i]));
            }

            // validation
            NNPR_ENSURE_STATUS(nnfw_train(session, false));

            // get validation loss and accuracy
            Metrics metric(output_data, expected_data, expected_infos);
            for (int32_t i = 0; i < num_expecteds; ++i)
            {
              float temp = 0.f;
              NNPR_ENSURE_STATUS(nnfw_train_get_loss(session, i, &temp));
              losses[i] += temp;
              if (args.getMetricType() == 0)
                metrics[i] += metric.categoricalAccuracy(i);
            }
          }

          // print validation loss and accuracy
          std::cout << std::fixed;
          std::cout.precision(4);
          std::cout << " - val_loss: ";
          for (uint32_t i = 0; i < num_expecteds; ++i)
          {
            std::cout << "[" << i << "] " << losses[i] / num_valid_step;
          }
          // TODO use init-statement in selection statements (c++17)
          std::string str;
          if ((str = getMetricTypeStr(args.getMetricType())) != "")
          {
            std::cout << " - val_" << str << ": ";
            for (uint32_t i = 0; i < num_expecteds; ++i)
            {
              std::cout << "[" << i << "] " << metrics[i] / num_valid_step;
            }
          }
        }

        std::cout << std::endl;
      }
    });

    if (args.getExportModelFilename() != "")
      NNPR_ENSURE_STATUS(nnfw_train_export_circle(session, args.getExportModelFilename().c_str()));

    NNPR_ENSURE_STATUS(nnfw_close_session(session));

    measure.printResult();

    return 0;
  }
  catch (boost::program_options::error &e)
  {
    std::cerr << "E: " << e.what() << std::endl;
    exit(-1);
  }
  catch (std::runtime_error &e)
  {
    std::cerr << "E: Fail to run by runtime error:" << e.what() << std::endl;
    exit(-1);
  }
}
