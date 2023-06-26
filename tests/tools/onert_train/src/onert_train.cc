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
#include "nnfw.h"
#include "nnfw_util.h"
#include "nnfw_internal.h"
#include "nnfw_experimental.h"
#include "randomgen.h"
#include "rawdataloader.h"

#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <libgen.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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
    benchmark::Phases phases(benchmark::PhaseOption{});

    nnfw_session *session = nullptr;
    NNPR_ENSURE_STATUS(nnfw_create_session(&session));

    // ModelLoad
    phases.run("MODEL_LOAD", [&](const benchmark::Phase &, uint32_t) {
      if (args.useSingleModel())
        NNPR_ENSURE_STATUS(
          nnfw_load_model_from_modelfile(session, args.getModelFilename().c_str()));
      else
        NNPR_ENSURE_STATUS(nnfw_load_model_from_file(session, args.getPackageFilename().c_str()));
    });

    uint32_t num_inputs;
    NNPR_ENSURE_STATUS(nnfw_input_size(session, &num_inputs));

    uint32_t num_expected;
    NNPR_ENSURE_STATUS(nnfw_output_size(session, &num_expected));

    // verify input and output
    auto verifyInputTypes = [session](uint32_t size) {
      for (uint32_t i = 0; i < size; ++i)
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

    auto verifyOutputTypes = [session](uint32_t size) {
      for (uint32_t i = 0; i < size; ++i)
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

    verifyInputTypes(num_inputs);
    verifyOutputTypes(num_expected);

    // prepare training info
    nnfw_train_info tri;
    tri.batch_size = args.getBatchSize();
    tri.learning_rate = args.getLearningRate();

    // prepare execution
    // TODO When nnfw_{prepare|run} are failed, can't catch the time
    phases.run("PREPARE", [&](const benchmark::Phase &, uint32_t) {
      NNPR_ENSURE_STATUS(nnfw_train_prepare(session, &tri));
    });

    // prepare data buffers & info lists
    std::vector<Allocation> input_data(num_inputs * tri.batch_size);
    std::vector<Allocation> expected_data(num_expected * tri.batch_size);

    std::vector<nnfw_tensorinfo> input_infos;
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));

      auto bufsz = bufsize_for(&ti);
      for (uint32_t n = 0; n < tri.batch_size; ++n)
      {
        input_data[i * tri.batch_size + n].alloc(bufsz);
      }
      input_infos.emplace_back(std::move(ti));
    }

    std::vector<nnfw_tensorinfo> expected_infos;
    for (uint32_t i = 0; i < num_expected; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));

      auto bufsz = bufsize_for(&ti);
      for (uint32_t n = 0; n < tri.batch_size; ++n)
      {
        expected_data[i * tri.batch_size + n].alloc(bufsz);
      }
      expected_infos.emplace_back(std::move(ti));
    }

    auto data_length = args.getDataLength();
    Generator generator;
    RawDataLoader rawDataLoader;
    if (!args.getInputData().empty() && !args.getExpectedData().empty())
    {
      generator = rawDataLoader.loadData(args.getInputData(), args.getExpectedData(), input_infos,
                                         expected_infos, data_length, tri.batch_size);
    }
    // TODO Support RamdonGenerator

    const int num_sample = data_length / tri.batch_size;
    const int num_epoch = args.getEpoch();
    for (uint32_t epoch = 0; epoch < num_epoch; ++epoch)
    {
      for (uint32_t n = 0; n < num_sample; ++n)
      {
        // get batchsize data
        if (!generator(n, input_data, expected_data))
          break;

        // prepare input
        for (uint32_t i = 0; i < num_inputs; ++i)
        {
          nnfw_tensorinfo ti(input_infos[i]);
          ti.dims[0] = tri.batch_size;
          NNPR_ENSURE_STATUS(nnfw_train_set_input(session, i, input_data[i].data(), &ti));
          NNPR_ENSURE_STATUS(nnfw_set_input_layout(session, i, NNFW_LAYOUT_CHANNELS_LAST));
        }

        // prepare output
        for (uint32_t i = 0; i < num_expected; ++i)
        {
          NNPR_ENSURE_STATUS(
            nnfw_train_set_expected(session, i, expected_data[i].data(), &expected_infos[i]));
        }

        // train
        phases.run("EXECUTE", [&](const benchmark::Phase &, uint32_t) {
          NNPR_ENSURE_STATUS(nnfw_train(session, true));
        });
      }

      // print loss
      for (uint32_t i = 0; i < num_expected; ++i)
      {
        float loss;
        NNPR_ENSURE_STATUS(nnfw_train_get_loss(session, i, &loss));
        std::cout << "[Epoch " << epoch << "] Output [" << i
                  << "] Loss: " << loss /* << ", Accuracy: " << accuracy*/ << std::endl;
      }
    }

    NNPR_ENSURE_STATUS(nnfw_close_session(session));

    // prepare result
    benchmark::Result result(phases);

    // to stdout
    benchmark::printResult(result);

    return 0;
  }
  catch (boost::program_options::error &e)
  {
    std::cerr << "E: " << e.what() << std::endl;
    exit(-1);
  }
  catch (std::runtime_error &e)
  {
    std::cerr << "E: Fail to train by runtime error:" << e.what() << std::endl;
    exit(-1);
  }
}
