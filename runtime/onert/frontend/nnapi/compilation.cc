/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <NeuralNetworks.h>

#include <new>

#include "wrapper/ANeuralNetworksModel.h"
#include "wrapper/ANeuralNetworksCompilation.h"
#include "util/logging.h"

//
// NNAPI Implementation
//
int ANeuralNetworksCompilation_create(ANeuralNetworksModel *model,
                                      ANeuralNetworksCompilation **compilation)
{
  if ((model == nullptr) || (compilation == nullptr))
  {
    VERBOSE(NNAPI::Compilation) << "create: Incorrect null pointer parameter(s)" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (!model->isFinished())
  {
    VERBOSE(NNAPI::Compilation) << "create: Model define is not finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  *compilation = new (std::nothrow) ANeuralNetworksCompilation(model);
  if (*compilation == nullptr)
  {
    VERBOSE(NNAPI::Compilation) << "create: ail to create compilation object" << std::endl;
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation *compilation)
{
  if (compilation == nullptr)
  {
    VERBOSE(NNAPI::Compilation) << "finish: Incorrect null pointer parameter" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (compilation->isFinished())
  {
    VERBOSE(NNAPI::Compilation) << "finish: Already finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  if (!compilation->finish())
  {
    VERBOSE(NNAPI::Compilation) << "finish: Fail to compile" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation *compilation)
{
  delete compilation;
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation *compilation,
                                             int32_t preference)
{
  if (compilation == nullptr)
  {
    VERBOSE(NNAPI::Compilation) << "setPreference: Incorrect null pointer parameter" << std::endl;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (compilation->isFinished())
  {
    VERBOSE(NNAPI::Compilation) << "setPreference: Already finished" << std::endl;
    return ANEURALNETWORKS_BAD_STATE;
  }

  const PreferenceCode FIRST_PREFERENCE_CODE = ANEURALNETWORKS_PREFER_LOW_POWER;
  const PreferenceCode LAST_PREFERENCE_CODE = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED;
  if ((preference < FIRST_PREFERENCE_CODE) || (preference > LAST_PREFERENCE_CODE))
  {
    VERBOSE(NNAPI::Compilation) << "setPreference: Incorrect preference code" << std::endl;
    return ANEURALNETWORKS_BAD_DATA;
  }

  // NYI: nothing to set
  return ANEURALNETWORKS_NO_ERROR;
}
