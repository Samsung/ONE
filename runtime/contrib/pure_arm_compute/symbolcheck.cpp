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

#include <NeuralNetworks.h>

#include <iostream>

#define SHOW_FUNC_ENTRY(name)                                                       \
  {                                                                                 \
    std::cout << #name << " is at " << reinterpret_cast<void *>(name) << std::endl; \
  }

int main(int argc, char **argv)
{
  SHOW_FUNC_ENTRY(ANeuralNetworksMemory_createFromFd);
  SHOW_FUNC_ENTRY(ANeuralNetworksMemory_free);

  SHOW_FUNC_ENTRY(ANeuralNetworksModel_create);
  SHOW_FUNC_ENTRY(ANeuralNetworksModel_addOperand);
  SHOW_FUNC_ENTRY(ANeuralNetworksModel_setOperandValue);
  SHOW_FUNC_ENTRY(ANeuralNetworksModel_setOperandValueFromMemory);
  SHOW_FUNC_ENTRY(ANeuralNetworksModel_addOperation);
  SHOW_FUNC_ENTRY(ANeuralNetworksModel_identifyInputsAndOutputs);
  SHOW_FUNC_ENTRY(ANeuralNetworksModel_finish);
  SHOW_FUNC_ENTRY(ANeuralNetworksModel_free);

  SHOW_FUNC_ENTRY(ANeuralNetworksCompilation_create);
  SHOW_FUNC_ENTRY(ANeuralNetworksCompilation_finish);
  // ANeuralNetworksCompilation_setPreference and ANeuralNetworksCompilation_free
  // are introduced to reuse NNAPI tests under runtimes/tests. Note that these APIs
  // are not necessary for supporting Tensorflow Lite interperter
  SHOW_FUNC_ENTRY(ANeuralNetworksCompilation_setPreference);
  SHOW_FUNC_ENTRY(ANeuralNetworksCompilation_free);
  SHOW_FUNC_ENTRY(ANeuralNetworksCompilation_create);

  SHOW_FUNC_ENTRY(ANeuralNetworksExecution_create);
  SHOW_FUNC_ENTRY(ANeuralNetworksExecution_setInput);
  SHOW_FUNC_ENTRY(ANeuralNetworksExecution_setOutput);
  SHOW_FUNC_ENTRY(ANeuralNetworksExecution_startCompute);
  SHOW_FUNC_ENTRY(ANeuralNetworksExecution_free);

  SHOW_FUNC_ENTRY(ANeuralNetworksEvent_wait);
  SHOW_FUNC_ENTRY(ANeuralNetworksEvent_free);

  // NOTE Pure CL runtime does not implement following NN API(s) as
  //      Tensorflow Lite does not use these API(s)
  // SHOW_FUNC_ENTRY(ANeuralNetworksExecution_setInputFromMemory);
  // SHOW_FUNC_ENTRY(ANeuralNetworksExecution_setOutputFromMemory);

  return 0;
}
