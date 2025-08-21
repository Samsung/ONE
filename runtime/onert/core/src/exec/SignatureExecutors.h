/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_EXEC_SIGNATURE_EXECUTORS_H__
#define __ONERT_EXEC_SIGNATURE_EXECUTORS_H__

#include "exec/IExecutors.h"
#include "ir/NNPkg.h"

#include "SingleModelExecutors.h"

namespace onert::exec
{

/**
 * @brief Class to gather executor set for signature entry
 *        Actually it is wrapper of IExecutors(SignatureExecutors)
 */
class SignatureExecutors : public SingleModelExecutors
{
public:
  /**
   * @brief Construct a new SignatureExecutors object
   */
  SignatureExecutors(void) = default;
  SignatureExecutors(const SignatureExecutors &) = delete;
  SignatureExecutors(SignatureExecutors &&) = default;

  /**
   * @brief Destroy the SignatureExecutors object
   */
  ~SignatureExecutors() = default;

  /**
   * @brief     Convert IExecutors to SignatureExecutors
   * @param[in] executors Executors object to convert
   * @param[in] signature signature name of the executors
   * @param[in] index     subgraph index of the signature
   */
  SignatureExecutors(const std::shared_ptr<IExecutors> &executors, const std::string &signature,
                     const ir::SubgraphIndex &entry_index);

public:
  IExecutor *entryExecutor() const override;

private:
  const std::shared_ptr<IExecutors> _executors;
  std::string _signature;
  ir::SubgraphIndex _entry_index;
};

} // namespace onert::exec

#endif // __ONERT_EXEC_SIGNATURE_EXECUTORS_H__
