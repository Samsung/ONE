/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "onnx2circle.h"
#include "cmdOptions.h"

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>

#define DEBUG_TYPE "o2c"
#include <llvm/Support/Debug.h>

#include <cstdlib>
#include <iostream>

#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>

namespace opts
{

llvm::cl::OptionCategory O2CirCat("onnx2circle options");
llvm::cl::OptionCategory O2CObsol("obsolete options");

static llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional, llvm::cl::desc("<onnx>"),
                                                llvm::cl::Required, llvm::cl::cat(O2CirCat));

static llvm::cl::opt<std::string> OutputFilename(llvm::cl::Positional, llvm::cl::desc("<circle>"),
                                                 llvm::cl::Required, llvm::cl::cat(O2CirCat));

// Note: If you look at the help desctription in this debug version program,
// it is configured to accept <input file> as the third Positional argument
// which is set from the `onnx-mlir`, but this is not used in actual work.

static llvm::cl::opt<bool> OptSaveOPS("save_ops", llvm::cl::desc(__opt_save_ops),
                                      llvm::cl::init(false), llvm::cl::cat(O2CirCat));

static llvm::cl::opt<bool> RunSingleInstance("o2c-single", llvm::cl::desc("run single instance"),
                                             llvm::cl::init(false), llvm::cl::cat(O2CirCat));

static llvm::cl::opt<bool> OptUnrollRNN("unroll_rnn", llvm::cl::desc(__opt_unroll_rnn_d),
                                        llvm::cl::init(false), llvm::cl::cat(O2CirCat));

static llvm::cl::opt<bool> OptUnrollLSTM("unroll_lstm", llvm::cl::desc(__opt_unroll_lstm_d),
                                         llvm::cl::init(false), llvm::cl::cat(O2CirCat));

static llvm::cl::opt<bool> OptExpDisBMMUnfold("experimental_disable_batchmatmul_unfold",
                                              llvm::cl::desc(__opt_edbuf_d), llvm::cl::init(false),
                                              llvm::cl::cat(O2CirCat));

static llvm::cl::opt<bool> OptKeepIOOrder("keep_io_order", llvm::cl::desc(__opt_keep_io_order_d),
                                          llvm::cl::init(false), llvm::cl::cat(O2CObsol));

static llvm::cl::opt<bool> OptSaveIntermediate("save_intermediate",
                                               llvm::cl::desc(__opt_save_int_d),
                                               llvm::cl::init(false), llvm::cl::cat(O2CObsol));

// shape inference validation
static llvm::cl::opt<bool> OptCheckShapeInf("check_shapeinf", llvm::cl::desc(__opt_check_shapeinf),
                                            llvm::cl::init(false), llvm::cl::cat(O2CirCat));
static llvm::cl::opt<bool> OptCheckDynShapeInf("check_dynshapeinf",
                                               llvm::cl::desc(__opt_check_dynshapeinf),
                                               llvm::cl::init(false), llvm::cl::cat(O2CirCat));

} // namespace opts

class SingleRun
{
public:
  static void Ensure(void)
  {
    SingleRun::_lock_fd = -1;
    int rc = -1;
    int retry = 100; // retry for 10 seconds

    do
    {
      rc = -1;
      SingleRun::_lock_fd = open(_lock_file, O_CREAT | O_RDWR, 0660);
      if (_lock_fd >= 0)
      {
        rc = flock(SingleRun::_lock_fd, LOCK_EX | LOCK_NB);
        if (rc == 0)
          break;
        close(SingleRun::_lock_fd);
        SingleRun::_lock_fd = -1;
      }
      usleep(100 * 1000); // wait for 100 msecs
      if (--retry < 0)
      {
        std::cerr << "Failed to SingleRun::Ensure." << std::endl;
        break;
      }
    } while (rc != 0);
  }

  static void Release(void)
  {
    if (SingleRun::_lock_fd >= 0)
    {
      close(_lock_fd);
      _lock_fd = -1;
    }
    if (_lock_file)
    {
      unlink(_lock_file);
    }
  }

private:
  inline static int _lock_fd = -1;
  inline static const char *const _lock_file = "/tmp/onnx2cirlce_run_single.lock";
};

void onexit() { SingleRun::Release(); }

int main(int argc, char *argv[])
{
  std::atexit(onexit);

  llvm::cl::ParseCommandLineOptions(argc, argv, "");

  LLVM_DEBUG({
    llvm::dbgs() << "onnx2circle debug enter\n";
    llvm::dbgs() << "Source model: " << opts::InputFilename << "\n";
    llvm::dbgs() << "Target model: " << opts::OutputFilename << "\n";
  });

  if (!llvm::sys::fs::exists(opts::InputFilename))
  {
    std::cerr << "Source model: " << opts::InputFilename << " not found." << std::endl;
    return -1;
  }

  if (opts::RunSingleInstance)
    SingleRun::Ensure();

  O2Cparam param;
  param.sourcefile = opts::InputFilename;
  param.targetfile = opts::OutputFilename;
  param.save_ops = opts::OptSaveOPS;
  param.unroll_rnn = opts::OptUnrollRNN;
  param.unroll_lstm = opts::OptUnrollLSTM;
  param.unfold_batchmatmul = !opts::OptExpDisBMMUnfold;
  param.check_shapeinf = opts::OptCheckShapeInf;
  param.check_dynshapeinf = opts::OptCheckDynShapeInf;

  auto result = entry(param);
  LLVM_DEBUG({ llvm::dbgs() << "Conversion done: " << result << "\n"; });
  return result;
}
