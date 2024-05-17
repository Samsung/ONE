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

#include "exporter/CircleExporter.h"

#include "circle_schema_generated.h"
#include "exec/Execution.h"
#include "ir/train/TrainingInfo.h"

#include <fcntl.h>    // O_RDONLY
#include <sys/mman.h> // mmap, munmap
#include <sys/stat.h> // fstat
#include <unistd.h>   // close
#include <stdint.h>   // SIZE_MAX

#include <fstream>
#include <iostream>

namespace onert
{
namespace exporter
{

class MMappedFile
{
public:
  MMappedFile(const char *filename) { _fd = open(filename, O_RDWR); }
  ~MMappedFile() { close(); }

  bool ensure_mmap()
  {
    struct stat file_stat;
    if (fstat(_fd, &file_stat) != 0 || file_stat.st_size < 0 ||
        static_cast<uint64_t>(file_stat.st_size) > SIZE_MAX)
      return false;

    _buf_sz = static_cast<size_t>(file_stat.st_size);
    _buf = mmap(NULL, _buf_sz, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
    return _buf != MAP_FAILED;
  }

  bool sync() { return msync(_buf, _buf_sz, MS_SYNC) == 0; }

  bool close()
  {
    bool ret = false;
    if (_buf != MAP_FAILED)
    {
      ret = munmap(_buf, _buf_sz) == 0;
      _buf = MAP_FAILED; // mark as cleaned up
    }
    if (_fd != -1)
    {
      ::close(_fd);
      _fd = -1; // mark as cleaned up
    }
    return ret;
  }

  uint8_t *buf() const { return static_cast<uint8_t *>(_buf); }
  size_t buf_size() const { return _buf_sz; }

private:
  int _fd;
  void *_buf = MAP_FAILED;
  size_t _buf_sz = 0;
};

CircleExporter::CircleExporter(const std::string &source, const std::string &path) : _path{path}
{
  std::ifstream src(source.c_str(), std::ios::binary);
  std::ofstream dst(path.c_str(), std::ios::binary);
  dst << src.rdbuf();

  _mmapfile = std::make_unique<MMappedFile>(path.c_str());
  if (!_mmapfile->ensure_mmap())
    throw std::runtime_error("Failed to ensure mmap file");

  // make sure the architecture is little endian before direct access to flatbuffers
  assert(FLATBUFFERS_LITTLEENDIAN);
}

CircleExporter::~CircleExporter() { finish(); }

void CircleExporter::updateWeight(const std::unique_ptr<onert::exec::Execution> &exec)
{
  exec->iterateTrainableTensors(
    [&](const onert::ir::OperandIndex &idx, const onert::backend::train::ITrainableTensor *tensor) {
      auto model = ::circle::GetModel(_mmapfile->buf());
      if (!model)
        throw std::runtime_error("Failed to get model from circle");

      auto subgs = model->subgraphs();
      if (!subgs || subgs->size() != 1)
        throw std::runtime_error("Circle does not has valid subgraph or has multiple subgraphs");

      auto subg = subgs->Get(0); // Get 1st subgraph
      if (!idx.valid() || idx.value() >= subg->tensors()->size())
        throw std::runtime_error("Trainable tensor index is out of range");

      auto buf_idx = subg->tensors()->Get(idx.value())->buffer();
      const ::circle::Buffer *buffer = (*model->buffers())[buf_idx];
      if (!buffer || !buffer->data())
        throw std::runtime_error("Buffer for trainable tensors is invalid");

      const flatbuffers::Vector<uint8_t> *array = buffer->data();
      if (!array)
        throw std::runtime_error("Data for trainable tensor's buffer is invalid");

      auto org_buf_sz = array->size();
      if (org_buf_sz != tensor->total_size())
        throw std::runtime_error("Trained tensor buffer size does not match original tensor's one");

      uint8_t *org_buf = const_cast<uint8_t *>(array->Data());
      if (!org_buf)
        throw std::runtime_error("Data for trainable tensor's buffer is invalid");

      memcpy(const_cast<uint8_t *>(org_buf), tensor->buffer(), org_buf_sz);
    });
}

void CircleExporter::updateMetadata(
  const std::unique_ptr<onert::ir::train::TrainingInfo> &training_info)
{
  UNUSED_RELEASE(training_info);
}

void CircleExporter::finish()
{
  if (_mmapfile->sync() == false)
    throw std::runtime_error("Failed to sync Circle file");

  if (_mmapfile->close() == false)
    throw std::runtime_error("Failed to close Circle file");
}
} // namespace exporter
} // namespace onert
