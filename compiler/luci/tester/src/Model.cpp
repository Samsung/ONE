#include "Model.h"

#include <fstream>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace
{

class FileModel final : public luci::Model
{
public:
  explicit FileModel(const std::string &filename) : _filename(filename) {}

public:
  FileModel(const FileModel &) = delete;
  FileModel(FileModel &&) = delete;

public:
  const ::circle::Model *model(void) override
  {
    std::ifstream file(_filename, std::ios::binary | std::ios::in);
    if (!file.good())
      return nullptr;

    file.unsetf(std::ios::skipws);

    std::streampos fileSize;
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // reserve capacity
    _data.reserve(fileSize);

    // read the data
    file.read(_data.data(), fileSize);
    if (file.fail())
      return nullptr;

    return ::circle::GetModel(_data.data());
  }

private:
  const std::string _filename;
  std::vector<char> _data;
};

} // namespace

namespace luci
{

std::unique_ptr<Model> load_model(const std::string &path)
{
  return std::unique_ptr<Model>{new FileModel(path)};
}

} // namespace luci
