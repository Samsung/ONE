import importlib
import sys

from pathlib import Path


# Check operators with final Circle MLIR
def check_model(models_root, model_name, ops_path):
    sys.path.append(models_root + "/net")
    sys.path.append(models_root + "/unit")
    module = importlib.import_module(model_name)

    m_keys = module.__dict__.keys()
    if 'check_circle_operators' in m_keys:
        try:
            opsDict = dict()
            with open(ops_path, 'rt') as file:
                for line in file:
                    op = line.strip()
                    if op in opsDict:
                        opsDict[op] = opsDict[op] + 1
                    else:
                        opsDict[op] = 1

                operators = list(opsDict.keys())
                print("ops dict:", opsDict)
                print("operators:", operators)
                return module.check_circle_operators(opsDict, operators)
        except:
            return 1
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        thispath = Path(sys.argv[0])
        sys.exit("Usage: " + thispath.name + " [models_root] [model_name] [ops_path]")

    result = check_model(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(result)
