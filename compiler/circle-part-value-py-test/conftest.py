import re


def extract_test_args(s):
    p = re.compile('add\\((.*)\\)')
    result = p.search(s)
    return result.group(1)


def pytest_addoption(parser):
    parser.addoption("--test_list", action="store", help="Path to test list")
    parser.addoption("--bin_dir", action="store", help="Directory including artifacts")
    parser.addoption(
        "--circle_part_driver", action="store", help="Path to circle part driver")


def pytest_generate_tests(metafunc):
    list_path = metafunc.config.getoption('test_list')
    bin_dir = metafunc.config.getoption('bin_dir')
    circle_part_driver = metafunc.config.getoption('circle_part_driver')

    with open(list_path) as f:
        contents = [line.rstrip() for line in f]

    comment_removed = [line for line in contents if not line.startswith('#')]
    newline_removed = [line for line in comment_removed if line.startswith('add(')]
    test_args = [extract_test_args(line) for line in newline_removed]
    # add(RECIPE_NAME PARTITION_NAME EXPECTED_OUTPUT_COUNT)
    partition_list = [(arg.split()[1], bin_dir, circle_part_driver) for arg in test_args]

    if 'test_name' in metafunc.fixturenames:
        metafunc.parametrize('test_name,bin_dir,part_driver_path', partition_list)
