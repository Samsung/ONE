import re


def extract_test_args(s):
    p = re.compile('eval\\((.*)\\)')
    result = p.search(s)
    return result.group(1)


def pytest_addoption(parser):
    parser.addoption("--test_list", action="store", help="Path to test list")
    parser.addoption(
        "--tflite_dir", action="store", help="Directory including tflite file")
    parser.addoption(
        "--circle_dir", action="store", help="Directory including circle file")
    parser.addoption(
        "--luci_eval_driver", action="store", help="Path to luci eval driver")


def pytest_generate_tests(metafunc):
    list_path = metafunc.config.getoption('test_list')
    tflite_dir = metafunc.config.getoption('tflite_dir')
    circle_dir = metafunc.config.getoption('circle_dir')
    eval_driver_path = metafunc.config.getoption('luci_eval_driver')
    if list_path is None:
        tests_default_tol = []
    else:
        with open(list_path) as f:
            contents = [line.rstrip() for line in f]

        comment_removed = [line for line in contents if not line.startswith('#')]
        newline_removed = [line for line in comment_removed if line.startswith('eval(')]
        test_args = [extract_test_args(line) for line in newline_removed]
        # eval(TEST_NAME PASS_1 PASS_2 ..)
        tests_default_tol = [(arg.split()[0], tflite_dir, circle_dir, eval_driver_path)
                             for arg in test_args]

    if 'test_name' in metafunc.fixturenames:
        metafunc.parametrize('test_name,tflite_dir,circle_dir,eval_driver_path',
                             tests_default_tol)
