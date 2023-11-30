import re


def extract_test_args(s):
    p = re.compile('eval\((.*)\)')
    result = p.search(s)
    return result.group(1)


def pytest_addoption(parser):
    parser.addoption("--test_list", action="store", help="Path to test list")
    parser.addoption("--artifacts", action="store", help="Path to test artifacts")
    parser.addoption(
        "--target_artifacts", action="store", help="Path to test target artifacts")
    parser.addoption(
        "--luci_eval_driver", action="store", help="Path to luci eval driver")


def pytest_generate_tests(metafunc):
    list_path = metafunc.config.getoption('test_list')
    artifacts_path = metafunc.config.getoption('artifacts')
    target_artifacts_path = metafunc.config.getoption('target_artifacts')
    eval_driver_path = metafunc.config.getoption('luci_eval_driver')
    if list_path is None:
        tests_default_tol = []
        tests_with_tol = []
        ref_tests_default_tol = []
        ref_tests_with_tol = []
    else:
        with open(list_path) as f:
            contents = [line.rstrip() for line in f]

        comment_removed = [line for line in contents if not line.startswith('#')]
        newline_removed = [line for line in comment_removed if line.startswith('eval(')]
        test_args = [extract_test_args(line) for line in newline_removed]
        # eval(TEST_NAME)
        tests_default_tol = [(arg, artifacts_path, eval_driver_path) for arg in test_args
                             if len(arg.split()) == 1]
        # eval(TEST_NAME RTOL ATOL)
        tests_with_tol = [(arg.split()[0], artifacts_path, eval_driver_path,
                           arg.split()[1], arg.split()[2]) for arg in test_args
                          if len(arg.split()) == 3]

    if 'default_test_name' in metafunc.fixturenames:
        metafunc.parametrize('default_test_name,artifacts_path,eval_driver_path',
                             tests_default_tol)

    if 'tol_test_name' in metafunc.fixturenames:
        metafunc.parametrize(
            'tol_test_name,artifacts_path,eval_driver_path,rtolf32,atolf32',
            tests_with_tol)

    if target_artifacts_path is not None:
        # eval(TEST_NAME)
        ref_tests_default_tol = [(arg, artifacts_path, target_artifacts_path,
                                  eval_driver_path) for arg in test_args
                                 if len(arg.split()) == 1]
        # eval(TEST_NAME RTOL ATOL)
        ref_tests_with_tol = [(arg.split()[0], artifacts_path,
                               target_artifacts_path, eval_driver_path, arg.split()[1],
                               arg.split()[2]) for arg in test_args
                              if len(arg.split()) == 3]
        #
        # for cross platform test
        #
        if 'default_ref_test_name' in metafunc.fixturenames:
            metafunc.parametrize(
                'default_ref_test_name,ref_artifacts_path,target_artifacts_path,eval_driver_path',
                ref_tests_default_tol)

        if 'tol_ref_test_name' in metafunc.fixturenames:
            metafunc.parametrize(
                'tol_ref_test_name,ref_artifacts_path,target_artifacts_path,eval_driver_path,rtolf32,atolf32',
                ref_tests_with_tol)
