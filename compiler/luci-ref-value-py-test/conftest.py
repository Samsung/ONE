import re
import os
import shutil


def extract_test_args(s):
    p = re.compile('eval\\((.*)\\)')
    result = p.search(s)
    return result.group(1)


def pytest_addoption(parser):
    parser.addoption("--test_list", action="store", help="Path to test list")
    parser.addoption("--artifacts", action="store", help="Path to test artifacts")
    parser.addoption("--tflrecipe", action="store", help="Path to tfl recipies")
    parser.addoption("--circlerecipe", action="store", help="Path to circle recipies")
    parser.addoption("--binary", action="store", help="Path to test binary")
    parser.addoption(
        "--luci_eval_driver", action="store", help="Path to luci eval driver")


def copy_if_changed(src_filepath, dst_filepath):
    do_copy = False
    if (os.path.isfile(dst_filepath)):
        file_diff = os.stat(src_filepath).st_mtime - os.stat(dst_filepath).st_mtime
        if file_diff > 1:
            print("file:" + src_filepath + " changed, update")
            do_copy = True
    else:
        do_copy = True

    if do_copy:
        print("file:" + src_filepath + " copy to: " + dst_filepath)
        shutil.copyfile(src_filepath, dst_filepath)


# prepare reference input/output files to build folder for luci-eval-driver
# from ref data in res/TensorFlowLiteRecipes/*/ref.input* and ref.output*
# as model_name.ref.input* and model_name.ref.output*
def copy_ref_files(ref_file_src, ref_file_dst):
    num_data = 0
    while True:
        input_file_src = ref_file_src + str(num_data)
        if (not os.path.isfile(input_file_src)):
            break
        input_file_dst = ref_file_dst + str(num_data)
        copy_if_changed(input_file_src, input_file_dst)
        # try next file
        num_data = num_data + 1


# copy circle mode from common-artifacts to build binary
def copy_circle_model(model_src, model_dst):
    copy_if_changed(model_src, model_dst)


def prepare_materials(test_name, tflrecipe_path, circlerecipe_path, binary_path,
                      artifacts_path):
    # tfl? or circle?
    recipe_path = tflrecipe_path
    # check with 'test.recipe' file as 'ref.input?' can be absent for no input model
    test_recipe = os.path.join(recipe_path, test_name, 'test.recipe')
    if (not os.path.isfile(test_recipe)):
        recipe_path = circlerecipe_path

    ref_input_src = os.path.join(recipe_path, test_name, 'ref.input')
    ref_input_dst = os.path.join(binary_path, test_name + '.ref.input')
    copy_ref_files(ref_input_src, ref_input_dst)

    ref_input_src = os.path.join(recipe_path, test_name, 'ref.output')
    ref_input_dst = os.path.join(binary_path, test_name + '.ref.output')
    copy_ref_files(ref_input_src, ref_input_dst)

    cirle_model_src = os.path.join(artifacts_path, test_name + '.circle')
    cicle_model_dst = os.path.join(binary_path, test_name + '.circle')
    copy_circle_model(cirle_model_src, cicle_model_dst)


def pytest_generate_tests(metafunc):
    list_path = metafunc.config.getoption('test_list')
    artifacts_path = metafunc.config.getoption('artifacts')
    tflrecipe_path = metafunc.config.getoption('tflrecipe')
    circlerecipe_path = metafunc.config.getoption('circlerecipe')
    binary_path = metafunc.config.getoption('binary')
    eval_driver_path = metafunc.config.getoption('luci_eval_driver')
    if list_path is None:
        tests_default_tol = []
        tests_with_tol = []
    else:
        with open(list_path) as f:
            contents = [line.rstrip() for line in f]

        comment_removed = [line for line in contents if not line.startswith('#')]
        newline_removed = [line for line in comment_removed if line.startswith('eval(')]
        test_args = [extract_test_args(line) for line in newline_removed]
        # eval(TEST_NAME)
        tests_default_tol = [(arg, binary_path, eval_driver_path) for arg in test_args
                             if len(arg.split()) == 1]
        # eval(TEST_NAME RTOL ATOL)
        tests_with_tol = [(arg.split()[0], binary_path, eval_driver_path, arg.split()[1],
                           arg.split()[2]) for arg in test_args if len(arg.split()) == 3]

        # copy circle file to binary
        for test_item in tests_default_tol:
            prepare_materials(test_item[0], tflrecipe_path, circlerecipe_path,
                              binary_path, artifacts_path)

        for test_item in tests_with_tol:
            prepare_materials(test_item[0], tflrecipe_path, circlerecipe_path,
                              binary_path, artifacts_path)

    if 'default_test_name' in metafunc.fixturenames:
        metafunc.parametrize('default_test_name,binary_path,eval_driver_path',
                             tests_default_tol)

    if 'tol_test_name' in metafunc.fixturenames:
        metafunc.parametrize('tol_test_name,binary_path,eval_driver_path,rtolf32,atolf32',
                             tests_with_tol)
