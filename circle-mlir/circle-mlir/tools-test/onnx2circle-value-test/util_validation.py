import numpy as np
import json


# Get PEIR(Peak Error-to-Interval Ratio) from two outputs
# Each output is an array, and it returns PEIR as string type.
def get_peir_from(baseline_output, target_output):
    if baseline_output.dtype == 'bool':
        diff = np.absolute(baseline_output ^ target_output).reshape(-1)
    else:
        diff = np.absolute(baseline_output - target_output).reshape(-1)

    baseline_min = np.min(baseline_output.reshape(-1))
    baseline_max = np.max(baseline_output.reshape(-1))

    PEAK_ERROR = np.max(diff)
    if baseline_max.dtype == 'bool':
        # ignore PEIR for bool type
        PEIR = 0
    else:
        INTERVAL = baseline_max - baseline_min
        if INTERVAL == 0:
            # Set to infinity because INTERVAL is zero
            PEIR = np.inf
        else:
            PEIR = PEAK_ERROR / INTERVAL

    PEIR_res = '{:.4f}'.format(PEIR) if PEIR <= 1.0 else '> 1'

    return PEIR_res


def save_output_peir_to_json(peir_results, target_model_name):
    try:
        with open(str(target_model_name) + '.peir.json', 'w') as res_file:
            json.dump(peir_results, res_file)
        return True
    except:
        print('Failed to save PEIR to .json file')
        return False


def load_output_peir_from_json(target_model_name):
    try:
        res_file = open(target_model_name + '.peir.json', 'r')
        peir_content = json.load(res_file)
        return peir_content
    except:
        print('Failed to load PEIR from .json file')
        return {}
