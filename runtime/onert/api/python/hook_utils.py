import os 

def read_env(env_var_name, default_value):
    validators = {
        "PLATFORM": _validate_platform,
        "GLIBC_VERSION": _validate_glibc_version
    }

    value = os.environ.get(env_var_name, None)
    print(f"value={value}")
    if value is None:
        return default_value
    else:
        validate = validators.get(env_var_name, None)
        if validate is not None:
            return validate(value)
        else:
            return value

def _validate_platform(value):
    return value

def _validate_glibc_version(value):
    return value