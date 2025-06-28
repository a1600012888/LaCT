import functools
import importlib
import os
from functools import partial
from inspect import isfunction
from dataclasses import dataclass
import copy
from ast import literal_eval


@dataclass
class ObjectParamConfig:
    class_name: str  # name of the Class
    params: dict  # parameters to pass to the Class


def instantiate_from_config(config, split_config=False):
    """
    config:
        class_name: str (module.class)
        other_params, etc. will be passed as a whole config to the class
    """
    if not "class_name" in config:
        raise KeyError("Expected key `class_name` to instantiate.")
    
    # if you want to pass the params to the class, you can do it like this:
    # return get_obj_from_str(config["class_name"])(config.get("params", dict()))
    # if you want to pass the whole config to the class, you can do it like this:
    if split_config:
        # first pop the class_name from the config
        config = copy.deepcopy(config) # avoid modifying the original config
        class_name = config.pop("class_name")
        return get_obj_from_str(class_name)(**config)
    else:
        return get_obj_from_str(config["class_name"])(config)


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def set_nested_key(data, keys, value):
    """Sets value in nested dictionary"""
    key = keys.pop(0)
    try:
        # Converting key to int to support
        key = int(key)
    except ValueError:
        key = key
    if len(keys) > 0:
        if not isinstance(key, int) and key not in data:
            data[key] = {}
        set_nested_key(data[key], keys, value)
    else:
        try:
            # attempt to eval it it (e.g. if bool, number, or etc)
            attempt = literal_eval(value)
        except (SyntaxError, ValueError):
            # if that goes wrong, just use the string
            attempt = value
        data[key] = attempt


def value_type(value):
    """Convert str to bool/int/float if possible"""
    try:
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
    except AttributeError:
        return value