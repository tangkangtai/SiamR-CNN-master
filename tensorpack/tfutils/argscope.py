# -*- coding: utf-8 -*-
# File: argscope.py

import copy
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from inspect import getmembers, isfunction

from ..utils import logger
from .tower import get_current_tower_context

__all__ = ['argscope', 'get_arg_scope', 'enable_argscope_for_module',
           'enable_argscope_for_function']

_ArgScopeStack = []


@contextmanager
def argscope(layers, **kwargs):
    """
    Args:
        layers (list or layer): layer or list of layers to apply the arguments.
        # 层(列表或层):应用参数的层或层列表
    Returns:
        a context where all appearance of these layer will by default have the
        arguments specified by kwargs.
        # 在上下文中，这些层的所有出现将默认具有kwargs指定的参数
    Example:
        .. code-block:: python

            with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):
                x = Conv2D('conv0', x)
                x = Conv2D('conv1', x)
                x = Conv2D('conv2', x, out_channel=64)  # override argscope

    """
    if not isinstance(layers, list):
        layers = [layers]

    # def _check_args_exist(l):
    #     args = inspect.getargspec(l).args
    #     for k, v in six.iteritems(kwargs):
    #         assert k in args, "No argument {} in {}".format(k, l.__name__)

    for l in layers:
        assert hasattr(l, 'symbolic_function'), "{} is not a registered layer".format(l.__name__)
        # _check_args_exist(l.symbolic_function)

    new_scope = copy.copy(get_arg_scope())
    for l in layers:
        new_scope[l.__name__].update(kwargs)
    _ArgScopeStack.append(new_scope)
    yield
    del _ArgScopeStack[-1]


def get_arg_scope():
    """
    Returns:
        dict: the current argscope.

    An argscope is a dict of dict: ``dict[layername] = {arg: val}``
    """
    if len(_ArgScopeStack) > 0:
        return _ArgScopeStack[-1]
    else:
        return defaultdict(dict)


def enable_argscope_for_function(func, log_shape=True):
    """Decorator for function to support argscope

    Example:

        .. code-block:: python

            from mylib import myfunc
            myfunc = enable_argscope_for_function(myfunc)

    Args:
        func: function which should be decorated.
        log_shape (bool): print input/output shapes of each function.

    Returns:
        The decorated function.

    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        actual_args = copy.copy(get_arg_scope()[func.__name__])
        actual_args.update(kwargs)
        out_tensor = func(*args, **actual_args)
        in_tensor = args[0]

        ctx = get_current_tower_context()
        name = func.__name__ if 'name' not in kwargs else kwargs['name']
        if log_shape:
            if ('tower' not in ctx.ns_name.lower()) or ctx.is_main_training_tower:
                logger.info('%20s: %20s -> %20s' %
                            (name, in_tensor.shape.as_list(),
                             out_tensor.shape.as_list()))

        return out_tensor
    # argscope requires this property
    wrapped_func.symbolic_function = None
    return wrapped_func


def enable_argscope_for_module(module, log_shape=True):
    """
    Overwrite all functions of a given module to support argscope.
    Note that this function monkey-patches the module and therefore could
    have unexpected consequences.
    It has been only tested to work well with `tf.layers` module.

    Example:

        .. code-block:: python

            import tensorflow as tf
            enable_argscope_for_module(tf.layers)

    Args:
        log_shape (bool): print input/output shapes of each function.
    """
    for name, obj in getmembers(module):
        if isfunction(obj):
            setattr(module, name, enable_argscope_for_function(obj,
                    log_shape=log_shape))
