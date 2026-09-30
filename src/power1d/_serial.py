'''
Serialisable form of power1d objects.

obj._to_dict() returns a plain dictionary (None, bool, int, float, str, numpy arrays, lists and
nested dictionaries) and _from_dict(d) rebuilds an equivalent object. Every dictionary carries a
"class" key naming the power1d class. Random values are not stored: a rebuilt noise or data
model draws new random values, exactly as a newly constructed one does.

Copyright (C) 2026  Todd Pataky
'''

import inspect

import numpy as np


# constructor arguments whose value is not stored under the same attribute name
_ATTR = {
    'Additive':          {'noise_models': 'models'},
    'Mixture':           {'noise_models': 'models'},
    'Experiment':        {'data_sample_models': 'data_models'},
    'RegionOfInterest':  {'x': 'value'},
}


def _classes():
    from . import _base, geom, noise, models, roi, results
    out = {}
    for mod in (_base, geom, noise, models, roi, results):
        for name, c in vars(mod).items():
            if inspect.isclass(c) and c.__module__ == mod.__name__ and not name.startswith('_'):
                out[name] = c
    return out


def _encode(v):
    if hasattr(v, '_to_dict'):
        return v._to_dict()
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.copy()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise ValueError(f'cannot serialise a value of type {type(v).__name__}')


def _numeric(v):
    if isinstance(v, list):
        return all(_numeric(x) for x in v)
    return isinstance(v, (int, float))


def _decode(v):
    if isinstance(v, dict) and 'class' in v:
        return from_dict(v)
    if isinstance(v, list):
        if v and _numeric(v):
            return np.asarray(v)     # an array that travelled as a plain list (e.g. through JSON)
        return [_decode(x) for x in v]
    return v


def to_dict(obj):
    '''Generic serialisation from the constructor signature (see _ATTR for renamed attributes).'''
    name = type(obj).__name__
    if getattr(obj, 'islinked', False):
        raise ValueError('linked noise models cannot be serialised')
    d = {'class': name}
    rename = _ATTR.get(name, {})
    for pname, p in list(inspect.signature(type(obj).__init__).parameters.items())[1:]:
        attr = rename.get(pname, pname)
        if pname == 'fn':
            d[pname] = _fn_name(obj)
        else:
            d[pname] = _encode(getattr(obj, attr))
    return d


def _fn_name(obj):
    from . import stats
    fn = obj.fn
    if type(obj).__name__ == 'SignalDependent':
        if getattr(obj, '_fn_default', False):
            return None
        raise ValueError('a SignalDependent noise model with a custom "fn" cannot be serialised')
    factory = getattr(fn, '_factory', None)
    if factory is not None:
        name, args = factory
        return dict(factory=name, args=[_encode(a) for a in args])
    name = getattr(fn, '__name__', None)
    if name and getattr(stats, name, None) is fn:
        return name
    raise ValueError('only test statistic functions from power1d.stats can be serialised')


def from_dict(d):
    '''Rebuild a power1d object from its dictionary.'''
    classes = _classes()
    name = d['class']
    if name not in classes:
        raise ValueError(f'unknown power1d class: {name!r}')
    cls = classes[name]
    if hasattr(cls, '_from_dict_custom'):
        return cls._from_dict_custom(d)
    args, kwargs = [], {}
    for pname, p in list(inspect.signature(cls.__init__).parameters.items())[1:]:
        if pname not in d:
            continue
        v = d[pname]
        if pname == 'fn':
            if v is None:
                continue
            from . import stats
            if isinstance(v, dict):
                v = getattr(stats, v['factory'])(*[np.asarray(a) if isinstance(a, list) else a for a in v['args']])
            else:
                v = getattr(stats, v)
        else:
            v = _decode(v)
        if p.kind == p.VAR_POSITIONAL:
            args.extend(v)
        elif p.kind == p.POSITIONAL_OR_KEYWORD and not kwargs and p.default is p.empty:
            args.append(v)
        else:
            kwargs[pname] = v
    return cls(*args, **kwargs)
