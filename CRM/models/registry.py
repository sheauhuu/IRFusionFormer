# import mmcv
import mmengine
# from mmcv.utils import Registry
from mmengine import Registry

# def _build_func(name: str, option: mmcv.ConfigDict, registry: Registry):
#     return registry.get(name)(option)
def _build_func(name: str, option: mmengine.ConfigDict, registry: Registry):
    return registry.get(name)(option)

MODELS = Registry('models', build_func=_build_func)
