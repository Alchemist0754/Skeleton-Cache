from .base import BackboneAdapter
from .purls import PurlsBackbone
from .sa_dvae import SaDvaeBackbone
from .smie import SmieBackbone
from .synse import SynseBackbone

_REGISTRY: dict[str, type[BackboneAdapter]] = {
    "smie": SmieBackbone,
    "purls": PurlsBackbone,
    "synse": SynseBackbone,
    "sa_dvae": SaDvaeBackbone,
}


def backbone_class(name: str) -> type[BackboneAdapter]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backbone {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


__all__ = [
    "BackboneAdapter", "SmieBackbone", "PurlsBackbone",
    "SynseBackbone", "SaDvaeBackbone", "backbone_class",
]
