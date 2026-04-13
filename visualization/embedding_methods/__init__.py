"""
visualization/embedding_methods/
=================================

Plugin registry for 2-D dimensionality-reduction methods used by the
Visualizer (voxel51_vis.py).

Built-in methods
----------------
- ``"umap"``  — UMAP via umap-learn
- ``"tsne"``  — t-SNE via sklearn

Adding a custom method
----------------------
Decorate your subclass with ``@register("name")`` anywhere before the
Visualizer runs (e.g. in your own script or a config-time import)::

    from visualization.embedding_methods import register, EmbeddingMethod

    @register("pca")
    class PCAMethod(EmbeddingMethod):
        def __init__(self, n_components=2, **kwargs):
            from sklearn.decomposition import PCA
            self._r = PCA(n_components=n_components)

        def fit_transform(self, embeddings):
            return self._r.fit_transform(embeddings)

Then set ``embedding_methods: ["pca"]`` in your visualization config and
it will be picked up automatically.
"""

from .base import EmbeddingMethod
from .umap_method import UMAPMethod
from .tsne_method import TSNEMethod

# ── Global registry ────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[EmbeddingMethod]] = {
    "umap": UMAPMethod,
    "tsne": TSNEMethod,
}


def register(name: str):
    """
    Class decorator — registers an EmbeddingMethod under *name*.

    Args:
        name: The string key used in ``embedding_methods`` config lists.

    Raises:
        TypeError: if the decorated class does not subclass EmbeddingMethod.
    """
    def decorator(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, EmbeddingMethod)):
            raise TypeError(
                f"@register target must be a subclass of EmbeddingMethod, got {cls!r}"
            )
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_method(name: str, **kwargs) -> EmbeddingMethod:
    """
    Instantiate a registered embedding method by name.

    Args:
        name:    Method name (case-insensitive), e.g. ``"umap"``, ``"tsne"``.
        **kwargs: Constructor keyword arguments forwarded to the method class.

    Returns:
        An EmbeddingMethod instance ready to call ``.fit_transform()``.

    Raises:
        ValueError: if *name* is not in the registry.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown embedding method '{name}'. "
            f"Available: {list(_REGISTRY)}. "
            "Register custom methods with @embedding_methods.register('name')."
        )
    return _REGISTRY[key](**kwargs)


def list_methods() -> list[str]:
    """Return the names of all currently registered embedding methods."""
    return list(_REGISTRY.keys())


__all__ = [
    "EmbeddingMethod",
    "UMAPMethod",
    "TSNEMethod",
    "register",
    "get_method",
    "list_methods",
]
