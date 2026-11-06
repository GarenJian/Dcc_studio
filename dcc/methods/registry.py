"""Registry of available dynamic connectivity analysis methods.

The registry is the single place where the application asks "which methods
exist?". It decouples discovery from implementation:

===========================  ===============================================
Consumer                     What it asks the registry
===========================  ===============================================
``dcc.pipelines``            ``get_method(cfg.method)`` before a run
``dcc.config``               nothing (stores only the selected name)
``gui.app``                  ``method_choices()`` to fill the selector
``docs/METHOD.md``           documents how to register a new method
===========================  ===============================================

Registration is explicit rather than based on dynamic imports or entry
points. That keeps the frozen PyInstaller build small and makes the set of
available methods deterministic and easy to audit. A third-party extension
only needs to import this module and call ``register_method`` before the
workflow starts.

Thread-safety: registration is expected to happen during import, before any
worker thread is started. Lookups are plain dictionary reads afterwards.
"""
from __future__ import annotations

from typing import Iterable

from .base import AnalysisMethod

# One shared instance per method. Instances are stateless, so reuse is safe and
# avoids re-creating objects for every sample or every GUI refresh.
_REGISTRY: dict[str, AnalysisMethod] = {}


def register_method(method: AnalysisMethod | type[AnalysisMethod]):
    """Register a method instance (or class) and return it unchanged.

    Can be used as a class decorator::

        @register_method
        class FlexibilityMethod(AnalysisMethod):
            name = "flexibility"
            ...

    or called directly with an instance. Re-registering an existing name is
    rejected instead of being silently overwritten: two different methods
    sharing a name would make cached results and ``run_tag`` ambiguous.

    Raises
    ------
    ValueError
        If the name is empty, contains characters that are unsafe in file
        names, or is already registered to a different class.
    """
    instance = method() if isinstance(method, type) else method
    if not isinstance(instance, AnalysisMethod):
        raise TypeError("register_method expects an AnalysisMethod subclass "
                        "or instance")
    if not instance.name or not all(c.isalnum() or c == "_"
                                    for c in instance.name):
        raise ValueError(f"Invalid method name: {instance.name!r}")
    existing = _REGISTRY.get(instance.name)
    if existing is not None and type(existing) is not type(instance):
        raise ValueError(f"Method '{instance.name}' is already registered by "
                         f"{type(existing).__name__}")
    _REGISTRY[instance.name] = instance
    return method


def get_method(name: str | None) -> AnalysisMethod:
    """Return the method registered under ``name``.

    ``None`` and the empty string both select the first registered method,
    which is the default DCC implementation. An unknown name raises a
    ``ValueError`` listing the available choices; the workflow surfaces this
    message directly in the GUI instead of failing deep inside the algorithm.
    """
    if not name:
        if not _REGISTRY:
            raise ValueError("No analysis methods are registered")
        return next(iter(_REGISTRY.values()))
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_REGISTRY)) or "none"
        raise ValueError(f"Unknown analysis method '{name}'. "
                         f"Available: {choices}") from exc


def available_methods() -> list[AnalysisMethod]:
    """Return all registered methods in a stable (alphabetical) order."""
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def method_choices() -> list[tuple[str, str]]:
    """Return ``(name, title)`` pairs suitable for a ``QComboBox``."""
    return [(method.name, method.title) for method in available_methods()]


def method_descriptions() -> dict[str, str]:
    """Return ``{name: description}`` for tooltips and generated docs."""
    return {method.name: method.description
            for method in available_methods()}


def registered_names() -> Iterable[str]:
    """Iterable of registered names, useful in diagnostics and tests."""
    return tuple(_REGISTRY)


def clear_registry() -> None:
    """Remove all registrations.

    Intended for isolated tests that want to verify registration behaviour;
    production code never calls it (``dcc.methods.__init__`` re-registers the
    built-in methods on the next import).
    """
    _REGISTRY.clear()
