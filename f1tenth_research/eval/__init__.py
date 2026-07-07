"""Evaluation module for F1Tenth RMA"""

try:
    from .generalization_sweep import GeneralizationEvaluator
    __all__ = ['GeneralizationEvaluator']
except ImportError:
    # generalization_sweep depends on matplotlib, which isn't always
    # installed (e.g. on the lab machine's training container). Other
    # eval modules (quick_eval, sanity_check_rollout) don't need it,
    # so don't let this block them from being imported.
    GeneralizationEvaluator = None
    __all__ = []
