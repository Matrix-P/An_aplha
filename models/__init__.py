__version__ = "0.1.0"
__author__ = "UsuAo"

from .base import ModelBase, cross_section_normalize
from .linear import EqualWeight, ICIRWeighted, RidgeRegression
from .evaluation import evaluate_model, compare_models

__all__ = [
    'ModelBase', 'cross_section_normalize',
    'EqualWeight', 'ICIRWeighted', 'RidgeRegression',
    'evaluate_model', 'compare_models',
]
