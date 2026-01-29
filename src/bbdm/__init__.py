"""
BBDM - Brownian Bridge Diffusion Model for Chess Image Translation.

This module provides the BBDM implementation for synthetic-to-real
chess board image translation.
"""

from src.bbdm.utils import dict2namespace, namespace2dict, get_runner
from src.bbdm.Register import Registers

# Import modules to trigger registration with @Registers decorators
from src.bbdm import dataloader  # noqa: F401 - registers datasets
from src.bbdm import runners  # noqa: F401 - registers runners

__all__ = [
    'dict2namespace',
    'namespace2dict', 
    'get_runner',
    'Registers',
]
