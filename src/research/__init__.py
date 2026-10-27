"""
Research and statistical analysis suite for subgroup bias and domain shift evaluation.
"""

from .subgroup_analysis import SubgroupEvaluator
from .domain_shift_analysis import DomainShiftEvaluator
from .statistical_tests import StatisticalTester
from .reporting import ResearchReporter

__all__ = [
    "SubgroupEvaluator",
    "DomainShiftEvaluator",
    "StatisticalTester",
    "ResearchReporter",
]
