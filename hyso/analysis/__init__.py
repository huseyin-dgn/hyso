"""Pure, explicit evidence indexing and unified reports; no workload execution."""
from ._codec import EvidenceError
from .evidence import EvidenceScope, EvidenceRecord, EvidenceIndex, compare_evidence
from .report import AnalysisContext, UnifiedAnalysisReport

__all__ = ['EvidenceError', 'EvidenceScope', 'EvidenceRecord', 'EvidenceIndex',
           'compare_evidence', 'AnalysisContext', 'UnifiedAnalysisReport']
