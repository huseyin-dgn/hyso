"""Read-only, bilingual presentation of retained HYSo results."""
from .catalog import feature_explanations
from .report import PresentationReport, ReportError, create_report, report_experiment

def render_export(result, *, format='markdown', language='en') -> str:
    """Render saved deployment evidence using the shared export presentation."""
    from ..export.presentation import render
    return render(result,format=format,language=language)

__all__ = ["PresentationReport", "ReportError", "create_report", "report_experiment", "feature_explanations", "render_export"]
