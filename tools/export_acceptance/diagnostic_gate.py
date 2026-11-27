"""Stdlib gates kept separate from actual native execution for offline tests."""
from pathlib import Path
from tools.export_acceptance import runner as r


def validate_ticket(path):
    path=Path(path).absolute();root=path.parent
    base=(r.ROOT/'diagnostics').resolve()
    if path.is_symlink() or root.is_symlink() or root.resolve().parent!=base or path.name!='ticket.json':
        raise RuntimeError('DIAGNOSTIC_TICKET_OWNERSHIP_INVALID')
    if (root/'consumed.json').exists():raise RuntimeError('DIAGNOSTIC_ALREADY_CONSUMED_NO_RETRY')
    ticket=r.engine.read_json(path)
    if r.engine.read_json(root/'ticket-seal.json')['sha256']!=r.engine.file_hash(path):raise RuntimeError('DIAGNOSTIC_TICKET_HASH_INVALID')
    if (ticket['root']!=str(root.resolve()) or ticket['operation']!='dll_diagnostic' or
        ticket['automatic_retries']!=0 or ticket['inference_allowed'] is not False or ticket['conversion_allowed'] is not False):
        raise RuntimeError('DIAGNOSTIC_SCOPE_INVALID')
    if r.engine.file_hash(r.ROOT/'plan.json')!=ticket['plan_sha256'] or r.engine.file_hash(r.ROOT/'checkpoint.json')!=ticket['checkpoint_sha256']:
        raise RuntimeError('DIAGNOSTIC_CAMPAIGN_CHANGED')
    if r.snapshot()!=ticket['source_sha256'] or r.engine.user_snapshot()!=ticket['user_files']:
        raise RuntimeError('DIAGNOSTIC_SOURCE_CHANGED')
    plan=r.engine.read_json(r.ROOT/'plan.json')
    if not r.engine.read_json(r.ROOT/'checkpoint.json').get('safety_hold'):raise RuntimeError('DIAGNOSTIC_HOLD_CHANGED')
    return ticket,root,plan
