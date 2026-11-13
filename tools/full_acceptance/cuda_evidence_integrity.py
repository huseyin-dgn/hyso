"""Additional on-disk evidence verification for native CUDA subtask PASS.

Not a substitute for OS Job admission, sealed manifest verification, or GPU
measurement validation. Does not import torch or launch work.
"""
import hashlib
import json
from pathlib import Path


def _digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _json(path):
    def unique(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate evidence JSON key')
            out[k] = v
        return out
    def invalid_constant(x):
        raise ValueError('nonfinite evidence JSON value')
    with path.open('r', encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique, parse_constant=invalid_constant)


def verify_native_pass(task_id, manifest_sha256, observed, expected_root):
    """Fail closed when a claimed PASS has absent/changed/misbound native files."""
    root = Path(expected_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('CUDA subtask evidence root is not a directory')
    if observed.get('status') != 'PASS' or observed.get('id') != task_id:
        raise ValueError('native PASS identity required')
    if Path(observed.get('evidence_directory', '')).resolve() != root:
        raise ValueError('native evidence directory mismatch')
    result_path = root / 'result.json'
    receipt_path = root / 'supervisor-receipt.json'
    ready_path = root / 'ready.json'
    if Path(observed.get('result_file', '')).resolve() != result_path:
        raise ValueError('native result file outside task directory')
    for p in (result_path, receipt_path, ready_path):
        if not p.is_file() or p.is_symlink():
            raise ValueError('missing or linked native proof: ' + p.name)
    result, receipt, ready = map(_json, (result_path, receipt_path, ready_path))
    if (result.get('status') != 'PASS' or result.get('id') != task_id or
            result.get('manifest_sha256') != manifest_sha256):
        raise ValueError('native worker terminal mismatch')
    if (receipt.get('experiment') != task_id or
            receipt.get('manifest_sha256') != manifest_sha256 or
            receipt.get('operation') != 'inference_cuda_subtask' or
            receipt.get('exit_code') != 0 or receipt.get('reason') is not None or
            receipt.get('native_crash') is not False or
            receipt.get('owned_job_only') is not True or
            receipt.get('cleanup', {}).get('active_processes') != 0):
        raise ValueError('native supervisor proof mismatch')
    if (ready.get('manifest_sha256') != manifest_sha256 or
            ready.get('gate', {}).get('exact_job_membership') is not True or
            ready.get('gate', {}).get('kill_on_close') is not True):
        raise ValueError('missing exact Windows Job admission proof')
    evidence = result.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('PASS without measurement evidence')
    for item in evidence:
        if not isinstance(item, dict) or not all(k in item for k in ('path','sha256','bytes')):
            raise ValueError('missing measurement evidence inventory')
        candidate = Path(item['path']).resolve(strict=True)
        if (not candidate.is_relative_to(root) or not candidate.is_file() or
                candidate.is_symlink() or candidate in (result_path, receipt_path)):
            raise ValueError('unowned measurement evidence path')
        if candidate.stat().st_size != item['bytes'] or _digest(candidate) != item['sha256']:
            raise ValueError('changed measurement evidence')
    return {'task_id': task_id, 'verified_artifacts': len(evidence),
            'evidence_integrity': True, 'scope': 'on_disk_native_pass_proof_only'}
