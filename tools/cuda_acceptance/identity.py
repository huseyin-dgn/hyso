"""Canonical physical UUIDs; importing this module never loads a GPU runtime."""
from uuid import UUID


def gpu_uuid(value):
    if isinstance(value,bytes):
        parsed=UUID(bytes=value) if len(value)==16 else UUID(value.decode('ascii').removeprefix('GPU-'))
    else:parsed=UUID(str(value).removeprefix('GPU-'))
    return 'GPU-'+str(parsed)


def parent_binding(ids,ready,binding):
    """Same interpreter or one venv redirector hop, as admit_interpreter enforces."""
    launcher=ids['launcher']['pid'];interpreter=ids['interpreter']['pid']
    return (ids['launcher']==binding['identity'] and ready['pid']==interpreter
        and ready['parent_pid']==ids['parent_pid']
        and (interpreter==launcher or ids['parent_pid']==launcher))
