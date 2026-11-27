"""Marker-owned staging, no-clobber publication and multi-file commit manifests."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import uuid
from .contracts import ExportError

def file_hash(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for slab in iter(lambda:stream.read(1024**2),b''):h.update(slab)
    return h.hexdigest()

def safe_file(root: Path, relative: str) -> Path:
    name=Path(relative)
    if not relative or name.is_absolute() or '..' in name.parts or ':' in relative or '\\' in relative:raise ExportError('ARTIFACT_PATH','unsafe manifest path')
    target=root/name
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):raise ExportError('ARTIFACT_PATH','linked/escaping artifact')
    for p in (target,*target.parents):
        if p==root:break
        if p.is_symlink() or os.name=='nt' and p.exists() and p.lstat().st_file_attributes & 0x400:raise ExportError('ARTIFACT_PATH','reparse artifact')
    return target

def verify_files(body: dict) -> None:
    rows=body.get('artifact_files',[])
    if not rows:raise ExportError('ARTIFACT_INTEGRITY','artifact manifest missing')
    root=Path(body['artifact_root']);seen=set()
    for row in rows:
        if row['path'] in seen:raise ExportError('ARTIFACT_INTEGRITY','duplicate manifest member')
        seen.add(row['path']);p=safe_file(root,row['path'])
        if not p.is_file() or p.stat().st_size!=row['bytes'] or file_hash(p)!=row['sha256']:raise ExportError('ARTIFACT_INTEGRITY','missing/modified artifact '+row['path'])
    main=Path(body['artifact_path'])
    if not main.is_relative_to(root) or main.relative_to(root).as_posix() not in seen:raise ExportError('ARTIFACT_INTEGRITY','main artifact not bound to manifest')

def owned_files(root):
    pending=[(root,0)];count=0
    while pending:
        directory,depth=pending.pop()
        if depth>16:raise ExportError('OWNERSHIP','staging tree exceeds bounded depth')
        for p in directory.iterdir():
            count+=1
            if count>4096:raise ExportError('OWNERSHIP','staging tree exceeds bounded entry count')
            if p.is_symlink() or os.name=='nt' and p.lstat().st_file_attributes & 0x400:raise ExportError('OWNERSHIP','refusing reparse staging traversal')
            if p.is_dir():pending.append((p,depth+1))
            elif p.is_file():yield p
            else:raise ExportError('OWNERSHIP','nonordinary staging member')

def onnx_tensors(graph,depth=0):
    """All external-data capable tensors, including constants and subgraphs."""
    if depth>16:raise ExportError('ARTIFACT_PATH','ONNX graph nesting exceeds bounded profile')
    if hasattr(graph,'graph'):
        yield from onnx_tensors(graph.graph,depth+1)
        for function in graph.functions:yield from onnx_tensors(function,depth+1)
        return
    yield from getattr(graph,'initializer',())
    for sparse in getattr(graph,'sparse_initializer',()):
        yield sparse.values;yield sparse.indices
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField('t'):yield attr.t
            yield from attr.tensors
            if attr.HasField('sparse_tensor'):
                yield attr.sparse_tensor.values;yield attr.sparse_tensor.indices
            for sparse in attr.sparse_tensors:yield sparse.values;yield sparse.indices
            if attr.HasField('g'):yield from onnx_tensors(attr.g,depth+1)
            for nested in attr.graphs:yield from onnx_tensors(nested,depth+1)

class Transaction:
    """One owned generation; manifest last is the multi-file commit point.

    Readers must verify the commit metadata, never consume a partial generation.
    Existing unrelated sidecar files are never adopted. Rollback only touches this
    transaction's files and explicitly authorized overwritten main/metadata files.
    """
    def __init__(self, output: str | Path, overwrite: bool, limit: int):
        self.output=Path(output).absolute();self.parent=self.output.parent.resolve(strict=True)
        if self.output.is_symlink() or self.output.parent.resolve()!=self.parent:raise ExportError('ARTIFACT_PATH','invalid output')
        self.output=self.parent/self.output.name;self.manifest=self.output.with_name(self.output.name+'.export.json')
        if self.output.name=='.owned.json' or self.output.name.startswith('.backup-'):raise ExportError('ARTIFACT_PATH','reserved staging filename')
        self.overwrite=overwrite
        if (self.output.exists() or self.manifest.exists()) and not overwrite:raise ExportError('EXISTS','output/metadata exists')
        if self.output.exists() and not self.output.is_file():raise ExportError('ARTIFACT_PATH','output is not an ordinary file')
        if self.manifest.exists():
            from .contracts import ExportResult
            previous=ExportResult.from_json(self.manifest)
            if previous.to_dict()['artifact_path']!=str(self.output):raise ExportError('OWNERSHIP','foreign export sidecar')
        self.token=uuid.uuid4().hex;self.root=self.parent/('.hyso-export-'+self.token)
        self.root.mkdir();(self.root/'.owned.json').write_text(json.dumps({'token':self.token,'scope':'hyso.export'}))
        self.limit=limit;self.published=[];self.backups=[];self.committed=False
    def __enter__(self):return self
    def inventory(self) -> list[dict]:
        rows=[];total=0
        for p in sorted(owned_files(self.root)):
            if p.name=='.owned.json':continue
            relative=p.relative_to(self.root).as_posix();safe_file(self.root,relative)
            total+=p.stat().st_size
            if total>self.limit:raise ExportError('RESOURCE_BUDGET','artifact exceeds byte budget')
            rows.append(dict(path=relative,sha256=file_hash(p),bytes=p.stat().st_size))
        if not rows:raise ExportError('ARTIFACT_INTEGRITY','conversion produced no files')
        return rows
    def publish(self, result) -> None:
        rows=result.to_dict()['artifact_files']
        # External files use unique names, never replace user-owned data.
        for row in rows:
            dest=safe_file(self.parent,row['path'])
            if dest!=self.output and dest.exists():raise ExportError('EXISTS','external artifact exists')
        try:
            for old in (self.output,self.manifest):
                if old.exists():
                    if not self.overwrite:raise ExportError('EXISTS','output appeared during conversion')
                    if old.is_symlink():raise ExportError('OWNERSHIP','linked output/sidecar')
                    backup=self.root/('.backup-'+str(len(self.backups)));os.replace(old,backup);self.backups.append((old,backup))
            for row in rows:
                src=safe_file(self.root,row['path']);dest=safe_file(self.parent,row['path']);dest.parent.mkdir(parents=True,exist_ok=True)
                # link is atomic no-clobber on this same filesystem.
                os.link(src,dest);self.published.append(dest)
            result.verify()
            with self.manifest.open('x',encoding='utf-8') as f:
                self.published.append(self.manifest);f.write(result.to_json());f.flush();os.fsync(f.fileno())
            self.committed=True
        finally:
            if not self.committed:
                for p in reversed(self.published):p.unlink()
                for old,backup in reversed(self.backups):os.replace(backup,old)
    def __exit__(self,*args):
        marker=self.root/'.owned.json'
        try:bound=json.loads(marker.read_text())
        except (OSError,ValueError) as exc:raise ExportError('OWNERSHIP','refusing cleanup without valid owned marker') from exc
        if self.root.is_symlink() or os.name=='nt' and self.root.lstat().st_file_attributes & 0x400 or bound!={'token':self.token,'scope':'hyso.export'}:raise ExportError('OWNERSHIP','refusing unowned staging cleanup')
        try:
            list(owned_files(self.root));shutil.rmtree(self.root)
        except OSError as exc:raise ExportError('OWNERSHIP','owned staging cleanup incomplete; no verified cleanup claim') from exc
