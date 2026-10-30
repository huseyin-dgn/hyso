"""Offline live-directory races; no worker launch or native imports."""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from full_acceptance import core,safety


class EvidenceScanTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='hyso-evidence-scan-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def test_atomic_heartbeat_replace_between_listing_and_stat(self):
        target=self.root/'heartbeat.json'
        replace=core.os.replace
        scanned=[]
        def publish_and_scan(temporary,destination):
            def listing(root,pattern):
                # The publisher moves a discovered entry before its metadata
                # is obtained by the supervisor.
                replace(temporary,destination)
                yield temporary
                yield destination
            with patch.object(Path,'rglob',listing):
                scanned.append(safety.stage_evidence_bytes(self.root))
        with patch.object(core.os,'replace',side_effect=publish_and_scan):
            core.publish(target,dict(unix_time=123,scope='liveness'))
        self.assertEqual(scanned,[target.stat().st_size])
        self.assertEqual(core.read(target)['unix_time'],123)
        self.assertEqual(list(self.root.glob('*.tmp')),[])

    def test_file_disappears_after_metadata_is_obtained(self):
        path=self.root/'heartbeat.json.uuid.tmp';path.write_bytes(b'123')
        original=Path.stat;calls=[]
        def stat_then_remove(candidate,*args,**kwargs):
            info=original(candidate,*args,**kwargs)
            if candidate==path:
                calls.append(candidate);candidate.unlink()
            return info
        with patch.object(Path,'rglob',return_value=iter([path])),patch.object(Path,'stat',stat_then_remove):
            self.assertEqual(safety.stage_evidence_bytes(self.root),3)
        self.assertEqual(calls,[path])

    def test_counts_nested_evidence_and_existing_temporary_files(self):
        (self.root/'nested').mkdir()
        (self.root/'nested'/'evidence.json').write_bytes(b'12345')
        (self.root/'heartbeat.json.uuid.tmp').write_bytes(b'123')
        self.assertEqual(safety.stage_evidence_bytes(self.root),8)

    def test_permission_errors_are_not_hidden(self):
        path=self.root/'evidence.json'
        with patch.object(Path,'rglob',return_value=iter([path])),patch.object(Path,'stat',side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):safety.stage_evidence_bytes(self.root)


if __name__=='__main__':unittest.main()
