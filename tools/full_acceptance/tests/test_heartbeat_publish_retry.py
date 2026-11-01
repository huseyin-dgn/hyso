"""Offline tests for transient heartbeat rename failures and external outputs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from full_acceptance import core


class AtomicPublishTests(unittest.TestCase):
    def test_temporary_permission_denial_is_retried_without_losing_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'heartbeat.json'
            core.publish(path,dict(number=1))
            original=core.os.replace
            calls=[]
            def deny_once(source,destination):
                calls.append((source,destination))
                if len(calls)==1:raise PermissionError('transient WinError 5')
                return original(source,destination)
            with patch.object(core.os,'replace',side_effect=deny_once),patch.object(core.time,'sleep'):
                core.publish(path,dict(number=2))
            self.assertEqual(json.loads(path.read_text())['number'],2)
            self.assertEqual(len(calls),2)
            self.assertFalse(list(Path(directory).glob('*.tmp')))

    def test_persistent_permission_denial_fails_bounded_and_preserves_old(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'heartbeat.json'
            core.publish(path,dict(number=1))
            with patch.object(core.os,'replace',side_effect=PermissionError('permanent denial')),\
                 patch.object(core.time,'monotonic',side_effect=[0.,6.]),\
                 patch.object(core.time,'sleep') as sleep:
                with self.assertRaises(PermissionError):
                    core.publish(path,dict(number=2))
                sleep.assert_not_called()
            self.assertEqual(json.loads(path.read_text())['number'],1)
            self.assertFalse(list(Path(directory).glob('*.tmp')))

    def test_output_exclusion_is_none_outside_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory).resolve()
            self.assertFalse(out.is_relative_to(core.ROOT))
            self.assertIsNone(out.relative_to(core.ROOT).as_posix() if out.is_relative_to(core.ROOT) else None)
