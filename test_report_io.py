import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from report_io import atomic_text


class ReportIOTests(unittest.TestCase):
    def test_transient_reader_lock_recovers(self):
        original = Path.replace
        calls = []
        def locked_once(path, target):
            calls.append(target)
            if len(calls) == 1:
                raise PermissionError('Simulated reader lock')
            return original(path, target)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'progress.json'
            with patch.object(Path, 'replace', locked_once), patch('report_io.time.sleep'):
                self.assertTrue(atomic_text(path, 'saved'))
            self.assertEqual(path.read_text(), 'saved')
            self.assertEqual(len(calls), 2)

    def test_locked_optional_report_does_not_stop_discovery(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(Path, 'replace', side_effect=PermissionError('locked')), patch('report_io.time.sleep'):
                self.assertFalse(atomic_text(Path(root)/'progress.json', 'data', required=False))
                with self.assertRaises(PermissionError):
                    atomic_text(Path(root)/'manifest.json', 'data', required=True)


if __name__ == '__main__':
    unittest.main()
