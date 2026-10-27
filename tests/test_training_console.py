import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.name == "posix","native logging uses POSIX file descriptors")
class ConsoleTests(unittest.TestCase):
    def test_python_native_and_stderr_are_saved_and_terminal_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"run.log")
            code = '''
from supermimic_study.training.console import capture_console
import os,sys,ctypes
with capture_console(sys.argv[1]):
    print("python-output",flush=True)
    os.write(2,b"native-stderr\\n")
    ctypes.CDLL(None).puts(b"libc-output")
print("after-context",flush=True)
'''
            result = subprocess.run([sys.executable,"-c",code,path],text=True,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            saved = Path(path).read_text()
            for expected in ("python-output","native-stderr","libc-output"):
                self.assertIn(expected,saved)
                self.assertIn(expected,result.stdout)
            self.assertNotIn("after-context",saved)
            self.assertIn("after-context",result.stdout)
