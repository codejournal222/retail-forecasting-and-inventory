import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest


class MeasuredResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_ids.append(test.id())


def run(output):
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2, resultclass=MeasuredResult).run(suite)
    record = {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "tests_passed": len(result.passed_ids),
        "passed_test_ids": result.passed_ids,
        "failure_details": [
            {"test": str(test), "traceback": trace}
            for test, trace in result.failures + result.errors
        ],
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob("*.py")
        },
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="test_results.json")
    raise SystemExit(0 if run(parser.parse_args().output)["status"] == "PASS" else 1)
