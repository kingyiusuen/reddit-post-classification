from typing import List

import pytest
import sh


def run_command(command: List[str]):
    """Default method for executing shell commands with pytest."""
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg=msg)


@pytest.mark.training
def test_fast_dev_run():
    """Run 1 train, val, test batch."""
    command = [
        "scripts/train.py",
        "trainer=default",
        "trainer.fast_dev_run=true",
    ]
    run_command(command)
