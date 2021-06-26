import logging

from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    log = logging.getLogger(name)
    log.setLevel(level)

    # This ensures all logging levels get marked with the rank zero decorator;
    # otherwise, logs would get multiplied for each GPU process in multi-GPU
    # setup.
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(log, level, rank_zero_only(getattr(log, level)))

    return log
