import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name="PSC", logdir=None):

    if logdir is None:
        logdir = Path.cwd()

    logdir = Path(logdir)
    logdir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logfile = logdir / f"{timestamp}_psc.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # prevent duplicate handlers when notebook cell is run again
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Log file created: {logfile}")

    return logger