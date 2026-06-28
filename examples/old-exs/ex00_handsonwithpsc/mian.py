from pathlib import Path

import logging
from psc.config.logconfig import setup_logging

from psc.lib.gspacer import GSpacer
from psc.lib.linearizer import EPALinearizer


# change this loca as per wish
log_folder = Path("./results")
setup_logging(log_folder, level=logging.INFO)

gspacer = GSpacer()
linearizer = EPALinearizer()