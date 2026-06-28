import os
import logging
from datetime import datetime



class PSCFormatter(logging.Formatter):

    def format(self, record):
        if record.name == "root":
            return record.getMessage()

        record.name = record.name.split(".")[-1]
        return super().format(record)


def setup_logging(log_path, level=logging.INFO):
    
    os.makedirs(log_path, exist_ok=True)

    logfile = os.path.join( log_path, f"{datetime.now():%Y%m%d-%H%M%S}-psc.log")

    formatter = PSCFormatter("[%(name)s | %(levelname)s] %(message)s")
    # formatter = PSCFormatter("[%(name)s%(levelname)s] %(message)s")
    
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # logging.basicConfig( level=level, handlers=[ file_handler, console_handler ] ) 
    root = logging.getLogger()
    
    # clear old handlers
    if root.hasHandlers():
        root.handlers.clear()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    
    # logging.info("PSC - Crystal structure determination - started")
    PSC_BANNER = r"""    
                #        ______     _____    _____
                #        |  __ \   / ____|  / ____|
                #        | |__) | | (___   | |     
                #        |  ___/   \___ \  | |     
                #        | |       ____) | | |____ 
                #        |_|      |_____/   \_____|
                
                # Parameter Space Concept in Crystallography
                """
    
    logging.info(f"# {'=' * 100} #")
    logging.info(f"{PSC_BANNER}")
    logging.info(f"# {'=' * 100} #\n")
    
    return logfile

