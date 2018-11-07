import logging
from datetime import datetime


def get_date():
    """Return a string in the format "yyyy-mm-dd" from the current date."""
    return datetime.now().isoformat().split(':')[0].split("T")[0]


class Log:
    """Logging utilities.

    :param log_filenamepath: a relative file path to the log file
    to be created.

    :param level: one of the following:
                    logging.CRITICAL	    50
                    logging.ERROR	    40
                    logging.WARNING	    30
                    logging.INFO	    20
                    logging.DEBUG	    10
    :param loggername: part of the logging line containing the name
    of the file doing logging.
    """
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10

    def __init__(self, log_filenamepath: str, level: int, loggername: str):
        """Instantiate a logging object handler."""

        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(level)

        # create a file handler
        handler = logging.FileHandler(log_filenamepath)
        handler.setLevel(level)

        # create a logging format                  #loggername    #level
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def critical(self, message=""):
        """Log a message at CRITICAL (50) level.
        Regardless of the level set when log was created, this message
        will allways be logged.
        """
        self.logger.critical(message)

    def error(self, message=""):
        """Log a message at ERROR (40) level."""
        self.logger.error(message)

    def warning(self, message=""):
        """Log a message at WARNING (30) level."""
        self.logger.warning(message)

    def info(self, message=""):
        """Log a message at INFO (20) level."""
        self.logger.info(message)

    def debug(self, message=""):
        """Log a message at DEBUG (10) level."""
        self.logger.debug(message)
