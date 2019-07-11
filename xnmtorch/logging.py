import os
import sys
from datetime import datetime
from logging import StreamHandler, FileHandler, basicConfig, getLogger

from xnmtorch import settings


def setup_logging():
    os.makedirs(settings.DEFAULT_LOG_DIR, exist_ok=True)

    filename = os.path.join(settings.DEFAULT_LOG_DIR, f"experiment_{datetime.now():%Y%m%d-%H:%M:%S}.log")

    stream_handler = StreamHandler()
    stream_handler.setLevel(settings.LOG_LEVEL_CONSOLE)

    file_handler = FileHandler(filename)
    file_handler.setLevel(settings.LOG_LEVEL_FILE)

    basicConfig(
        datefmt='%H:%M:%S',
        format='{asctime} {name} {levelname}: {message}',
        style='{',
        level=min(settings.LOG_LEVEL_FILE, settings.LOG_LEVEL_CONSOLE),
        handlers=[stream_handler, file_handler]
    )

    root_logger = getLogger()

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            root_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
