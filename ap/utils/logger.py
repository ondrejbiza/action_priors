import os
import logging
import sys


class Logger:

    def __init__(self, save_file=None, print_logs=True):

        self.save_file = save_file
        self.print_logs = print_logs

        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers = []

        if self.save_file is not None:

            file_dir = os.path.dirname(self.save_file)

            if len(file_dir) > 0 and not os.path.isdir(file_dir):
                os.makedirs(file_dir)

            file_handler = logging.FileHandler(self.save_file)
            file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(file_handler)

        if self.print_logs:

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def close(self):

        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
