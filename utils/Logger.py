import logging


class ModelLog:
    def __init__(self, filename):
        logging.basicConfig(filename=filename)

    def log(self):
        pass