import logging

LOG_FORMAT = "[%(asctime)s] {%(filename)s:%(lineno)s} %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
