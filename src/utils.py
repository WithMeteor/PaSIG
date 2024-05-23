#!/user/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys


def setup_logger(logger_name, filename, delete_old=False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(filename, mode='w') if delete_old else logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger
