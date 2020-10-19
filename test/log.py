from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def test_basic():
    logger = logging.getLogger("filename")
    logger.info("This is a info log.")
    logger.debug("This is a debug log.")
    logger.error("This is a error log.")
    logger.warning("This is a warning log.")


if __name__ == "__main__":
    test_basic()
