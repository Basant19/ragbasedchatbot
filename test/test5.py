# test5.py

import sys
import os

# Add the parent directory to the Python path so we can import from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exception import CustomException
from logging_config import logger

def raise_custom_exception():
    try:
        # Intentional error: divide by zero
        result = 10 / 0
    except Exception as e:
        raise CustomException(e, sys)

def main():
    logger.info("Starting test5.py to check custom exception and logger.")

    try:
        raise_custom_exception()
    except CustomException as ce:
        logger.error(ce)
        print("Custom exception caught:")
        print(ce)

    logger.info("Finished executing test5.py.")

if __name__ == "__main__":
    main()
