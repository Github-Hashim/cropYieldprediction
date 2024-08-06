from src.cropYieldprediction.logger import logging
from src.cropYieldprediction.exception import CustomException
import sys

if __name__ == "__main__":
    logging.info("The Execution has started.")
    try:
        a = 1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)