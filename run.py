from src.tools.tools import prepareData
import logging
from src.tools.colors_terminal import bcolors
import traceback
import time


def configure_logger():

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    log_format = f'{bcolors.OKBLUE}%(filename)s {bcolors.ENDC}'\
        f'{bcolors.OKCYAN}line: %(lineno)d{bcolors.ENDC}'\
        f' [%(levelname)s]: {bcolors.WARNING}%(message)s{bcolors.ENDC}'

    logging.basicConfig(format=log_format)

    return root


if __name__ == "__main__":

    logger = configure_logger()

    try:
        start_time = time.time()
        prep_obj = prepareData(logger)
        prep_obj.main()
        end_time = time.time()
        print(
            f"{bcolors.FAIL}Run time:"
            f" {bcolors.ENDC} --- {end_time-start_time} --- seconds")
    except Exception:
        logger.error(traceback.format_exc())
