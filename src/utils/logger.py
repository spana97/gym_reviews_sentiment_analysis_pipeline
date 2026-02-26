import logging


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Formatters
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")  # noqa: E501
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # Handlers
    file_handler = logging.FileHandler("appLogger.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(console_formatter)

    # Add handlers only if logger has none
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()
