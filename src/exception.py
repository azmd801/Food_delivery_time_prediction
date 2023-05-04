import sys
from src.logger import logging

import sys

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Returns a formatted error message with details about where the error occurred.

    Args:
        error (Exception): The error that occurred.
        error_detail (sys): Additional details about the error.

    Returns:
        str: A string containing details about the error, including the name of the Python script where the error occurred,
        the line number where the error occurred, and the error message itself.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

import sys

class CustomException(Exception):
    """
    Custom exception class that provides a formatted error message with details about where the error occurred.

    Attributes:
        error_message (str): A formatted error message with details about where the error occurred.

    Args:
        error_message (str): The error message to be displayed.
        error_detail (sys): Additional details about the error.

    """

    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message


   
    

# if __name__=="__main__":
#     logging.info("Logging has started")
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info('Dicision by zero') 
#         raise CustomException(e,sys)
