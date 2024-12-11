import sys


def get_error_details(error, error_traceback: sys):
    """
    Extracts detailed error information, including file name, line number, and error message.
    """
    _, _, traceback = error_traceback.exc_info()
    file_name = traceback.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: {file_name}, line: {traceback.tb_lineno}, message: {str(error)}"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_traceback: sys):
        super().__init__(error_message)
        self.error_message = get_error_details(error_message, error_traceback)

    def __str__(self):
        return self.error_message

