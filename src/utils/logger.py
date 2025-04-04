import inspect
import logging


# Logs an exception message with detailed context,
# including the filename, line number, and function
# name where the exception occurred. Constructs a
# full log message and writes it to the error log.
# This function is used for debugging and tracking
# errors in the application.

# Before function calling must be set up log file!
def log_exception(msg):
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split('/')[-1]
    lineno = frame.f_lineno
    func = frame.f_code.co_name


    full_msg = f"{filename}:{lineno} ({func}) - {msg}"
    logging.error(full_msg)
