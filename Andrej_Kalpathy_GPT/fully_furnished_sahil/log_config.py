import logging

# create a log file for each run
# import datetime
# log_file_name = datetime.datetime.now().strftime("./logs/logging_%Y-%m-%d_%H-%M-%S.log")
# with open(log_file_name, "w"):
#     pass


# overwriting one logging file before a run
log_file_name = "./logs/logging.log"
with open(log_file_name, "w"):
    pass

# variable checker logger
var_chk_logger = logging.getLogger("variable_checker")
var_chk_logger.setLevel(logging.DEBUG)
var_chk_handler = logging.FileHandler(log_file_name)
var_chk_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
var_chk_handler.setFormatter(var_chk_formatter)
var_chk_logger.addHandler(var_chk_handler)


# script running logger
script_run_logger = logging.getLogger("script_runner")
script_run_logger.setLevel(logging.INFO)
script_run_handler = logging.FileHandler(log_file_name)
script_run_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
script_run_handler.setFormatter(script_run_formatter)
script_run_logger.addHandler(script_run_handler)
