import logging
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
from config.config import Config


class LoggerFactory(object):
    _logger = None

    def __init__(self, log_level: str = "INFO") -> None:
        self.log_level = log_level

    def get_logger(self):
        if self._logger is None:
            self._logger = logging.getLogger()
            log_formater = logging.Formatter(
                "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
            )
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formater)
            self._logger.addHandler(console_handler)

            # set the logging level based on the user selection
            if self.log_level == "INFO":
                self._logger.setLevel(logging.INFO)
            elif self.log_level == "ERROR":
                self._logger.setLevel(logging.ERROR)
            elif self.log_level == "DEBUG":
                self._logger.setLevel(logging.DEBUG)
            elif self.log_level == "WARNING":
                self._logger.setLevel(logging.WARNING)
            elif self.log_level == "CRITICAL":
                self._logger.setLevel(logging.CRITICAL)
            parent_log_dir = f'./log/log_{datetime.now().strftime("%m-%d-%Y")}'
            if not os.path.isdir(parent_log_dir):
                os.mkdir(parent_log_dir)
            file_handler = logging.FileHandler(os.path.join(parent_log_dir, "logs.log"))
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(log_formater)

            self._logger.addHandler(file_handler)

        return self._logger


Logger = LoggerFactory("INFO")


def load_data_csv(path: str) -> pd.DataFrame:
    """
    Load file csv
    :params path: path of csv file, end with .csv
    :return:
    """
    try:
        data_df = pd.read_csv(path, lineterminator="\n")
    except OSError as e:
        Logger.get_logger().info(e)
        return pd.DataFrame()
    return data_df


def load_data_npy(path: str) -> List:
    """
    Load file npy
    :params path: path of npy file, end with .npy
    :return:
    """
    try:
        data_list = np.load(path, allow_pickle=True).tolist()
    except OSError as e:
        Logger.get_logger().info(e)
        return []
    return data_list


def save_csv(data_df: pd.DataFrame, path: str) -> None:
    """
    Save file as csv format
    :params data_df: dataframe
    :params path: path of csv file
    :return:
    """
    if isinstance(data_df, pd.DataFrame):
        if data_df.empty:
            message = "Data is empty"
        else:
            if path == "":
                message = "This path is not identify"
            else:
                if not path.endswith(".csv"):
                    message = "This path is not suitable with csv"
                else:
                    data_df.to_csv(path, encoding="utf_8", index=False)
                    message = "Data is saved"
    else:
        message = "This is not a dataframe"
    Logger.get_logger().info(message)
    return None


def set_device(device_id: int = Config.CUDA_CORE) -> torch.device:
    """
    Check cuda is available or not
    :params device: add ID of device
    """
    Logger.get_logger().info("Check cuda is available or not:")

    # Check cuda in device
    if torch.cuda.is_available():
        if device_id != -1:
            torch.cuda.set_per_process_memory_fraction(0.5, device_id)
            device = torch.device(f"cuda:{device_id}")
            Logger.get_logger().info(torch.cuda.get_device_name())
            Logger.get_logger().info("Cuda is available")
        else:
            device = torch.device("cpu")
            Logger.get_logger().info("Cuda is not used")
    else:
        device = torch.device("cpu")
        Logger.get_logger().info("Cuda is not available")
    return device
