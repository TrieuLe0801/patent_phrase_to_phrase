import json
import os

from dotenv import load_dotenv


load_dotenv()

basedir = os.path.abspath(os.path.dirname(__file__))
config_path = "./config/config.json"
config = json.load(open(config_path))


class Config:
    MAX_LENGTH = (
        os.getenv("MAX_LENGTH") if os.getenv("MAX_LENGTH") else config["MAX_LENGTH"]
    )
    BATCH_SIZE = (
        os.getenv("BATCH_SIZE") if os.getenv("BATCH_SIZE") else config["BATCH_SIZE"]
    )
    DROPOUT_RATE = (
        os.getenv("DROPOUT_RATE")
        if os.getenv("DROPOUT_RATE")
        else config["DROPOUT_RATE"]
    )
    PRETRAINED_MODEL = (
        os.getenv("PRETRAINED_MODEL")
        if os.getenv("PRETRAINED_MODEL")
        else config["PRETRAINED_MODEL"]
    )
    SAVE_MODEL_PATH = (
        os.getenv("SAVE_MODEL_PATH")
        if os.getenv("SAVE_MODEL_PATH")
        else config["SAVE_MODEL_PATH"]
    )
    SAVE_MODEL_NAME = (
        os.getenv("SAVE_MODEL_NAME")
        if os.getenv("SAVE_MODEL_NAME")
        else config["SAVE_MODEL_NAME"]
    )
    DATA_PATH = (
        os.getenv("DATA_PATH") if os.getenv("DATA_PATH") else config["DATA_PATH"]
    )
    DATA_TRAIN_FILE = (
        os.getenv("DATA_TRAIN_FILE")
        if os.getenv("DATA_TRAIN_FILE")
        else config["DATA_TRAIN_FILE"]
    )
    DATA_TEST_FILE = (
        os.getenv("DATA_TEST_FILE")
        if os.getenv("DATA_TEST_FILE")
        else config["DATA_TEST_FILE"]
    )
