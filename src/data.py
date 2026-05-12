"""
Data loaders for AdvBench (harmful) and Alpaca (harmless).
Mirrors llm_activation_control/utils.py but adds --small mode support.
"""
import io
from functools import lru_cache

import pandas as pd
import requests
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main"
    "/data/advbench/harmful_behaviors.csv"
)


@lru_cache(maxsize=1)
def get_harmful_instructions():
    response = requests.get(ADVBENCH_URL)
    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = df["goal"].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


@lru_cache(maxsize=1)
def get_harmless_instructions():
    dataset = load_dataset("tatsu-lab/alpaca")
    instructions = [
        item["instruction"]
        for item in dataset["train"]
        if item["input"].strip() == ""
    ]
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def load_data(n_train: int | None = None, n_test: int | None = None):
    """
    Returns (harmful_train, harmful_test, harmless_train, harmless_test).
    Pass n_train / n_test to subsample for fast iteration (--small mode).
    """
    harmful_train, harmful_test = get_harmful_instructions()
    harmless_train, harmless_test = get_harmless_instructions()

    if n_train is not None:
        harmful_train = harmful_train[:n_train]
        harmless_train = harmless_train[:n_train]
    if n_test is not None:
        harmful_test = harmful_test[:n_test]
        harmless_test = harmless_test[:n_test]

    return harmful_train, harmful_test, harmless_train, harmless_test
