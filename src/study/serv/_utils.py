"""utils of services"""
import csv
import logging

LOGGER = logging.getLogger(__name__)


def get_txt(pth: str, eos: str = '') -> list[str]:
    """get text data from flat files and add trailing EOS token"""
    with open(pth, newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        return [i[0] + eos for i in csv_reader]
