import os
import logging


def init_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, 'out.log')

    fmt = "%(asctime)s %(levelname)s %(message)s"
    date_fmt = "%d %b %H:%M:%S"
    formatter = logging.Formatter(fmt, date_fmt)

    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
