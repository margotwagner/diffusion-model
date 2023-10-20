from utils.diffusion_utils import *
import numpy as np


def main():

    # TODO: run and save data from finite diff experiment
    FD_DIR = "../data/fdm-diffusion/08242023/u.npy"

    #SD_DIR = "../data/spectral-diffusion/08202023/u.npy"
    SD_DIR = "../data/spectral-diffusion/eigenmode-exps/5/u.npy"

    SAVE_DIR = "../figures/08312023/"

    ca_percent_error(FD_DIR, SD_DIR, SAVE_DIR)


if __name__ == "__main__":
    main()
