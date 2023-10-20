from utils.diffusion_utils import percent_error
import numpy as np


def main():

    # TODO: run and save data from finite diff experiment
    FD_DIR = "../data/fdm-diffusion/08242023/u.npy"

    SD_DIR = "../data/spectral-diffusion/08192023/u.npy"

    # SAVE_DIR = "../figures/08282023/"

    # percent_error(FD_DIR, SD_DIR, SAVE_DIR)

    u_fd = np.load(FD_DIR)
    u_sd = np.load(SD_DIR)

    print(u_fd.shape)
    print(u_sd.shape)


if __name__ == "__main__":
    main()
