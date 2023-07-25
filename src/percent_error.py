from utils.diffusion_utils import percent_error


def main():

    # TODO: run and save data from finite diff experiment
    FD_DIR = "../data/fdm-diffusion/08242023/u-test.npy"

    SD_DIR = "../data/spectral-diffusion/08192023/u.npy"

    SAVE_DIR = "../figures/08242023/"

    percent_error(FD_DIR, SD_DIR, SAVE_DIR)


if __name__ == "__main__":
    main()
