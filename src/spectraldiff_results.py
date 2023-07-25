from utils.diffusion_utils import plot_T, plot_rxn_diffusion


def main():

    plot_T(
        load_dir="../data/spectral-diffusion/08202023/T.npy",
        save_dir="../figures/08202023/",
        orientation="vertical",
    )

    plot_rxn_diffusion(
        load_dir="../data/spectral-diffusion/08202023/u.npy",
        save_dir="../figures/08202023/",
        orientation="horizontal",
    )


if __name__ == "__main__":
    main()
