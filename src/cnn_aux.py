import numpy as np
import pandas as pd
from collections import namedtuple
import os
import glob


def files_list_npy_csv(data_path):
    """
    Returns:
     - imx:  the prefix of the .npy files storing images 
     - mdx:  the prefix of the .csv file storing metadata
     - imn:  the index (e.g order of files)
    
    """
    def select_files(ext="*.npy"):
        npys = os.path.join(data_path, ext)
        return glob.glob(npys)

    images = [f1.split("/")[-1] for f1 in select_files(ext="*.npy")]
    metadata = [f1.split("/")[-1] for f1 in select_files(ext="*.csv")]
    
    imn = [int(im.split(".")[0].split("_")[1]) for im in images]
    mdn = [int(im.split(".")[0].split("_")[1]) for im in metadata]
    assert (np.sort(mdn) == np.sort(imn)).all()

    imx=images[0].split(".")[0].split("_")[0]
    mdx = metadata[0].split(".")[0].split("_")[0]
    return imx, mdx, np.sort(imn)


def select_image_and_lbl(data_path, file_id):
    """
    Returns, for file_id (e.g, the number that identifies both images and metadata):
     - imgs: a .npy file containing images 
     - mdata: a DataFrame with metadata
    
    """
    img_name, lbl_name, indx = files_list_npy_csv(data_path)
    
    if file_id < 0 or file_id > len(indx) -1:
        assert False
    else:
        img_fname = f"{img_name}_{indx[file_id]}.npy"
        lbl_fname = f"{lbl_name}_{indx[file_id]}.csv"
        print(f"Selected files: img = {img_fname}, metdata = {lbl_fname}")
        imgs  = np.load(os.path.join(data_path,img_fname))
        mdata = pd.read_csv(os.path.join(data_path, lbl_fname))
        return imgs, mdata


def get_energy(imgs):
    """
    Compute the energy of the images (imgs) by adding the contents (number of photons)
    in each pixel
    
    """
    energies = [imgs[i].sum() for i in range(0,imgs.shape[0])]
    return np.array(energies)


def mean_rms(energies, fwhm_only=False):
    """
    Compute the mean, std and std/mean (FWHM) of the energy vector stored in ```energies```

    """
    if fwhm_only:
        return 2.3*np.std(energies)/np.mean(energies)
    else:
        return np.mean(energies), np.std(energies)/np.mean(energies), 2.3*np.std(energies)/np.mean(energies)





