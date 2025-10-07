# Dataset folder structure

Please download the original datasets, `ARAD_1K`, `CAVE`, `ICVL`, `KAUST`, to this folder. The structure of this folder should look like the following.

```
📦datasets
 ┣ 📂ARAD_1K
 ┣ 📂CAVE
 ┣ 📂ICVL
 ┣ 📂KAUST
 ┗ 📜README.md
```

## 1. ARAD_1K

The information about downloading the dataset can be found at https://github.com/boazarad/ARAD_1K.

We generate metameric data from the original dataset by running the meramer generation script `metamer/generate_metamer.py`. See `TODO: XXX` for detailed instructions.

Once metameric data are generated, the full data folder should look like the following.

```
📦ARAD_1K
 ┗ 📂split_txt
 ┃ ┣ 📜train_list.txt
 ┃ ┗ 📜valid_list.txt
 ┣ 📂poisson0_double_gauss_50mm_F1.8
 ┣ 📂poisson0_None
 ┣ 📂poisson1000_double_gauss_50mm_F1.8
 ┣ 📂poisson1000_None
 ┣ 📂Train_RGB
 ┣ 📂Train_RGB
 ┗ 📂Train_Spec
```

## 2. CAVE

The original dataset can be downloaded at https://cave.cs.columbia.edu/repository/Multispectral.

First, we convert the original CAVE data (`.png` files) to Matlab `.mat` files by running a decicaded script `dataset_tools/prepare_CAVE.py`. This step makes the data format compatible for processing in a unified manner.

Second, we generate metameric data from the prepared `.mat` files by running the meramer generation script `metamer/generate_metamer.py` with specific parameters. See `TODO: XXX` for detailed instructions and configurations we use in the paper.

Once metameric data are generated, the full data folder should look like the following.

```
📦CAVE
 ┗ 📂split_txt
 ┃ ┣ 📜train_list.txt
 ┃ ┗ 📜valid_list.txt
 ┣ 📂poisson0_double_gauss_50mm_F1.8
 ┣ 📂poisson0_None
 ┣ 📂poisson1000_double_gauss_50mm_F1.8
 ┣ 📂poisson1000_None
 ┣ 📂Train_RGB
 ┣ 📂Train_RGB
 ┗ 📂Train_Spec
```

## 3. ICVL

The original dataset can be downloaded at https://icvl.cs.bgu.ac.il/hyperspectral/ or https://huggingface.co/datasets/danaroth/icvl.

First, we convert the original ICVL data (`.mat` files) in full spectral range to data (`.mat` files) in the target wavelength range set in the `config/icvl.py` by running a decicaded script `dataset_tools/prepare_ICVL.py`. This step makes the data format compatible for processing in a unified manner.

Second, we generate metameric data from the prepared `.mat` files by running the meramer generation script `metamer/generate_metamer.py` with specific parameters. See `TODO: XXX` for detailed instructions and configurations we use in the paper.

Once metameric data are generated, the full data folder should look like the following.

```
ICVL
 ┗ 📂split_txt
 ┃ ┣ 📜train_list.txt
 ┃ ┗ 📜valid_list.txt
 ┣ 📂poisson0_double_gauss_50mm_F1.8
 ┣ 📂poisson0_None
 ┣ 📂poisson1000_double_gauss_50mm_F1.8
 ┣ 📂poisson1000_None
 ┣ 📂Train_RGB
 ┣ 📂Train_RGB
 ┗ 📂Train_Spec
```

## 4. KAUST

The original dataset can be downloaded at https://repository.kaust.edu.sa/handle/10754/670368. `TODO: upload raw mat files to the repository`

First, we convert the original KAUST data (`.mat` files) in full spectral range to data (`.mat` files) in the target wavelength range set in the `config/kaust.py` by running a decicaded script `dataset_tools/prepare_KAUST.py`. This step makes the data format compatible for processing in a unified manner.

Second, we generate metameric data from the prepared `.mat` files by running the meramer generation script `metamer/generate_metamer.py` with specific parameters. See `TODO: XXX` for detailed instructions and configurations we use in the paper.

Once metameric data are generated, the full data folder should look like the following.

```
KAUST
 ┗ 📂split_txt
 ┃ ┣ 📜train_list.txt
 ┃ ┗ 📜valid_list.txt
 ┣ 📂poisson0_double_gauss_50mm_F1.8
 ┣ 📂poisson0_None
 ┣ 📂poisson1000_double_gauss_50mm_F1.8
 ┣ 📂poisson1000_None
 ┣ 📂Train_RGB
 ┣ 📂Train_RGB
 ┗ 📂Train_Spec
```