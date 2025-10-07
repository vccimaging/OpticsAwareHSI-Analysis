The code in this folder is adapted from the MST_plus_plus repository.

Original source: https://github.com/caiyuanhao1998/MST-plus-plus/blob/master/train_code/

Accessed on: 2025-04-23

We have made necessary changes in the code to 
- adapt to all the 17 architectures used in the paper;
- enable a percentage of training data to be used for training;
- (optionally) log results in [WandB](https://wandb.ai/)

By default, WandB logging is disabled. To enable it:

```bash
pip install wandb
wandb login
```

If you’re not logged in or don’t have a WandB account, the script will automatically skip logging without errors.

The initial learning rate depends on the method. We adopt the [settings from MST++](https://github.com/caiyuanhao1998/MST-plus-plus?tab=readme-ov-file#5-training), as shown below.

| **Method** | **Initial learning rate** |
|------------|---------------------------|
|    MST++   |            4e-4           |
|     MST    |            4e-4           |
|   MPRNet   |            2e-4           |
|  Restormer |            2e-4           |
|   MIRNet   |            4e-4           |
|    HINet   |            2e-4           |
|    HDNet   |            4e-4           |
|    AWAN    |            1e-4           |
|    EDSR    |            1e-4           |
|    HRNet   |            1e-4           |
|   HSCNN+   |            2e-4           |
|    HySAT   |            4e-4           |
|    HPRN    |           1.2e-4          |
|  SSTHyper  |            4e-4           |
|    MSFN    |            4e-4           |
|    GMSR    |            1e-4           |
|   SSRNet   |            2e-4           |

Finally, this folder should show the following structure

```
📦MST_plus_plus_code
 ┣ 📂train_code
 ┃ ┣ 📂architecture
 ┃ ┃ ┣ 📜AWAN.py
 ┃ ┃ ┣ 📜GMSR.py
 ┃ ┃ ┣ 📜HDNet.py
 ┃ ┃ ┣ 📜HPRN.py
 ┃ ┃ ┣ 📜HSCNN_Plus.py
 ┃ ┃ ┣ 📜MIRNet.py
 ┃ ┃ ┣ 📜MPRNet.py
 ┃ ┃ ┣ 📜MSFN.py
 ┃ ┃ ┣ 📜MST.py
 ┃ ┃ ┣ 📜MST_Plus_Plus.py
 ┃ ┃ ┣ 📜Restormer.py
 ┃ ┃ ┣ 📜SSRnet.py
 ┃ ┃ ┣ 📜SSTHyper.py
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜edsr.py
 ┃ ┃ ┣ 📜hinet.py
 ┃ ┃ ┣ 📜hrnet.py
 ┃ ┃ ┗ 📜hysat.py
 ┃ ┣ 📜hsi_dataset.py
 ┃ ┣ 📜train.py
 ┃ ┗ 📜utils.py
 ┗ 📜README.md
```