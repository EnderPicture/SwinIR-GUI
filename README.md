# SwinIR GUI


## Get started
### Install dependencies

* Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links)
* Install conda environment with command 
`conda env create -f environment.yml`
* Activate conda environment with command `conda activate pytorch1`
* Install [CUDA Toolkit 11 (windows)](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

### Run
- Run the gui with command `python gui.py`
- GUI
  - select files
    - select multiple files
    - supported format list is not complete, please let me know if I should add any extensions
  - image preview window
    - only shows the first of the selected files
    - click anywhere on the image to show expanded view and wait for ai enhanced results
  - dropdown #1
    - pre trained model selector
  - dropdown #2
    - tile size selector
    - tile size work with `window_size` and `overlap` to create final tile size
      - `window_size` is always `8` except for de-jpeg model, which is `7`
      - for example 50(tile size) * 8(window_size) + 32(overlap) = 432px * 432px tiles.
  - batch run files through selected model, results in `/output` folder


## What is this
A GUI built on top of [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) that 
* provides auto downloading of pre-trained models upon first use
* interactive preview for selected model
* batch image processor
* tile size selector to maximize vram usage without going over

Still early in development. Can be unstable and no guarantee is provided.