<h1 align="center">MRI Sequence Classification </h1>
<div align="center">
<a href="https://github.com/AIM-KannLab/mri-sequence-classification" target="_blank"><img src="https://img.shields.io/github/stars/AIM-KannLab/mri-sequence-classification?style=social" alt="Stars Badge"/></a>
<a href="https://github.com/AIM-KannLab/mri-sequence-classification/fork" target="_blank"><img src="https://img.shields.io/github/forks/AIM-KannLab/mri-sequence-classification?style=social" alt="Forks Badge"/></a>
<a href="https://github.com/AIM-KannLab/mri-sequence-classification/stargazers" target="_blank"><img src="https://img.shields.io/github/watchers/AIM-KannLab/mri-sequence-classification?style=social" alt="Watchers Badge"/></a>
<a href="https://github.com/AIM-KannLab/mri-sequence-classification" target="_blank"><img src="https://img.shields.io/github/repo-size/AIM-KannLab/mri-sequence-classification" alt="Repository size Badge"/></a>
<a href="https://github.com/AIM-KannLab/mri-sequence-classification/commits/" target="_blank"><img src="https://img.shields.io/github/last-commit/AIM-KannLab/mri-sequence-classification" alt="Last commit Badge"/></a>
<a href="https://github.com/AIM-KannLab/mri-sequence-classification/issues" target="_blank"><img src="https://img.shields.io/github/issues/AIM-KannLab/mri-sequence-classification" alt="Open issues Badge"/></a>
<a href="https://github.com/AIM-KannLab/mri-sequence-classification/pulls" target="_blank"><img src="https://img.shields.io/github/issues-pr/AIM-KannLab/mri-sequence-classification" alt="Open pull requests Badge"/></a>
<br>
<a href="https://www.linkedin.com/in/juan-carlos-climent-pardo/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&color=blue" alt="LinkedIn Badge"/></a>
<a href="https://github.com/AIM-KannLab" target="_blank"><img src="https://img.shields.io/badge/GitHub-black?style=flat&logo=github&color=black" alt="GitHub Badge"/></a>
</div>

This repository contains the basic inference scripts to get longitudinal MRI images classified into one of the four modalities: T1, T1c, T2W, and FLAIR or a class called OTHER, which should entail all other MRI modalities. There are three main scripts and 2 configuration files corresponding to the inference and the later postprocessing.

## Disclaimer
As this script is adjusted from the code provided in the paper [_Deep Learning-based Type Identification of Volumetric MRI Sequences_](https://arxiv.org/pdf/2106.03208.pdf) there is further room for improvement on the quality of predictions such as training a model with other data, or fine-tuning the model to the BCH pediatric data. The accuracy proposed on the paper is about ~96%. The accuracy on the pediatric BCH dataset is a little bit less, with error rates of 6.5% error for T2W, 3% error for FLAIR and T1/T1c together around 4%. So it is generally good to automatize a pipeline which separates T1/T1c's from T2W/ FLAIR images.

## How to run
First, create a conda environment, activate it, and install the dependencies.

```
conda create --name mri-seq-classf

conda activate mri-seq-classf

pip install -r requirements.txt
```

When you clone the repository, you should also see the empty folder _data_csv/long/_. Here is where all of the outputs are going to be saved, namely in subfolders containing the names of the datasets the inference was run on. The scripts should be run in the following order:

### 1. Inference
- Take a look at the configuration file ``inference_cfg.py``
- Adjust all of the paths to match the ones in your project:
  - Adjust  ``DATA_DIR`` to where you have the datasets you want to infer. As you can see, for the datatsets are relative paths of the main path.
  - Similarly, adjust ``BASE_OUTPUT`` to where you want your output saved to. As recommended above, this path should end in the folder structure _data_csv/long/_ which located in this repo. Again, the output paths for the datasets are relative to this base directory.
  - Finally, adjust the ``MODELS_DIR`` to the path where you downloaded the models from the [_Deep Learning-based Type Identification of Volumetric MRI Sequences_](https://arxiv.org/pdf/2106.03208.pdf) paper. Download [link](https://drive.google.com/drive/folders/1h6fgWXEUxQaFFM72XvaUMLw0ExR-6dFU).
- Set the GET_CSV parameter to ``TRUE`` to get firstly a list of csv files of the dataset where the inference is going to happen.
- Change line 244  in the ``inference.py`` script to the paths that you want to firstly get your data from and run the script. You will see a list of .csv files in a dataset subfolder inside of _data_csv/long/_.
-  Set the GET_CSV parameter to ``FALSE``. Run inference again, but this time do it by executing the ``-t`` flag, e.g. ``python3 inference.py -t OUTPUT_DIR_XYZ``, where ``OUTPUT_DIR_XYZ`` is the directory where you have your .csv´s stored. Otherwise a default might get used that you do not want.
- After the script has run you will see the .csv´s files which were empty before now filled with the inference data corresponding to the image characteristics and the prediction of the algorithm.

### 2. Parser
Now that you have a subfolder with your dataset and .csv´s containing the predictions, it is time to run some post-processing and filter out badly classified, conflicting, and erroneous data. For that we are going to use the ``parser.py`` file, which is also accompanied by a ``parser_cfg.py`` file. In this configuration file, you should make sure again that you have the path ``BASE_DIR`` well aligned with your project directory.  
- After adjusting the ``parser_cfg.py``, you can run the parser script. Take into account the list of main directories defined at the beginning of the file and (un)comment the name of the directory / dataset you want to run. This may take a while and consume a lot of memory, so make sure you have enough space in the output directory you defined.

### 3. Evaluation
Finally, you have an output directory with 6 subfolders corresponding to all of the files of your data classified into one of the four possible sequences, the OTHER class and the erronous files in a NO PREDICTION folder. You can start reviewing the data with the help of [3D Slicer](https://www.slicer.org/) and the custom plugin of [Segmentation Review](https://github.com/zapaishchykova/SegmentationReview). This plugin allows you on the dev branch to execute it in a review mode, where, once you load the datafolder to the 3D Slicer software, it creates a ``annotations.csv`` file with the ratings and comments you provide. For more detailed information please consult the software and plugin websites.
