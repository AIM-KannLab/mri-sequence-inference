# mri-sequence-classification
This repository contains the basic inference scripts to get our longitudinal mri images classified into one of the four modalities: T1, T1c, T2w, and FLAIR or a class called OTHER, which should entail all other MRI modalities. There are three main scripts and 2 configuration files corresponding to the inference and the later postprocessing.

## How to run
First, create a conda environment, activate it, and install the dependencies.

```
conda create --name mri-seq-classf

conda activate mri-seq-classf

pip install -r requirements.txt
```

When you clone the repository, you should also see a the empty folder _data_csv/long/_. Here is where all of the outputs are going to be saved, namely in subfolders containing the name of the dataset the inference was run on. The scripts should be run in the following order:

### 1. Inference
- Take a look at the configuration file ``config_inference.py``
- Adjust all of the paths to match the ones in your project:
  - Adjust  ``DATA_DIR`` to where you have the datasets you want to infer. As you can see, for the four datatsets are relative paths of this main path.
  - Similarly, adjust ``BASE_OUTPUT`` to where you want your output saved to. As recommended above, this path should end in the folder structure _data_csv/long/_ which located in this repo. Again, the four output paths for the datasets are relative to this base directory.
  - Finally, adjust the ``MODELS_DIR`` to the path where you downloaded the models from the _Deep Learning-based Type Identification of Volumetric MRI Sequences_ paper (link).
- Set the GET_CSV parameter to ``TRUE`` to get firstly a list of csv files of the dataset.
- Change line 244  in the ``inference.py`` script to the paths that you want to firstly get your data from / on and run the script. You will see a list of .csv files in a dataset subfolder inside of _data_csv/long/_.
-  Set the GET_CSV parameter to ``FALSE``. Run inference again, but this time do it by executing the ``-t`` flag, e.g. ``python3 inference.py -t OUTPUT_DIR_XYZ``, where ``OUTPUT_DIR_XYZ`` is the directory where you have your .csv´s stored.

### 2. Parser
- TODO
