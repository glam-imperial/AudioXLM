# AudioXLM
Repository for Improving Audio Explanations using Audio Language Models

## Contents:

**[audio_explanation_generation](codes/audio_explanation_generation.py)** --> Inference script for generating audio explanations using AudioXLM.\
**[fidelity_performance](codes/fidelity_performance.py)** --> Script to measure the fidelity score of explanations on the Speech Commands and TESS datasets.\
**[ASR_WER_performance_TESS](codes/ASR_WER_performance_TESS.py)** --> Script for evaluating automatic speech recognition performance of AudioXLM explanations in the speech emotion recognition task.\
**[encode_dataset_AudioGen_SC](codes/encode_dataset_AudioGen_SC.py)** --> Script for encoding datasets into the embedding space of AudioGen for classifier models and AudioXLM, ensuring representation consistency.\
**[AudioGen_update](codes/AudioGen_update/)** --> Scripts for modifying the original AudioGen library. After installing AudioGen, copy these scripts into the appropriate [path](https://github.com/facebookresearch/audiocraft/tree/main/audiocraft/models) in AudioGen library.\
**[models](models/)** --> Folder containing classification models that predict on encoded datasets.\
**[sample_explanations](sample_explanations/)** --> Sample audio explanations generated by AudioXLM.

## AudioGen Installation
Follow the [AudioGen installation instructions](https://github.com/facebookresearch/audiocraft/blob/main/README.md) from the AudioCraft repository.

AudioCraft requires Python 3.9, PyTorch 2.1.0. To install AudioCraft, you can run the following:

```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
python -m pip install 'torch==2.1.0'
# You might need the following before trying to install the packages
python -m pip install setuptools wheel
# Then proceed to one of the following
python -m pip install -U audiocraft  # stable release
python -m pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft  # bleeding edge
python -m pip install -e .  # or if you cloned the repo locally (mandatory if you want to train).
python -m pip install -e '.[wm]'  # if you want to train a watermarking model
```

We also recommend having `ffmpeg` installed, either through your system or Anaconda:
```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

## Citation

The citation will be provided upon publication.

