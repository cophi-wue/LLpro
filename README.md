# Event Classification
Classify events into the four categories: "non-event", "change-of-state", "process" and "stative-event".
The spans of events are inferred using a rule-based system on top of a dependency parser.

Three entry points to perform different tasks exist:
* `main.py`, perform training on gold-span data from the JSON dataset
    * e.g: `python main.py batch_size=16`
* `predict.py`, perform classification inference on a pre-processed or plain-text file
    * e.g.: `python predict.py plain-text-file <model_path> <input_txt_file> <output_json_file>`

## Setup
The project uses poetry for dependency management. You can just run: 
```
poetry install 
```

If you want to make use of a GPU, you will need additonal depdencies, just run `poetry install -E gpu` instead.

Open a shell with the python packages and interpreter:
```
poetry shell
```

If cupy is taking what feels like forever to install/compile this is expected behaviour, you may chose to install a prebuilt binary instead. Check their [releases](https://github.com/cupy/cupy/releases) and pick a wheel that matches your cpython and cuda versions as well as your OS and architecture: run `poetry add <whl_url>`.

## Usage

### Inference
You can easily process a single text file:
```
python predict.py plain-text-file <model_path> <input_txt_file> <output_json_file>
```

If your system does not have a cuda device pass `--device=cpu` as the script currently does not properly recognize this by itself.

The JSON data will contain information besides the event types, these predictions are however not of good quality and should not be used for any purposes.

You may optionally use `preprocess.py` to cache the segmentation into verb phrases, e.g. `python preprocess.py text_1.txt text_2.txt all_texts.json`

### Training
If you want to train a model you will first need to download the annotation data by runnning `./download.sh`.
Alternativley, if you have access, you can retrieve the content of the CATMA submodule: `git submodule update --init --recursive`

The training script `main.py` can be configured via `conf/config.yaml`,
individual parameters can be overridden using command line parameters like this: `python main.py label_smoothing=false`.

Model weights and logs are saved to `outputs/<date>/<time>`, mlflow logs are created in `runs`.
Start the mlflow ui like this: `mlflow ui`, if the binary is not in your path and you set up a virtual environment you may need to run `venv/bin/mlflow ui`.


### Setting up torchserve

If you want to perform inference via an HTTP API this can be done using torchserve. This provides the API that consumed by [the narrativity graph frontend](https://github.com/uhh-lt/narrativity-frontend).

This guide assumes you have a fully trained model, you may download one from the releases tab on Github.

- Make sure you have `torchserve` and the `torch-model-archiver` installed: `poetry install -E torchserve`
- Create a model archive using an existing model in model_directory  `./archive-model.sh <model_archive_name> <model_directory>`. This will create a `.mar` file in the current directory that is named according to specified model archive name.
- You can now serve the model using `torchserve --foreground --model-store $(pwd) --models model_name=model_name.mar --ncs`

Common pitfalls:
- Wherever you serve the model from needs the dependencies from requirements.txt installed
- torchserve makes use of java so if you are getting errors this could also be related to your java version