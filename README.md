![](assets/banner.png)

This repository contains the modeling code for [AO3 Disco](https://ao3-disco.app/), a fanfiction
recommendation engine which is available for web, iOS, and Android.

## Usage
After installing this library (`pip install .`), to use a pre-trained model, you can simply 
unpickle it and pass a list of works to the `embed` method. Note that we use Pydantic for
data validation so the JSON fetched from the database can be parsed as shown below.

```python
import pickle
from ao3_disco_ai import Work

with open("wrapped_model.pkl", "rb") as fin:
    model = pickle.load(fin)

works = [
    Work(**{...workJSON...})
]
embeddings = model.embed(works)
```

## Development
Due to some unique quirks of this dataset, we have a custom preprocessing and 
model export pipeline. As a minimal example of getting a model up and running, 
you can do the following:

```bash
ao3-disco-ai preprocess --dev
ao3-disco-ai train --dev
ao3-disco-ai export lightning_logs/version_0
```

Note that both preprocess and train are in dev mode, which means they are trained
on minimal data which is very fast and is useful for testing the pipeline. The 
output of the export step can be directly plugged into the API (although the results
will be quite bad due to the limited data).

### Preprocessing
In this stage, we build the feature extraction pipeline
and split the dataset into train and test sets. The outputs are versioned 
and logged in the `data` directory.

> ao3-disco-ai preprocess --help

### Training
In this stage, we train the models and log the results in the
`lightning_logs` directory. This stage allows us to experiment with different
losses and architectures.

> ao3-disco-ai train --help

### Export
In this stage, given the best model, we extract only the relevant parameters and 
bundle them together with the corresponding feature pipeline.

> ao3-disco-ai export
