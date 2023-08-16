![](assets/banner.png)

This repository contains the modeling code for [AO3 Disco](https://ao3-disco.app/), a fanfiction
recommendation engine which is available for web, iOS, and Android.

## Usage
After installing this library (`pip install .`), you can load a pre-trained model simply by
unpickling it and passing a list of works to the `embed` method. Note that we use Pydantic for
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

## Design
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

## Experimentation
When experimenting with a new idea, here is the recommended process:

### Local
First, experiment locally using dev mode which uses 0.001% of the dataset.

1. Create a new branch from `master` with a descriptive name.
2. Run the baseline model to make sure it works.
3. Make the appropriate changes while making sure it's backwards compatible.
4. Re-run the baseline model (with your changes hidden behind feature flags) and 
   verify the baseline performance has not changed.
5. Run your experiments and compare against the baselines.

The actual performance on the dev dataset is not important. What's important is 
making sure that the baseline is stable.

### Remote
After verifying everything works as expected locally, it's time to test it on 
the full dataset. The best way to do this is to run the deploy script which
will do the following:

1. Build a Docker container and push it to gcloud's Artifact Registry.
2. Upload the local version of your datasets to gcloud Cloud Storage.

Then, you can run the `gcloud batch jobs` command using the JSON templates (i.e. 
`gcloud/batch.json`) which will launch the containers with the appropriate volume 
mounts for the data + logs.

After you've launched your experiments, you can download the logs from Cloud 
Storage and inspect them locally with Tensorboard.
