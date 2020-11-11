"""
======================
Image Classification
======================
"""
import numpy as np

import torchvision.datasets

from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline


# Get the training data for tabular classification
trainset = torchvision.datasets.FashionMNIST(root="../datasets/", train=True, download=True)
data = trainset.data.numpy()
data = np.expand_dims(data, axis=3)
# Create a proof of concept pipeline!
dataset_properties = dict()
pipeline = ImageClassificationPipeline(dataset_properties=dataset_properties)

# Configuration space
pipeline_cs = pipeline.get_hyperparameter_search_space()
print("Pipeline CS:\n", "_" * 40, f"\n{pipeline_cs}")
config = pipeline_cs.sample_configuration()
print("Pipeline Random Config:\n", "_" * 40, f"\n{config}")
pipeline.set_hyperparameters(config)

# Fit the pipeline
print("Fitting the pipeline...")

pipeline.fit(
    X=dict(
        X_train=data,
        is_small_preprocess=True,
        channelwise_mean=np.array([np.mean(data[:, :, :, i]) for i in range(1)]),
        channelwise_std=np.array([np.std(data[:, :, :, i]) for i in range(1)]),
        num_classes=10,
        num_features=data.shape[1] * data.shape[2],
    )
)

# Showcase some components of the pipeline
print(pipeline)
