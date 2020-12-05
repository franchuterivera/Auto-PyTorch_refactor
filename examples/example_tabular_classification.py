
"""
======================
Tabular Classification
======================
"""
import os

import numpy as np

import sklearn.datasets
import sklearn.model_selection
from sklearn.utils.multiclass import type_of_target

from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.utils.backend import create


# Get the training data for tabular classification
# Move to Australian to showcase numerical vs categorical
X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state=1,
    test_size=0.2,
)

train_indices, val_indices = sklearn.model_selection.train_test_split(
    list(range(X_train.shape[0])),
    random_state=1,
    test_size=0.25,
)

output_type = type_of_target(y)
print(f"X_train={X_train.shape} train_indices={train_indices} output_type={output_type}")

# Mock the categories
categorical_columns = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
numerical_columns = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
categories = [np.unique(X[a]) for a in categorical_columns]

# Create a proof of concept pipeline!
dataset_properties = {
    'task_type': 'tabular_classification',
    'categorical_columns': categorical_columns,
    'numerical_columns': numerical_columns,
    'output_type': output_type,
}


# Save data via backend to fit the pipeline
datamanager = TabularDataset(
    X=X_train, Y=y_train,
    X_test=X_test, Y_test=y_test,
)

backend = create(temporary_directory='./tmp/autoPyTorch_tabular_classification_tmp',
                 output_directory='./tmp/autoPyTorch_tabular_classification_out',
                 delete_tmp_folder_after_terminate=False)
backend.save_datamanager(datamanager)

pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)

# Create a fit dictionary
fit_dictionary = {
    'categorical_columns': categorical_columns,
    'numerical_columns': numerical_columns,
    'num_features': X.shape[1],
    'num_classes': len(np.unique(y)),
    'is_small_preprocess': True,
    'categories': categories,
    'X_train': X_train,
    'y_train': y_train,
    'train_indices': train_indices,
    'val_indices': val_indices,
    'X_test': X_test,
    'y_test': y_test,
    'dataset_properties': dataset_properties,
    # Training configuration
    'job_id': 'example_tabular_classification_1',
    'working_dir': './tmp/example_tabular_classification_1',  # Hopefully generated by backend
    'device': 'cpu',
    'budget_type': 'epochs',
    'epochs': 100,
    'runtime': 3600,
    'torch_num_threads': 1,
    'early_stopping': 20,
    'use_tensorboard_logger': True,
    'use_pynisher': False,
    'metrics_during_training': True,
    'backend': backend,
    'split_id': 0,
}

# Configuration space
pipeline_cs = pipeline.get_hyperparameter_search_space()
print("Pipeline CS:\n", '_' * 40, f"\n{pipeline_cs}")
config = pipeline_cs.sample_configuration()
print("Pipeline Random Config:\n", '_' * 40, f"\n{config}")
pipeline.set_hyperparameters(config)

# Make sure the working directory exists. Something that backend will handle
os.makedirs('./tmp/example_tabular_classification_1', exist_ok=True)

# Fit the pipeline
print("Fitting the pipeline...")
pipeline.fit(fit_dictionary)

# Showcase some components of the pipeline
print(pipeline)

# Showcase performance of pipeline
print(pipeline.named_steps['trainer'].run_summary.repr_last_epoch())
