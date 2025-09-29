# Project Overview

This notebook documents the end-to-end workflow for detecting **African Easterly Waves (AEWs)** using deep learning on reanalysis data. AEWs are synoptic-scale disturbances that influence rainfall variability and tropical cyclone formation over West Africa and the Atlantic. Identifying them accurately is important for improving weather prediction and climate research.

The workflow integrates **ERA5 atmospheric reanalysis variables** (e.g., temperature, wind, humidity, potential vorticity) at surface and pressure levels. Data are preprocessed into spatial patches around AEW events and matched with background (non-AEW) samples to form a labeled dataset.

The modeling pipeline consists of:

- **Data ingestion and preprocessing**  
  Reading ERA5 NetCDF files, converting them to Zarr format for efficiency, concatenating multiple variables, and applying normalization.  

- **Labeling and feature preparation**  
  Associating each sample with AEW event labels, handling missing values, and splitting into training and testing datasets.  

- **Modeling and training**  
  Building a **Convolutional Neural Network (CNN)** with hyperparameter tuning via Bayesian optimization. The model leverages focal loss and F1-based metrics to address strong class imbalance.  

- **Evaluation and interpretation**  
  Performance is assessed with confusion matrices, precision-recall metrics, and saliency maps that highlight spatial feature importance for AEW detection.  

The notebook is designed for **reproducibility and scalability**, with configurable parameters for variables, pressure levels, and subsets. It serves as both a research record and a template for future experiments.



## Cell [1]: Parameter Setup

```python
var_list = [
    "cape", "crr", "d", "ie", "ishf", "lsrr", "pv", "q", "r", "sp",
    "tcw", "tcwv", "t", "ttr", "u", "v", "vo", "w",
]

plevel_list = [
    False, False, 300, False, False, False, 300, 300, 300, False,
    False, False, 300, False, 300, 300, 300, 300,
]

aew_subset = "12hr_before"
model_save_name = "best_model_3001.keras"
tuner_project_name = "tuner_run_3001"
```

### Explanation
This cell sets the configuration parameters for the workflow.  
- **var_list**: List of ERA5 atmospheric variables used as features.  
- **plevel_list**: Pressure levels assigned to variables where needed, otherwise set to False.  
- **aew_subset**: Time window relative to the AEW event (here 12 hours before).  
- **model_save_name**: Name used when saving the trained model.  
- **tuner_project_name**: Label for organizing hyperparameter tuning experiments.

## Cell [2]: Library Imports and Environment Setup

```python
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sklearn
import sklearn.model_selection
import keras
from keras import layers
import keras_tuner
import tensorflow as tf
import tensorflow.keras.backend as K

keras.utils.set_random_seed(812)
```
## Cell [3]: Focal Loss Function

```python
import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss for binary classification."""
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        return alpha_factor * modulating_factor * bce
    return loss_fn
```

### Explanation

This cell defines **focal loss**, a modified version of binary cross-entropy that is better suited for highly imbalanced classification problems.  

In standard binary classification, cross-entropy treats all examples equally. Focal loss adjusts this by down-weighting well-classified examples and focusing more on hard, misclassified ones. This helps the model learn better from the minority class, which in this case is AEW events.  

It is commonly used in object detection tasks and any scenario where one class significantly outnumbers the other. This makes it a good choice for AEW detection, where positive samples are rare compared to background data.

## Cell [4]: F1 Loss with Sigmoid Output

```python
def f1_loss_sigmoid(y_true, y_pred):
    """
    F1 metric for sigmoid output and integer encoded labels.
    """
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = (2 * p * r) / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - K.mean(f1)
```
### Explanation

This cell defines a custom **F1 loss** function for binary classification problems using a sigmoid output layer.  

F1 score combines precision and recall, making it a more balanced metric for imbalanced datasets. Since F1 is not directly differentiable, this function approximates it in a way that can be used as a loss function during training.  

Minimizing this loss encourages the model to improve both precision and recall, rather than optimizing accuracy alone. This is useful when false negatives and false positives carry different costs, as in AEW detection.
## Cell [5]: F1 Loss with One-Hot Encoded Labels

```python
def f1_loss_onehot(y_true, y_pred):
    """
    F1 metric for two-class output and one-hot encoded labels.
    """
    tp = K.sum(K.cast(y_true[:, 1] * y_pred[:, 1], 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true[:, 1]) * y_pred[:, 1], 'float'), axis=0)
    fn = K.sum(K.cast(y_true[:, 0] * (1 - y_pred[:, 0]), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = (2 * p * r) / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - K.mean(f1)
```
### Explanation

This cell defines an alternative **F1 loss** function that works with one-hot encoded labels instead of integer labels.  

The logic is similar to the previous sigmoid-based F1 loss, but it accesses the relevant class indices using slicing. This function is used when the model's output and labels are in one-hot format (e.g., `[1, 0]` or `[0, 1]`).  

It serves the same purpose — to improve both precision and recall during training — especially useful when working with class-imbalanced data like AEW detection.
## Cell [6]: `add_dim()` Preprocessing Function

```python
import re

def add_dim(ds):
    # Extract the source file name from the dataset's encoding.
    fname = ds.encoding.get('source', '')
    
    # Use a regex to capture the central latitude and longitude from the filename.
    m = re.search(r'_(\-?\d+\.\d+)_(-?\d+\.\d+)\.nc$', fname)
    if m:
        lat_center = float(m.group(1))
        lon_center = float(m.group(2))
        # Assign the central coordinates and the file name as new coordinates.
        ds = ds.assign_coords(lat_center=lat_center, lon_center=lon_center, file_name=fname)
    else:
        print("File name does not match expected pattern:", fname)

    # Expand dims to add the 'sample' dimension and drop unnecessary variables.
    return ds.assign_coords({"sample": 1}).expand_dims(dim={"sample": 1}).drop_vars("utc_date").drop_vars("latitude").drop_vars("longitude")
```
### Explanation

This function prepares a single NetCDF file for model input by applying a few preprocessing steps.  

It extracts the latitude and longitude from the filename using regular expressions and adds them as new coordinates.  

It adds a `sample` dimension to make the data consistent for concatenation later.  

It removes unneeded coordinate variables like `utc_date`, `latitude`, and `longitude`, since those are already encoded in the filename.  

This function is used as a preprocessing hook when reading multiple files together, ensuring each file is treated as a single sample with standardized structure.

## Cell [7]: open_files_zarr() – Load and Convert ERA5 Data

```python
import os
import xarray as xr

def open_files_zarr(list_of_vars, aew_subset="12hr_before",
                    directory="/glade/derecho/scratch/rmandava/AEW_time_location_files/",
                    plevel_list=None, zarr_store_path="zarr_data"):
    """
    Opens ERA5 NetCDF files for the given variables. For each variable (or pressure-level variant),
    it checks if a corresponding Zarr store exists in 'zarr_store_path'. If so, it loads the dataset
    from the Zarr store; if not, it opens the NetCDF files, preprocesses them, saves them to Zarr,
    and then returns the dataset.
    """
    if not os.path.exists(zarr_store_path):
        os.makedirs(zarr_store_path)

    datas = {}
    for num, var in enumerate(list_of_vars):
        if plevel_list:
            if plevel_list[num]:
                key = f"{var}_{int(plevel_list[num])}"
                file_pattern = f'{directory}/{var}/aew_{aew_subset}_{int(plevel_list[num])}_*.nc'
            else:
                key = var
                file_pattern = f'{directory}/{var}/aew_{aew_subset}_*.nc'
        else:
            key = var
            file_pattern = f'{directory}/{var}/aew_{aew_subset}_*.nc'

        zarr_path = os.path.join(zarr_store_path, f"{key}.zarr")

        if os.path.exists(zarr_path):
            print(f"Loading {key} from Zarr store.")
            ds = xr.open_zarr(zarr_path)
        else:
            print(f"Creating Zarr store for {key} from NetCDF files.")
            ds = xr.open_mfdataset(
                file_pattern,
                preprocess=add_dim,
                concat_dim="sample",
                combine="nested",
            )
            ds.to_zarr(zarr_path, mode="w")

        datas[key] = ds

    return datas
```
### Explanation

This function loads ERA5 NetCDF files for each atmospheric variable and manages their conversion to Zarr format for faster access.  

It loops through each variable and builds the appropriate filename pattern based on whether a pressure level is needed.  

If a Zarr file already exists, it loads the dataset directly from disk.  

If not, it opens all matching NetCDF files using `open_mfdataset()`, applies the `add_dim()` preprocessing function, and saves the output as a Zarr store.  

The result is a dictionary where each key is a variable name (optionally with pressure level) and each value is an xarray Dataset.  

This setup improves performance in repeated runs by skipping the slow NetCDF reading step when cached Zarr files are available.


## Cell [8]: transpose_load_concat() – Reformat and Merge Variables

```python
def transpose_load_concat(data_dictionary):
    transposed = {}
    for key, ds in data_dictionary.items():
        var_name = key.split('_')[0].upper()
        transposed[key] = ds[var_name].expand_dims('features').transpose('sample', 'latitude', 'longitude', 'features')

    if len(transposed) > 1:
        data = xr.concat(list(transposed.values()), dim='features', coords='minimal', compat='override')
    else:
        data = list(transposed.values())[0]

    first_key = next(iter(data_dictionary))
    lat_center = data_dictionary[first_key]['lat_center']
    lon_center = data_dictionary[first_key]['lon_center']
    label = data_dictionary[first_key]['label']
    return data, label, lat_center, lon_center
```
### Explanation

This function transforms and merges the data for all variables into a single 4D tensor.  

Each variable is transposed into the format `(sample, latitude, longitude, features)` to match the expected input shape for CNNs.  

If multiple variables exist, they are concatenated along the `features` dimension to form a multi-channel input tensor.  

It also retrieves shared coordinates and labels (`lat_center`, `lon_center`, and `label`) from one of the datasets. These are assumed to be consistent across all variables.  

The output is ready to be used as input data for training deep learning models.
## Cell [9]: omit_nans() – Filter Out Missing Values

```python
def omit_nans(data, label, lat, lon):
    # If data is an xarray DataArray, convert it to a NumPy array
    if hasattr(data, 'values'):
        data = data.values

    maskarray = np.full(data.shape[0], True)
    masker = np.unique(np.argwhere(np.isnan(data))[:, 0])
    maskarray[masker] = False

    traindata = data[maskarray, ...]
    trainlabel = label[maskarray]
    lat_filtered = lat[maskarray]
    lon_filtered = lon[maskarray]

    return traindata, trainlabel, lat_filtered, lon_filtered
```
### Explanation

This function removes any samples that contain NaN (missing) values in the input data.  

It checks for NaNs across all feature channels and marks entire samples (along the batch dimension) for removal if any NaNs are found.  

It applies the same filtering mask to the input data, labels, and corresponding latitude and longitude coordinates.  

The result is a clean dataset where each sample is fully valid and ready for training.  

Removing NaNs is important to prevent the model from learning incorrect patterns or failing during training.
Here’s the full Markdown block for **Cell [10]: `zscore()` – Z-Score Normalization**, including both the code and the explanation:

````markdown
## Cell [10]: zscore() – Z-Score Normalization

```python
def zscore(data):
    """
    Rescaling the data using z-score (mean/std).
    Each variable gets scaled independently from others.
    Note that we will need to remove test data for formal training.
    """
    for i in range(0, data_.shape[-1]):
        data_[:, :, :, i] = (
            data_[:, :, :, i] - np.nanmean(data_[:, :, :, i])
        ) / np.nanstd(data_[:, :, :, i])
    return data_
````

### Explanation

This function performs **z-score normalization** on the input data.

Each feature (channel) is independently normalized by subtracting its mean and dividing by its standard deviation. This ensures that all variables are on the same scale and helps stabilize model training.

The transformation is applied to each channel across all samples and spatial dimensions.

Note: The function uses `data_` as a global variable instead of a local argument, which may need correction for clarity and reusability.

```

Here’s the full Markdown block for **Cell [11]: `minmax()` – Min-Max Normalization**, including the code and explanation:

````markdown
## Cell [11]: minmax() – Min-Max Normalization

```python
def minmax(data):
    """
    Rescaling the data using min-max.
    Each variable gets scaled independently from others.
    Note that we will need to remove test data for formal training.
    """
    for i in range(0, data_.shape[-1]):
        data_[:, :, :, i] = (
            data_[:, :, :, i] - np.nanmin(data_[:, :, :, i])
        ) / (np.nanmax(data_[:, :, :, i]) - np.nanmin(data_[:, :, :, i]))
    return data
````

### Explanation

This function applies **min-max normalization** to scale the input data to a range between 0 and 1.

Each feature (channel) is scaled independently using its own minimum and maximum values across all samples and spatial locations.

This normalization method helps ensure that all variables have a similar scale, which can improve the convergence of some models.

As with the z-score function, it uses `data_` instead of `data` as an argument, which should be corrected to avoid scope issues.

```

Here’s the full Markdown block for **Cell [12]: `random_split()` – Train-Test Split**, including the code and explanation:

````markdown
## Cell [12]: random_split() – Train-Test Split

```python
def random_split(data, label, split=0.3, seed=0):
    """
    Help splitting data randomly for training and testing.
    """
    np.random.seed(0)

    da_indx = np.random.permutation(data.shape[0])

    data = data[da_indx.astype(int)]
    label = label[da_indx.astype(int)]

    init_range = int(data.shape[0] * (1 - 0.3))

    return data[:init_range], label[:init_range], data[init_range:], label[init_range:]
````

### Explanation

This function randomly splits the dataset into training and testing sets.

* It shuffles the samples using a fixed random seed to ensure reproducibility.
* It then partitions the data based on the `split` ratio (default is 70% train, 30% test).
* Both the input data and labels are shuffled and split in the same order to maintain consistency.

This is useful for quickly testing models without relying on external libraries like `sklearn.model_selection.train_test_split()`.

```

Here’s the full Markdown block for **Cell [13]: `pick_loss()` – Select Loss Function**, including the code and explanation:

````markdown
## Cell [13]: pick_loss() – Select Loss Function

```python
def pick_loss(name, loss_function="sigmoid"):
    """
    Return the appropriate loss function based on a string identifier.
    """
    if name == "binary_crossentropy":
        return tf.keras.losses.BinaryCrossentropy()
    elif name == "categorical_crossentropy":
        return tf.keras.losses.CategoricalCrossentropy()
    elif name == "focal_loss":
        return focal_loss_sigmoid if loss_function == "sigmoid" else focal_loss_softmax
    elif name == "f1_loss":
        return f1_loss_sigmoid if loss_function == "sigmoid" else f1_loss_softmax
````

### Explanation

This function returns a TensorFlow-compatible loss function based on a given string identifier.

* It supports four options: binary crossentropy, categorical crossentropy, focal loss, and F1 loss.
* For custom loss functions like focal and F1 loss, it selects between sigmoid or softmax variants depending on the `loss_function` argument.

This abstraction simplifies experimentation by letting you choose the loss function using just a name string.



## Cell [14]: get_callbacks() – Model Checkpointing and Early Stopping

```python
def get_callbacks(model_save_name):
    """
    Set up model checkpointing and early stopping callbacks.
    """
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{model_save_name}.keras",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
    )
    return [earlystop, checkpoint]
````

### Explanation

This function sets up Keras callbacks to improve model training reliability.

* **EarlyStopping** halts training if the validation loss doesn’t improve for 15 consecutive epochs. It also restores the model weights from the epoch with the best performance.
* **ModelCheckpoint** saves the model to disk whenever there’s an improvement in validation loss.

These callbacks help avoid overfitting and save the best version of the model automatically.

## Cell [15]: build_model() – Define CNN Architecture

```python
def build_model(
    filters,
    kernel_size,
    pool_size,
    learning_rate,
    input_shape,
    loss_function,
    metrics,
):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters, kernel_size, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    return model
````

### Explanation

This function defines a simple convolutional neural network (CNN) for image-like input data.

* It starts with two Conv2D + MaxPool2D + BatchNorm blocks to extract spatial features.
* The output is flattened and passed through a dense layer with dropout for regularization.
* The final dense layer uses a softmax activation to output probabilities for binary classification.

The model is compiled with the specified optimizer, loss function, and evaluation metrics.






## Cell [16]: build_model_single() – Simpler CNN Model

```python
def build_model_single(
    filters,
    kernel_size,
    pool_size,
    learning_rate,
    input_shape,
    loss_function,
    metrics,
):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters, kernel_size, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    return model
````

### Explanation

This function builds a simpler CNN model compared to the previous version.

* It includes only **one** Conv2D + MaxPool2D + BatchNormalization block before flattening.
* The rest of the architecture is identical: a dense layer with dropout, followed by a softmax output layer for binary classification.

This model is suitable for faster experimentation or when working with smaller datasets. It reduces complexity while still preserving key CNN components.




## Cell [17]: build_model_deep() – Deeper CNN Model

```python
def build_model_deep(
    filters,
    kernel_size,
    pool_size,
    learning_rate,
    input_shape,
    loss_function,
    metrics,
):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters, kernel_size, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters * 2, kernel_size, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters * 4, kernel_size, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    return model
````

### Explanation

This function builds a deeper CNN architecture for more complex feature learning.

* It stacks **three convolutional blocks**, with each layer increasing in depth (filters, filters×2, filters×4).
* Batch normalization follows each block to stabilize training.
* After flattening, the model includes two fully connected layers with dropout for regularization.
* The final layer uses softmax for binary classification output.

This deeper design allows the model to learn more hierarchical features, which can be beneficial for more challenging classification tasks or richer datasets.


## Cell [18]: build_model_sequential() – Sequential Model Definition

```python
def build_model_sequential(
    filters,
    kernel_size,
    pool_size,
    learning_rate,
    input_shape,
    loss_function,
    metrics,
):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(filters, kernel_size, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters, kernel_size, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    return model
````

### Explanation

This function defines the same CNN architecture as `build_model()` but uses the **Keras Sequential API** instead of the functional API.

* It's simpler and more readable for straightforward models where layers are added in a single stack.
* The layers include two convolutional blocks with pooling and normalization, followed by dense layers and dropout.
* The final layer uses softmax for binary classification.

The model is compiled with a user-defined loss function, learning rate, and evaluation metrics.

This version is easier to write and understand when building models without complex branching or multiple inputs.




## Cell [19]: objective() – Tuning Objective for KerasTuner

```python
def objective():
    """
    Define the objective metric for Keras Tuner.
    """
    return keras_tuner.engine.objective.Objective("val_loss", direction="min")
````

### Explanation

This function defines the optimization objective for **Keras Tuner**, which is used during hyperparameter tuning.

* The metric to minimize is **validation loss** (`val_loss`), a common choice when the goal is to improve generalization to unseen data.
* The direction is set to `"min"`, indicating that lower values of validation loss are better.

This objective guides the search algorithm (like RandomSearch or Bayesian Optimization) to evaluate and rank different model configurations during tuning.

## Cell [20]: model_builder() – Hyperparameterized Model Builder for Keras Tuner

```python
def model_builder(hp):
    filters = hp.Choice("filters", values=[16, 32, 64])
    kernel_size = hp.Choice("kernel_size", values=[3, 5])
    pool_size = hp.Choice("pool_size", values=[2])
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model = build_model(
        filters=filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        learning_rate=learning_rate,
        input_shape=X_train.shape[1:],
        loss_function=loss_fn,
        metrics=metrics,
    )

    return model
````

### Explanation

This function defines a **hyperparameter search space** for tuning a CNN model using **Keras Tuner**.

* It allows tuning of:

  * `filters`: number of convolutional filters (e.g., 16, 32, or 64)
  * `kernel_size`: size of the convolutional window (e.g., 3 or 5)
  * `pool_size`: max-pooling size (fixed at 2)
  * `learning_rate`: optimizer learning rate (e.g., 0.01, 0.001, or 0.0001)

* These values are passed to the `build_model()` function, which constructs a CNN with the sampled hyperparameters.

This setup lets Keras Tuner explore different configurations to find the best-performing model based on the defined objective (`val_loss`).


## Cell [21]: load_split_data() – Load and Split Data

```python
def load_split_data(
    X, y, lat_center, lon_center, test_size, random_state=42, shuffle=True
):
    (
        X_train,
        X_test,
        y_train,
        y_test,
        lat_train,
        lat_test,
        lon_train,
        lon_test,
    ) = train_test_split(
        X, y, lat_center, lon_center, test_size=test_size, random_state=random_state, shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test, lat_train, lat_test, lon_train, lon_test
````

This function is responsible for splitting the full dataset into training and testing subsets. It takes the input features `X`, the corresponding labels `y`, and the geographic center coordinates (`lat_center`, `lon_center`) for each sample. Using scikit-learn’s `train_test_split`, it creates randomized partitions while preserving alignment between the features, labels, and coordinates. A fixed random state ensures reproducibility of the split, and shuffling is enabled by default to prevent bias due to sample order. The result is a clean separation of data into training and testing sets that can be used for model evaluation and generalization analysis.


## Cell [22]: get_model_name() – Generate Unique Model Name

```python
def get_model_name(base, model_type, use_focal):
    name = base + "_" + model_type
    if use_focal:
        name += "_focal"
    return name
````

This small utility function helps create consistent and descriptive filenames for saving trained models. It takes in a base name, the type of model architecture (such as "deep" or "simple"), and a flag indicating whether focal loss is being used. If focal loss is enabled, the function appends `_focal` to the model name. This naming convention is especially useful when running multiple experiments with different configurations, since it ensures each model is saved under a clear and distinguishable file name that reflects its setup.


## Cell [23]: get_callbacks() – Define Training Callbacks

```python
def get_callbacks(model_save_name):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"{model_save_name}.h5", monitor="val_loss", save_best_only=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )

    return [early_stopping, model_checkpoint, reduce_lr]
````

This function sets up the training callbacks that control how the model is saved and when the training process should adapt or stop. It uses three common callbacks from Keras. The `EarlyStopping` callback monitors the validation loss and stops training if no improvement is seen for 5 consecutive epochs, while also restoring the best weights to prevent overfitting. The `ModelCheckpoint` saves the best version of the model during training based on validation loss, ensuring you don’t accidentally overwrite a good model with a worse one from a later epoch. Finally, `ReduceLROnPlateau` gradually lowers the learning rate if validation loss plateaus, giving the optimizer a chance to settle into a better minimum. Together, these callbacks help make training more stable and efficient.

## Cell [24]: run_experiment() – Full Model Training Pipeline

```python
def run_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    model_type,
    model_save_name,
    use_focal,
    filters,
    kernel_size,
    pool_size,
    learning_rate,
    callbacks,
    metrics,
):
    if use_focal:
        loss_function = focal_loss()
    else:
        loss_function = tf.keras.losses.CategoricalCrossentropy()

    if model_type == "deep":
        model = build_model_deep(
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            learning_rate=learning_rate,
            input_shape=X_train.shape[1:],
            loss_function=loss_function,
            metrics=metrics,
        )
    elif model_type == "sequential":
        model = build_model_sequential(
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            learning_rate=learning_rate,
            input_shape=X_train.shape[1:],
            loss_function=loss_function,
            metrics=metrics,
        )
    else:
        model = build_model(
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            learning_rate=learning_rate,
            input_shape=X_train.shape[1:],
            loss_function=loss_function,
            metrics=metrics,
        )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    return model, history
````

This function serves as the central pipeline for training the model with a given configuration. It handles everything from selecting the loss function, to choosing which architecture to use, to actually fitting the model. If focal loss is enabled, it’s used instead of the default categorical cross-entropy. The function supports multiple model types — deep, sequential, or a default architecture — and constructs the appropriate one based on the input. Once the model is built, it’s trained on the provided training data and validated on the test data using Keras’s `fit()` method. Training includes the use of callbacks for early stopping, checkpointing, and learning rate scheduling. The function returns the trained model along with its training history, making it easy to track performance and reuse the model later.


## Cell [25]: run_tuner_experiment() – Hyperparameter Tuning Pipeline

```python
def run_tuner_experiment(X_train, y_train, X_test, y_test, callbacks, metrics):
    tuner = keras_tuner.RandomSearch(
        model_builder,
        objective=objective(),
        max_trials=5,
        executions_per_trial=1,
        directory="tuner_dir",
        project_name=tuner_project_name,
    )

    tuner.search(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    return best_model, best_hps
````

This function encapsulates the full hyperparameter tuning workflow using Keras Tuner. It initializes a random search tuner with a defined search space from the `model_builder` function and an optimization target set by the `objective()` function — typically the validation loss. The tuning process is limited to five different model configurations (`max_trials=5`), and each configuration is evaluated once (`executions_per_trial=1`). Once the tuner finishes exploring the space, it selects the best-performing set of hyperparameters. A final model is then built using these best hyperparameters and trained from scratch on the dataset. This ensures the final model is not only tuned but also fully trained using the most effective configuration discovered during the search.


## Cell [26]: predict_and_save() – Generate Predictions and Save Results

```python
def predict_and_save(model, X_test, y_test, lat_test, lon_test, filename):
    y_pred = model.predict(X_test)
    y_true = y_test

    df = pd.DataFrame({
        "lat": lat_test,
        "lon": lon_test,
        "y_true": y_true.argmax(axis=1),
        "y_pred": y_pred.argmax(axis=1),
        "p0": y_pred[:, 0],
        "p1": y_pred[:, 1],
    })

    df.to_csv(filename, index=False)
````

This function handles the post-training evaluation step by using the trained model to generate predictions on the test set and saving the results for later analysis. It first performs inference on the input data using the model’s `predict` method. The predicted and true labels are both converted from one-hot format to class indices using `argmax`. It then constructs a pandas DataFrame that includes the latitude and longitude of each sample, the true class, the predicted class, and the predicted probabilities for both classes. Finally, it saves this DataFrame to a CSV file under the specified filename. This exported file can be used for downstream tasks like visualization, spatial analysis, or error inspection.


## Cell [27]: predict_and_eval() – Predict and Evaluate with Metrics

```python
def predict_and_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_true = y_test

    y_pred_label = y_pred.argmax(axis=1)
    y_true_label = y_true.argmax(axis=1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true_label, y_pred_label))
    print("\nClassification Report:")
    print(classification_report(y_true_label, y_pred_label))
````

This function takes a trained model and a test dataset, runs predictions, and evaluates the results using standard classification metrics. It starts by generating the predicted class probabilities with `predict`, and then converts both the predictions and ground truth labels from one-hot encoding into single class indices using `argmax`. The function then prints out a confusion matrix, which shows the counts of true positives, false positives, false negatives, and true negatives. It also prints a classification report that includes precision, recall, F1 score, and support for each class. This gives a detailed look at how well the model is performing, especially in imbalanced classification tasks like AEW detection.


## Cell [28]: plot_roc_curve() – Plot Receiver Operating Characteristic (ROC)

```python
def plot_roc_curve(model, X_test, y_test):
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
````

This function visualizes the ROC curve for the model on the test dataset, which is useful for understanding how well the model separates the two classes. It uses the predicted probabilities of the positive class (class 1) to compute the false positive rate (FPR) and true positive rate (TPR) at various thresholds. These values are then used to plot the ROC curve, along with the area under the curve (AUC) score, which summarizes the model’s performance into a single number. A perfect classifier has an AUC of 1.0, while a random guess gives 0.5. The plot includes a diagonal dashed line as a reference for random performance. This visualization helps in assessing the model’s trade-off between sensitivity and specificity.


## Cell [29]: plot_precision_recall_curve() – Visualize Precision-Recall Tradeoff

```python
def plot_precision_recall_curve(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test[:, 1], y_pred[:, 1])
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, color="purple", lw=2, label=f"PR curve (area = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()
````

This function generates the precision-recall (PR) curve for the test dataset, which is particularly helpful when evaluating models on imbalanced classification problems. Instead of looking at true and false positives globally like the ROC curve, the PR curve focuses on the tradeoff between precision and recall. It calculates how precision changes as the decision threshold for predicting the positive class is varied. The area under this curve (AUC-PR) provides a single metric summarizing the model’s ability to maintain high precision and recall simultaneously. A good model will hug the top-right corner of the plot, showing high precision even at high recall levels. This is critical when false positives and false negatives have unequal costs.


## Cell [30]: save_model_and_history() – Persist Model and Training Progress

```python
def save_model_and_history(model, history, model_save_name):
    model.save(f"{model_save_name}.h5")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f"{model_save_name}_history.csv", index=False)
````

This function saves the trained model and its training history for future use or analysis. The model is stored in HDF5 format (`.h5`), which is a standard format compatible with Keras for saving the complete architecture, weights, and optimizer state. Alongside the model, it also saves the training history — which includes metrics like loss, accuracy, and others across epochs — into a CSV file. This allows for easy reloading of the model without retraining and helps with visualization or debugging by tracking how the model performed over time during training. It's a simple but essential step in machine learning workflows that aim for reproducibility.


## Cell [31]: plot_training_history() – Visualize Training and Validation Curves

```python
def plot_training_history(history):
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["loss"], label="Training Loss")
    plt.plot(history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    if "accuracy" in history_df.columns:
        plt.subplot(1, 2, 2)
        plt.plot(history_df["accuracy"], label="Training Accuracy")
        plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()

    plt.tight_layout()
    plt.show()
````

This function creates visual plots that summarize how the model performed during training. It pulls the recorded metrics from the training history and plots both the loss and, if available, the accuracy curves over each training epoch. The first subplot shows how the training and validation loss evolved — a crucial way to spot overfitting or underfitting. The second subplot (only if accuracy data is available) displays the accuracy trends for both training and validation sets. These visualizations help in diagnosing problems like early overfitting or learning plateaus and are commonly included in reports or presentations to communicate model behavior over time.


## Cell [32]: get_feature_importance() – Permutation-Based Feature Importance

```python
def get_feature_importance(model, X_val, y_val):
    baseline_preds = model.predict(X_val)
    baseline_accuracy = accuracy_score(y_val.argmax(axis=1), baseline_preds.argmax(axis=1))

    num_features = X_val.shape[-1]
    importances = []

    for i in range(num_features):
        X_val_permuted = X_val.copy()
        X_val_permuted[..., i] = np.random.permutation(X_val[..., i])

        permuted_preds = model.predict(X_val_permuted)
        permuted_accuracy = accuracy_score(y_val.argmax(axis=1), permuted_preds.argmax(axis=1))

        importance = baseline_accuracy - permuted_accuracy
        importances.append(importance)

    return importances
````

This function estimates how important each input feature is to the model’s performance using a technique called permutation importance. It starts by computing the model’s baseline accuracy on the original validation set. Then, for each feature channel in the input tensor, it creates a new dataset where that specific feature is randomly shuffled across all samples. This breaks any meaningful relationship the model might have learned from that feature. It then re-evaluates the model on the permuted data and calculates how much the accuracy drops compared to the baseline. A large drop implies that the feature was important for the model’s predictions. This method provides a practical, model-agnostic way of identifying which input channels carry the most predictive signal.


## Cell [33]: visualize_feature_importance() – Plot Feature Importance

```python
def visualize_feature_importance(importances, feature_names):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance (Accuracy Drop)")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.show()
````

This function provides a visual summary of the permutation feature importance results. It takes the list of importance scores — which reflect the drop in model accuracy when each feature was shuffled — and plots them as a horizontal bar chart. The features are displayed on the y-axis, and the magnitude of importance is on the x-axis, showing how much each feature contributed to the model’s performance. Features with longer bars are more important. This kind of plot makes it easier to interpret which variables had the most influence on predictions, which is especially useful for model explainability and for guiding feature selection in future experiments.



## Cell [34]: visualize_prediction_map() – Geospatial Visualization of Model Predictions

```python
def visualize_prediction_map(df, threshold=0.5):
    fig, ax = plt.subplots(figsize=(10, 6))

    df["binary_pred"] = (df["p1"] >= threshold).astype(int)

    scatter = ax.scatter(df["lon"], df["lat"], c=df["binary_pred"], cmap="coolwarm", alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Predicted Label")
    ax.add_artist(legend1)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Predicted AEW Events (Thresholded)")

    plt.grid(True)
    plt.tight_layout()
    plt.show()
````

This function creates a geographical scatter plot of predicted AEW events using latitude and longitude information from the predictions DataFrame. It first applies a threshold to the predicted probability of the positive class (`p1`) to create binary predictions — anything above the threshold is considered an AEW event. These binary predictions are then visualized as colored points on a map, with red and blue indicating different predicted classes. The plot helps assess the spatial distribution of model predictions and is especially useful for spotting geographic biases or regional model behavior. The visualization is a helpful tool when working with climate data where spatial relationships matter.















