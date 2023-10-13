## File Structure

Computation for nearest MRT, mall, school is expensive. Hence, we pre-generated the files for both training and testing data.

Running script need to be triggered at top level, because relative path is used.

## Data Pipeline

### Generate processed data frame

```python
python preprocessing/process_auxiliary.py
```

The pre-generated data is alread in data/train_final.csv and data/test_final.csv. You only need to run this if you have additional features add to the final dataframe. It's important that to not shuffle test.csv, because kaggle submission depends on the ordering. 

### Model selection using k_fold

```python
python training/k_fold.py base_conf 
```

Add your new configuration in config/sk_config.py, and replace the "base_conf" with your new config name. 

### Generate prediction

```python
python prediction/sk_prediction.py base_conf 
```

Add your new configuration in config/sk_config.py, and replace the "base_conf" with your new config name.

### Alternative: Pytorch training (For GPU acceleration only)

```python
python training/nn_pipe.py True 500 base_conf base_model_conf BaseMLPRegressor 0.001 128 0
```

Add new model configuration at config/model_conf.py. Add new model at training/torch_models.py. Notice that the configuration of model must match the model initialization parameters. The above parameters are described below:

FLAGS
    -v, --verbose=VERBOSE
        Default: 1

POSITIONAL ARGUMENTS

    K_FOLD_VAL
    EPOCHES
    CONF_NAME
    MODEL_CONF_NAME
    MODEL_CLASS_NAME
    LR
    BATCH_SIZE
