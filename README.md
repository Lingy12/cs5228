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
python training/k_fold.py <base_conf>
```

Add your new configuration in config/sk_config.py, and replace the "base_conf" with config in config/sk_config.py.

### Generate prediction

```python
python prediction/sk_prediction.py <base_conf> 
```

Add your new configuration in config/sk_config.py, and replace the base_conf" with config in config/sk_config.py..

### Benchmark

```python
python benchmarks.py

python benchmark_with_conf_gen.py <generator_name> <run_name>
```

This script "benchmarks.py" runs all configuration in config/sk_config.py. Which produce the table showing linear regression result and MLP result.

The script "benchmark_with_conf_gen.py" runs a configuration generator in config/config_generator.py. run_name is defined by user for the purpose of this configuration.

### Alternative: Pytorch training (For GPU acceleration only) (Not mained further)

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
