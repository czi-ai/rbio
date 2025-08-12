# rbio-1
![rbio Model Architecture](rbio-motivation-fig.png)


## Usage

We recommend creating a virtual env with:

```
python3 -m venv rbio-env
source rbio-env/bin/activate
pip3 install -r requirements.txt
```

### 1. Inference Scripts
The inference scripts will run an rbio model version on a list of user-provided questions. The script will automatically download the model weights from AWS S3. 

The model arguments are:

| argument | description | default_value |
| - | - | - |
| base_model_name | base model name | Qwen/Qwen2.5-3B-Instruct |
| rbio_model_ckpt | rbio_model_variation | rbio_TF_ckpt 
| results_output_folder | optional folder where to save the results | predictions |
| results_output_filename | optional filename for the results |results.csv |


## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
