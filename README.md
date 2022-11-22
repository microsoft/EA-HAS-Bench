# EA-HAS-Bench: Energy-Aware Hyperparameter and Architecture Search Benchmark
We present the first large-scale energy-aware benchmark that allows studying AutoML methods to achieve better trade-offs between performance and search energy consumption, named EA-HAS-Bench. EA-HAS-Bench provides a **large-scale architecture/hyperparameter joint search space, covering diversified configurations related to energy consumption**. Furthermore, we propose a novel surrogate model specially designed for large joint search space, which proposes a Bezier curve-based model to predict learning curves with unlimited shape and length.

<p align="center">
<img src="BSC\figures\BSC.png" alt="EA-NAS-Bench" width="80%">
</p>

Most of the existing conventional benchmarks like `NAS-Bench-101` do not directly provide training energy cost but use model training time as the training resource budget, which as verified by our experiments, is an inaccurate estimation of energy cost. `HW-NAS-bench`  provides the inference latency and inference energy consumption of different model architectures but also does not provide the search energy cost
<p align="center">
<img src="BSC\figures\Differences.png" alt="Differece" width="80%">
</p>

## Dataset Overview
### EA-HAS-Bench's Search Space
Unlike the search space of existing mainstream NAS-Bench that focuses only on network architectures, our EA-HAS-Bench consists of a combination of two parts: the network architecture space- $\mathrm{RegNet}$  and the hyperparameter space for optimization and training, in order to cover diversified configurations that affect both performance and energy consumption. The details of the search space are shown in Table.

<p align="center">
<img src="BSC\figures\search_space.png" alt="SearchSpace" width="80%">
</p>

### Evaluation Metrics
The EA-HAS-Bench dataset provides the following three types of metrics to evaluate different configurations. 
+ **Model Complexity:** parameter size, FLOPs, number of network activations (the size of the output tensors of each convolutional layer), as well as the inference energy cost of the trained model.
+ **Model Performance:** **full training information** including training, validation, and test accuracy learning curves.
+ **Search Cost:** energy cost (in kWh) and time (in seconds)

### Dataset statistics
The left plot shows the validation accuracy box plots for each NAS benchmark in CIFAR-10. The right plot shows a comparison of training time, training energy consumption (TEC), and test accuracy in the dataset.

<cfigure class="half">
    <img src="BSC\figures\Box_plot.jpg" width="88"/><img src="BSC\figures\Tiny_Acc_as_color.jpg" width="100"/>
</figure>

Although training the model for a longer period is likely to yield a higher energy cost, the final cost still depends on many other factors including power (i.e., consumed energy per hour). The left and right plots of Figure also verifies the conclusion, where the models in the Pareto Frontier on the accuracy-runtime coordinate (right figure) are not always in the Pareto Frontier on the accuracy-TEC coordinate (left figure), showing that training time and energy cost are not equivalent. 

<center class="half">
    <img src="BSC\figures\Tiny_Acc_Cost.jpg" width="292"/><img src="BSC\figures\Tiny_Acc_Time.jpg" width="300"/>
</center>

## Installation

Clone this repository and install its requirements.
```bash
git clone https://github.com/microsoft/EA-HAS-Bench
cd ea_nas_bench
cat requirements.txt | xargs -n 1 -L 1 pip install
pip install -e .
```

Download the pretrained [surrogate models]() and place them into ``BSC/checkpoints/``. The current models are v0.1. 

NOTE: This codebase is still subject to changes. Upcoming updates include improved versions of the surrogate models and code for all experiments from the paper. The API may still be subject to changes.

## Using the API
The api is located in [`api.py`](https://github.com/microsoft/EA-HAS-Bench/blob/main/api.py).

Here is an example of how to use the API:
```python

def get_ea_has_bench_api(dataset):
    full_api = {}
    # load the ea-nas-bench surrogate models
    if dataset=="cifar10":
        ea_has_bench_model = load_ensemble('checkpoints/ea_has_bench-v0.2'))
        train_energy_model = load_ensemble('checkpoints/ea_has_bench-trainE-v0.2')
        test_energy_model = load_ensemble('checkpoints/ea_has_bench-testE-v0.1')
        runtime_model = load_ensemble('checkpoints/ea_has_bench-runtime-v0.1')
    elif dataset=="tiny":
        ea_has_bench_model = load_ensemble('checkpoints/ea-nas-bench-tiny-v0.2')
        train_energy_model = load_ensemble('checkpoints/ea-nas-bench-trainE-v0.1')
        test_energy_model = load_ensemble('checkpoints/ea-nas-bench-testE-v0.1')
        runtime_model = load_ensemble('checkpoints/ea-nas-bench-runtime-v0.1')

    full_api['ea_has_bench_model'] = [ea_has_bench_model, runtime_model, train_energy_model, test_energy_model]
    return full_api

ea_api = get_ea_has_bench_api("cifar10")

# output the learning curve, train time, TEC and IEC
lc = ea_api['ea_has_bench_model'][0].predict(config=arch_str)
train_time = ea_api['ea_has_bench_model'][1].predict(config=arch_str)
train_cost = ea_api['ea_has_bench_model'][2].predict(config=arch_str)
test_cost = ea_api['ea_has_bench_model'][3].predict(config=arch_str)
```

## Run NAS experiments

```
# Supported optimizers: rs, re, {EA}-(ls bananas), hb, bohb 
cd naslib
bash naslib/benchmarks/nas/run_nbgree.sh 
bash naslib/benchmarks/nas/run_nbtoy.sh 
```
Results will be saved in ``results/``.


## How to Re-create EA-HAS-Bench from Scratch
### Sampling points in $\mathrm{RegNet}$ + hpo
For EA-HAS-Bench’s search space that contains both model architectures and hyperparameters, we use random search (RS) to sample unbiased data to build a robust surrogate benchmark.

The following command will train all architecture candidate in the search space.
```
cd RegNet+HPO
python tools/azure_sweep.py --mode amlt --config_path configs\sweeps\cifar\mb_v0.4.yaml
python tools/azure_sweep.py --mode amlt --config_path configs\sweeps\tinyimagenet\mb_v0.1.yaml
```

After training these candidate architectures, please use the following command to re-organize all logs into the single file.

```
python tools/sweep_collect.py
```

### Creating Bézier Curves-based surrogated model
To fit a Bézier Curves-based surrogated model surrogate model run
```
cd BSC
python3 fit_model.py --search_space regnet --model bezier_nn_STAR
```

## Citation
```bibtex
@inproceedings{eahasbench2023,
title={{EA}-{HAS}-Bench: Energy-aware Hyperparameter and Architecture Search Benchmark},
author={Anonymous},
booktitle={Submitted to The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=n-bvaLSCC78},
note={under review}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
