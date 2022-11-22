import argparse
import os
import subprocess

from ruamel import yaml

# executable file path
sweep_setup_path = '/home/v-boli4/teamdrive/msrashaiteamdrive/users/drluodian/pycls_main/tools/sweep_setup.py'
run_file_path = '/home/v-boli4/teamdrive/msrashaiteamdrive/users/drluodian/pycls_main/tools/run_net.py'


def parse_args():
    parser = argparse.ArgumentParser(description='Generate configs and sweeply launch configs locally or submit to AMLT.')
    parser.add_argument('--mode', choices=['local', 'amlt'])
    parser.add_argument('--config_path', help="sweep config path")
    args = parser.parse_args()
    return args


def main():
    global run_file_path
    args = parse_args()
    mode = args.mode
    # notice that the 'cifar/cifar_best' would determine the experiment name/tag.
    sweep_config_path = args.config_path
    # read name from yaml
    with open(sweep_config_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        generated_dir_name_tag = data_loaded['NAME']

    # generate configs
    ENV_NAME = 'torch'  # the environment I setup with conda, if you don't use conda, remove 'conda init' and 'conda activate'.
    SHELL_TYPE = 'zsh'
    setup_cmd = f"""source ~/miniconda3/etc/profile.d/conda.sh
                    conda activate {ENV_NAME}
                    python {sweep_setup_path} --sweep-cfg {sweep_config_path}"""

    subprocess.call(setup_cmd, shell=True, executable=f'/usr/bin/{SHELL_TYPE}')

    exp_name = generated_dir_name_tag.replace('/', '_')  # suppose to be cifar_cifar_best by default
    # find generate sweep configs with sweep_setup.py
    sweep_config_dir_path = '/home/v-boli4/teamdrive/msrashaiteamdrive/users/drluodian/pycls_main/checkpoint/sweeps/{}/cfgs'.format(generated_dir_name_tag)

    # list all configs and run them in sequential order
    config_paths = []
    output_dir_paths = []

    for file in os.listdir(sweep_config_dir_path):
        if file.endswith('.yaml'):
            concat_path = os.path.join(sweep_config_dir_path, file)
            output_dir_path = concat_path[:-5]
            config_paths.append(concat_path)
            output_dir_paths.append(output_dir_path)

    if mode == 'local':
        for i in range(len(config_paths)):
            config_path = config_paths[i]
            output_dir_path = output_dir_paths[i]
            cmd = f"""source ~/miniconda3/etc/profile.d/conda.sh
                conda activate {ENV_NAME}
                python {run_file_path} --mode train --cfg {config_path} WANDB True NOTES {exp_name} OUT_DIR {output_dir_path}"""
            subprocess.call(cmd, shell=True, executable=f'/usr/bin/{SHELL_TYPE}')

    elif mode == 'amlt':
        template_path = './azure_template.yaml'
        with open(template_path, 'r') as stream:
            azure_template = yaml.safe_load(stream)

        root_path = azure_template['code']['local_dir']
        run_file_path = run_file_path.replace(root_path, '.')
        azure_template['description'] = exp_name
        sku = azure_template['jobs'][0]['sku']
        sku_count = azure_template['jobs'][0]['sku_count']  # read the first item
        process_count_per_node = azure_template['jobs'][0]['aml_mpirun']['process_count_per_node']
        communicator = azure_template['jobs'][0]['aml_mpirun']['communicator']
        for indx in range(len(config_paths)):
            # remove absolute prefix when populating yamls (cuz we use it on azure, it should be relative path)
            config_path = config_paths[indx].replace(root_path, '.')
            output_dir_path = output_dir_paths[indx].replace(root_path, '.')
            job_cmd = f"python {run_file_path} --mode train --cfg {config_path} WANDB True NOTES {exp_name} OUT_DIR $$AMLT_OUTPUT_DIR DATA_DIR $$AMLT_DATA_DIR"
            job = {
                'name': os.path.basename(output_dir_path),
                'sku': sku,
                'sku_count': sku_count,
                'aml_mpirun': {
                    'process_count_per_node': process_count_per_node,
                    'communicator': communicator,
                },
                'command': [job_cmd]
            }
            # if index larger than current list size, use append to popluate the list.
            if indx >= len(azure_template['jobs']):
                azure_template['jobs'].append(job)
            else:
                azure_template['jobs'][indx] = job

        # make sure all configs are loaded with cmds into azure template yaml file.
        assert len(azure_template['jobs']) == len(config_paths)

        # the generated yaml file may look different with the original one, but no worry. AMLT reads yaml into dict, it just works well.
        with open('azure_sweep_test.yaml', 'w', encoding='UTF-8') as stream:
            yaml.safe_dump(azure_template, stream)

        amlt_submit_cmd = f"""source ~/miniconda3/etc/profile.d/conda.sh
                conda activate {ENV_NAME}
                amlt run azure_sweep_test.yaml {exp_name} --yes"""

        subprocess.call(amlt_submit_cmd, shell=True, executable=f'/usr/bin/{SHELL_TYPE}')


if __name__ == "__main__":
    main()
