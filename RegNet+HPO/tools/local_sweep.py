import os
import subprocess

import yaml

sweep_setup_path = '/home/v-boli4/teamdrive/msrashaiteamdrive/users/drluodian/pycls_main/tools/sweep_setup.py'

sweep_config_path = '/home/v-boli4/teamdrive/msrashaiteamdrive/users/drluodian/pycls_main/configs/sweeps/cifar/cifar_best.yaml'
# read name from yaml
with open(sweep_config_path, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
    generated_dir_name_tag = data_loaded['NAME']

# generate configs
setup_cmd = "python {} --sweep-cfg {}".format(sweep_setup_path, sweep_config_path)
subprocess.call(setup_cmd, shell=True)

run_file_path = '/home/v-boli4/teamdrive/msrashaiteamdrive/users/drluodian/pycls_main/tools/run_net.py'

# find generate sweep configs with sweep_setup.py
sweep_config_dir_path = '/home/v-boli4/teamdrive/msrashaiteamdrive/users/drluodian/pycls_main/checkpoint/sweeps/{}/cfgs'.format(generated_dir_name_tag)

# list all configs and run them in sequential order
for file in os.listdir(sweep_config_dir_path):
    if file.endswith('.yaml'):
        concat_path = os.path.join(sweep_config_dir_path, file)
        output_dir_path = concat_path[:-5]
        cmd = "python {} --mode train --cfg {} OUT_DIR {}".format(run_file_path, concat_path, output_dir_path)
        subprocess.call(cmd, shell=True)
