from yacs.config import CfgNode as CfgNode

SAMPLERS = CfgNode()

# Sampler for uniform sampling from a list of values
SAMPLERS.VALUE_SAMPLER = CfgNode()
SAMPLERS.VALUE_SAMPLER.TYPE = "value_sampler"
SAMPLERS.VALUE_SAMPLER.VALUES = []

# Sampler for floats with RAND_TYPE sampling in RANGE quantized to QUANTIZE
# RAND_TYPE can be "uniform", "log_uniform", "power2_uniform", "normal", "log_normal"
# Uses the closed interval RANGE = [LOW, HIGH] (so the HIGH value can be sampled)
# Note that both LOW and HIGH must be divisible by QUANTIZE
# For the (clipped) normal samplers mu/sigma are set so ~99.7% of samples are in RANGE
SAMPLERS.FLOAT_SAMPLER = CfgNode()
SAMPLERS.FLOAT_SAMPLER.TYPE = "float_sampler"
SAMPLERS.FLOAT_SAMPLER.RAND_TYPE = "uniform"
SAMPLERS.FLOAT_SAMPLER.RANGE = [0.0, 0.0]
SAMPLERS.FLOAT_SAMPLER.QUANTIZE = 0.00001

# Sampler for ints with RAND_TYPE sampling in RANGE quantized to QUANTIZE
# RAND_TYPE can be "uniform", "log_uniform", "power2_uniform", "normal", "log_normal"
# Uses the closed interval RANGE = [LOW, HIGH] (so the HIGH value can be sampled)
# Note that both LOW and HIGH must be divisible by QUANTIZE
# For the (clipped) normal samplers mu/sigma are set so ~99.7% of samples are in RANGE
SAMPLERS.INT_SAMPLER = CfgNode()
SAMPLERS.INT_SAMPLER.TYPE = "int_sampler"
SAMPLERS.INT_SAMPLER.RAND_TYPE = "uniform"
SAMPLERS.INT_SAMPLER.RANGE = [0, 0]
SAMPLERS.INT_SAMPLER.QUANTIZE = 1

# Sampler for a list of LENGTH items each sampled independently by the ITEM_SAMPLER
# The ITEM_SAMPLER can be any sampler (like INT_SAMPLER or even anther LIST_SAMPLER)
SAMPLERS.LIST_SAMPLER = CfgNode()
SAMPLERS.LIST_SAMPLER.TYPE = "list_sampler"
SAMPLERS.LIST_SAMPLER.LENGTH = 0
SAMPLERS.LIST_SAMPLER.ITEM_SAMPLER = CfgNode(new_allowed=True)

# RegNet Sampler with ranges for REGNET params (see base config for meaning of params)
# This sampler simply allows a compact specification of a number of RegNet params
# QUANTIZE for each params below is fixed to: 1, 8, 0.1, 0.001, 8, 1/128, respectively
# RAND_TYPE for each is fixed to uni, log, log, log, power2_or_log, power2, respectively
# Default parameter ranges are set to generate fairly good performing models up to 16GF
# For models over 16GF, higher ranges for GROUP_W, W0, and WA are necessary
# If including this sampler set SETUP.CONSTRAINTS as needed
SAMPLERS.REGNET_SAMPLER = CfgNode()
SAMPLERS.REGNET_SAMPLER.TYPE = "regnet_sampler"
SAMPLERS.REGNET_SAMPLER.DEPTH = [6, 15]
SAMPLERS.REGNET_SAMPLER.W0 = [48, 128]
SAMPLERS.REGNET_SAMPLER.WA = [8.0, 32.0]
SAMPLERS.REGNET_SAMPLER.WM = [2.5, 3.0]
SAMPLERS.REGNET_SAMPLER.GROUP_W = [1, 32]


def load_sampler(sampler_type="REGNET_SAMPLER"):
    full_sampler = SAMPLERS[sampler_type].clone()
    return full_sampler