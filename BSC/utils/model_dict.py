from BSC.models.lgboost import LGBModel, LGBModelTime
from BSC.models.xgboost import XGBModel, XGBModelTime
from BSC.models.svd_lgb import SVDLGBModel
from BSC.models.svd_lgb_hpo import SVDLGBHPOModel
from BSC.models.svd_xgb import SVDXGBModel
from BSC.models.svd_nn import SVDNNModel
from BSC.models.svd_nn_hpo import SVDNNHPOModel
from BSC.models.bezier_nn_hpo import BEZIERNNHPOModel
from BSC.models.lc_nn_hpo import LCNNHPOModel

from BSC.models.svd_nn_star import SVDNNSModel
from BSC.models.bezier_nn_star import BEZIERNNSModel
from BSC.models.lc_nn_star import LCNNSModel

from BSC.models.nn_E import NNSModel
from BSC.models.vae_nn import VAENNModel
from BSC.models.vae_lgb import VAELGBModel
from BSC.models.vae_xgb import VAEXGBModel
from BSC.models.lgb_E import LGBEModel


model_dict = {

    # NOTE: RUNTIME MODELS SHOULD END WITH "_time"
    'xgb': XGBModel,
    'svd_lgb': SVDLGBModel,
    'svd_lgb_hpo': SVDLGBHPOModel,
    'svd_xgb': SVDXGBModel,
    'svd_nn': SVDNNModel,
    'svd_nn_hpo': SVDNNHPOModel,

    'bezier_nn_hpo': BEZIERNNHPOModel,
    'lc_nn_hpo': LCNNHPOModel,
    'svd_nn_STAR': SVDNNSModel,
    'bezier_nn_STAR': BEZIERNNSModel,
    'lc_nn_STAR': LCNNSModel,

    'nn_E': NNSModel,
    'vae_lgb': VAELGBModel,
    'vae_xgb': VAEXGBModel,
    'vae_nn': VAENNModel,
    'xgb_time': XGBModelTime,
    'lgb': LGBModel,
    'lgb_E': LGBEModel,
    'lgb_time': LGBModelTime,
}