from nas_bench_x11.models.lgboost import LGBModel, LGBModelTime
from nas_bench_x11.models.xgboost import XGBModel, XGBModelTime
from nas_bench_x11.models.svd_lgb import SVDLGBModel
from nas_bench_x11.models.svd_lgb_hpo import SVDLGBHPOModel
from nas_bench_x11.models.svd_xgb import SVDXGBModel
from nas_bench_x11.models.svd_nn import SVDNNModel
from nas_bench_x11.models.svd_nn_hpo import SVDNNHPOModel
from nas_bench_x11.models.bezier_nn_hpo import BEZIERNNHPOModel
from nas_bench_x11.models.lc_nn_hpo import LCNNHPOModel

from nas_bench_x11.models.svd_nn_star import SVDNNSModel
from nas_bench_x11.models.bezier_nn_star import BEZIERNNSModel
from nas_bench_x11.models.lc_nn_star import LCNNSModel

from nas_bench_x11.models.nn_E import NNSModel
from nas_bench_x11.models.vae_nn import VAENNModel
from nas_bench_x11.models.vae_lgb import VAELGBModel
from nas_bench_x11.models.vae_xgb import VAEXGBModel
from nas_bench_x11.models.lgb_E import LGBEModel


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