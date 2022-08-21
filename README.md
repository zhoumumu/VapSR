Before testing:

In order to load the right EMA model in pth file(contains two models),

change your Basicsr’s code in position of

'BasicSR/basicsr/models/sr_model.py line29' from

'param_key = self.opt['path'].get('param_key_g', 'params')' to

param_key = self.opt['path'].get('param_key_g', 'params_ema')