import importlib


def enforce_config_constraints(config):
    """
    check hyperparameters of AlphaFold
    """
    def string_to_setting(s):
        path = s.split('.')
        setting = config
        for p in path:
            setting = setting[p]

        return setting

    mutually_exclusive_bools = [
        (
            "model.template.average_templates", 
            "model.template.offload_templates"
        ),
        (
            "globals.use_lma",
            "globals.use_flash",
        ),
    ]

    for s1, s2 in mutually_exclusive_bools:
        s1_setting = string_to_setting(s1)
        s2_setting = string_to_setting(s2)
        if(s1_setting and s2_setting):
            raise ValueError(f"Only one of {s1} and {s2} may be set at a time")

    fa_is_installed = importlib.util.find_spec("flash_attn") is not None
    if(config["globals"]["use_flash"] and not fa_is_installed):
        raise ValueError("use_flash requires that FlashAttention is installed")

    if(
        config["globals"]["offload_inference"] and 
        not config["model"]["template"]["average_templates"]
    ):
        config["model"]["template"]['offload_templates'] = True
