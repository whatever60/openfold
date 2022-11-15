import json

from openfold.config import config


if __name__ == "__main__":
    config = config.to_dict()
    config_dir = f"configs"

    with open(f"{config_dir}/base.json", "w") as f:
        json.dump(config, f, indent=4)

    with open(f"{config_dir}/initial_training.json", "w") as f:
        json.dump({}, f, indent=4)

    with open(f"{config_dir}/finetuning.json", "w") as f:
        json.dump(
            {
                "data": {
                    "train": {
                        "crop_size": 384,
                        "max_extra_msa": 5120,
                        "max_msa_clusters": 512,
                    }
                },
                "loss": {
                    "violation": {"weight": 1.0},
                    "experimentally_resolved": {"weight": 0.01},
                },
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/finetuning_no_templ.json", "w") as f:
        # same as finetuning but no templ
        json.dump(
            {
                "data": {
                    "train": {
                        "crop_size": 384,
                        "max_extra_msa": 5120,
                        "max_msa_clusters": 512,
                    }
                },
                "loss": {
                    "violation": {"weight": 1.0},
                    "experimentally_resolved": {"weight": 0.01},
                },
                "model": {"template": {"enabled": False}},
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/model_1.1.json", "w") as f:
        json.dump(
            {
                "data": {
                    "train": {
                        "max_msa_clusters": 512,
                    }
                },
                "loss": {
                    "violation": {"weight": 1.0},
                    "experimentally_resolved": {"weight": 0.01},
                },
            },
            f,
            indent=4
        )
    with open(f"{config_dir}/model_1.2.json", "w") as f:
        # same as 1.1 but no template
        json.dump(
            {
                "data": {
                    "train": {
                        "max_msa_clusters": 512,
                    }
                },
                "loss": {
                    "violation": {"weight": 1.0},
                    "experimentally_resolved": {"weight": 0.01},
                },
                "model": {"template": {"enabled": False}},
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/model_1.1.1.json", "w") as f:
        json.dump(
            {
                "data": {
                    "train": {"crop_size": 384, "max_extra_msa": 5120},
                    "predict": {"max_extra_msa": 5120},
                    "common": {
                        "reduce_max_clusters_by_max_templates": True,
                        "use_template": True,
                        "use_template_torsion_angles": True,
                    },
                },
                "model": {"template": {"enabled": True}},
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/model_1.1.2.json", "w") as f:
        # same as 1.1.1 but less extra seq
        json.dump(
            {
                "data": {
                    "train": {"crop_size": 384},
                    "common": {
                        "reduce_max_clusters_by_max_templates": True,
                        "use_template": True,
                        "use_template_torsion_angles": True,
                    },
                },
                "model": {"template": {"enabled": True}},
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/model_1.2.1.json", "w") as f:
        # same as 1.1.1 but no templ
        json.dump(
            {
                "data": {
                    "train": {"crop_size": 384, "max_extra_msa": 5120},
                    "predict": {"max_extra_msa": 5120},
                },
                "model": {"template": {"enabled": False}},
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/model_1.2.2.json", "w") as f:
        # exactly same as 1.2.1
        json.dump(
            {
                "data": {
                    "train": {"crop_size": 384, "max_extra_msa": 5120},
                    "predict": {"max_extra_msa": 5120},
                },
                "model": {"template": {"enabled": False}},
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/model_1.2.3.json", "w") as f:
        # same as 1.2.2 but less extra seq
        json.dump(
            {
                "data": {"train": {"crop_size": 384}},
                "model": {"template": {"enabled": False}},
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/ptm.json", "w") as f:
        json.dump(
            {
                "loss": {"tm": {"weight": 0.1}},
                "model": {"heads": {"tm": {"enabled": True}}},
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/inference_long_seq.json", "w") as f:
        json.dump(
            {
                "globals": {
                    "offload_inference": True,
                    "use_lma": True,
                    "use_flash": False,
                },
                "model": {
                    "template": {
                        "offload_inference": True,
                        "template_pair_stack": {"tune_chunk_size": False},
                    },
                    "extra_msa": {"extra_msa_stack": {"tune_chunk_size": False}},
                    "evoformer_stack": {"tune_chunk_size": False},
                },
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/train.json", "w") as f:
        json.dump(
            {
                "globals": {
                    "blocks_per_ckpt": 1,
                    "chunk_size": None,
                    "use_lma": False,
                    "offload_inference": False,
                },
                "model": {
                    "template": {"average_templates": False, "offload_templates": False}
                },
            },
            f,
            indent=4
        )

    with open(f"{config_dir}/low_prec.json", "w") as f:
        inf = 1e4
        json.dump(
            {
                "globals": {"eps": 1e-4},
                "model": {
                    "evoformer_stack": {"inf": inf},
                    "extra_msa": {"extra_msa_stack": {"inf": inf}},
                    "recycling_embedder": {"inf": inf},
                    "structure_module": {"inf": inf},
                    "template": {
                        "inf": inf,
                        "template_pair_stack": {"inf": inf},
                        "template_pointwise_attention": {"inf": inf},
                    },
                },
            },
            f,
            indent=4
        )
