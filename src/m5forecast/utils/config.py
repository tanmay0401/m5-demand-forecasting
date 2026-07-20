"""Config composition without Hydra's CLI layer.

Why this exists: Hydra 1.3 crashes on Python 3.14 before main() even runs
(3.14's stricter argparse rejects Hydra's help strings). Rather than pin
an older Python, entry points compose the same configs/ tree directly
with OmegaConf. Semantics preserved:

    load_config()                                   -> defaults from config.yaml
    load_config(["model=tft"])                      -> swap a config group
    load_config(["training.horizon=14", "seed=7"])  -> dotlist overrides

Group swaps are recognized by the existence of configs/<group>/<value>.yaml;
anything else is treated as a dotlist override.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(overrides: list[str] | None = None, config_dir: str | Path = CONFIG_DIR) -> DictConfig:
    """Compose configs/config.yaml + its defaults list + CLI-style overrides."""
    config_dir = Path(config_dir)
    root = OmegaConf.load(config_dir / "config.yaml")
    OmegaConf.set_struct(root, False)
    defaults = root.pop("defaults", [])

    cfg = OmegaConf.create()
    merged_self = False
    for item in defaults:
        if item == "_self_":
            cfg = OmegaConf.merge(cfg, root)
            merged_self = True
            continue
        group, name = next(iter(item.items()))
        cfg = OmegaConf.merge(cfg, {group: OmegaConf.load(config_dir / group / f"{name}.yaml")})
    if not merged_self:
        cfg = OmegaConf.merge(cfg, root)

    dotlist: list[str] = []
    for ov in overrides or []:
        key, _, value = ov.partition("=")
        candidate = config_dir / key / f"{value}.yaml"
        if candidate.exists():  # group swap, e.g. model=tft
            cfg = OmegaConf.merge(cfg, {key: OmegaConf.load(candidate)})
        else:  # plain value override, e.g. training.horizon=14
            dotlist.append(ov)
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))

    OmegaConf.set_struct(cfg, True)  # typo'd keys now raise instead of silently creating
    return cfg
