"""
dump_config.py -- one JSON + one markdown file with every EFFECTIVE constant
the paper's pipeline runs with, for the appendix (reviewers 3.6 and 4.3).

Nothing is re-declared here: every value is read from the place it lives,
at dump time, so the appendix cannot drift from the code.

  checkpoint cfg    torch.load(<model-dir>/diffusion_model.pt)["config"]
  set_diffusion.py  NULL_CLASS / TEST_SPLIT_SEED (imported), the argparse
                    defaults of the surviving `train` / `sample` flags PLUS the
                    frozen module constants that replaced the others (read with
                    `ast`, no execution), and the SOURCE of diffusion_loss, cosine schedule,
                    DDIM+guidance sampler and the EMA update (via `ast`)
  domain_scaling.py every module-level constant (imported)
  HPPMDinamico3.c   every numeric `#define` (regex) + the convergence / failure
                    logic quoted from the source
  verify_maps.py    imported constants + the source of evaluate_trajectory, so the
                    segment-obstacle collision criterion is quoted, not paraphrased

Usage:
    python3 -m src.evaluation.dump_config --model-dir artifacts/ckpt_tiles_seed42 \\
        --output-dir artifacts/config_dump
(`--model-dir` is optional; without it the checkpoint block is left out.)
"""
import argparse
import ast
import inspect
import json
import os
import re

from src.common.compat import load_checkpoint

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SET_DIFFUSION = os.path.join(REPO, "src", "model", "set_diffusion.py")
ENGINE_C = os.path.join(REPO, "engine", "HPPMDinamico3.c")

CKPT_KEYS = ["T", "d_model", "layers", "heads", "n_max", "success_only", "val_frac",
             "test_frac", "test_split_seed", "data", "test_holdout"]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def checkpoint_block(model_dir):
    import torch
    path = os.path.join(model_dir, "diffusion_model.pt")
    ck = load_checkpoint(path)
    cfg = ck["config"]
    out = {"checkpoint": os.path.abspath(path), "step": ck.get("step"), "val_loss": ck.get("val"),
           "config": {k: cfg.get(k, "<absent in this checkpoint>") for k in CKPT_KEYS}}
    out["config"]["hist_n_success_total"] = int(sum(cfg["hist_n_success"])) if "hist_n_success" in cfg else None
    return out


def module_constants(mod):
    """UPPER_CASE, non-callable, JSON-able module attributes."""
    out = {}
    for k, v in vars(mod).items():
        if k.isupper() and not k.startswith("_") and isinstance(v, (int, float, str, bool, dict, list, tuple)):
            out[k] = list(v) if isinstance(v, tuple) else v
    return out


def argparse_defaults(src, subcommand_var, scope):
    """{flag: default} of every `<subcommand_var>.add_argument(...)` in the file,
    found statically. Names (e.g. DEFAULT_DDIM_STEPS) resolve from `scope`."""
    out = {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == subcommand_var):
            continue
        flag = next((a.value for a in node.args if isinstance(a, ast.Constant)
                     and isinstance(a.value, str) and a.value.startswith("--")), None)
        if flag is None:
            continue
        default = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
        action = next((kw.value.value for kw in node.keywords
                       if kw.arg == "action" and isinstance(kw.value, ast.Constant)), None)
        if default is None:
            out[flag] = False if action == "store_true" else None
            continue
        try:
            out[flag] = ast.literal_eval(default)
        except ValueError:
            out[flag] = scope.get(getattr(default, "id", None), ast.unparse(default))
    return out


# `set_diffusion` no longer exposes these as argparse flags: they are frozen
# module constants now. The appendix must keep showing the effective values of
# the campaign, so every frozen flag is mapped to its constant.
SET_DIFFUSION_FROZEN = {
    "e": {
        "--success-only": "SUCCESS_ONLY", "--steps": "STEPS", "--batch": "TRAIN_BATCH",
        "--lr": "LR", "--warmup": "WARMUP", "--ema": "EMA", "--cfg-dropout": "CFG_DROPOUT",
        "--d-model": "D_MODEL", "--layers": "LAYERS", "--heads": "HEADS", "--T": "T",
        "--eval-every": "EVAL_EVERY", "--compile": "COMPILE", "--test-frac": "TEST_FRAC",
        "--val-frac": "VAL_FRAC", "--overwrite-test": "OVERWRITE_TEST",
    },
    "m": {
        "--L": "L_FIXED", "--ports-per-side": "PORTS_PER_SIDE", "--port-margin": "PORT_MARGIN",
        "--max-cluster": "MAX_CLUSTER", "--overlap-margin": "OVERLAP_MARGIN",
        "--guidance": "GUIDANCE", "--eta": "ETA", "--batch": "SAMPLE_BATCH",
        "--m1": "DEFAULT_M1", "--m2": "DEFAULT_M2", "--timeout": "TIMEOUT",
        "--num-obs": "NUM_OBS",
    },
}


def module_constants_src(src, scope):
    """{NAME: value} of the module-level UPPER_CASE constants of `src`,
    evaluated against `scope` (the namespace of domain_scaling)."""
    out = {}
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper()):
            try:
                out[node.targets[0].id] = eval(ast.unparse(node.value), dict(scope), {})
            except Exception:
                continue
    return out


def effective_defaults(src, subcommand_var, scope, frozen):
    """The argparse defaults of the surviving flags, PLUS the values of the flags
    that are now frozen module constants (`frozen`: {flag: constant name})."""
    consts = module_constants_src(src, scope)
    out = dict(argparse_defaults(src, subcommand_var, scope))
    for flag, name in frozen.items():
        if name in consts:
            out[flag] = consts[name]
        elif name in scope:            # shared constant (e.g. DEFAULT_M1 in domain_scaling)
            out[flag] = scope[name]
    return out


def function_source(src, name):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def source_lines(src, pattern):
    return [ln.strip() for ln in src.splitlines() if re.search(pattern, ln)]


def engine_block():
    src = _read(ENGINE_C)
    defines = {}
    for m in re.finditer(r"^#define\s+(\w+)\s+(.+?)\s*(?://.*|/\*.*)?$", src, re.M):
        name, val = m.group(1), m.group(2).strip()
        if re.fullmatch(r"-?[0-9.]+(e-?[0-9]+)?", val):
            defines[name] = float(val) if re.search(r"[.e]", val) else int(val)
        elif name not in ("LIMITAR_ACOS",):
            defines[name] = val   # derived expression, quoted as written
    return {
        "source": os.path.relpath(ENGINE_C, REPO),
        "defines": defines,
        "slopes_m1_m2": {
            "read_from_pendientes_txt": True, "used": False,
            "effective": source_lines(src, r"entorno->pendiente_(inicial|final)\s*="),
            "note": ("pendientes.txt carries m1/m2 but leer_archivos_configuracion reads them into "
                     "temporaries and assigns the hardcoded 1.0 / 4.0 -- the --m1/--m2 flags of the "
                     "Python side have NO effect on the compiled engine.")},
        "max_steps": ("max_pasos_permitidos is read from pendientes.txt (written as 1000 by "
                      "verify_maps.write_hppm_files) and clamped to MAX_PASOS_HISTORIAL"),
        "convergence_criteria": {
            "corrector_converged": ("squared residual of F = (f1,f2,f3) <= "
                                    "TOLERANCIA_ERROR_F^2, i.e. ||F|| <= TOLERANCIA_ERROR_F, within "
                                    "MAX_ITERACIONES_NEWTON damped-Newton iterations "
                                    "(Jacobian diagonal + FACTOR_AMORTIGUACION)"),
            "goal_reached_in_engine": source_lines(src, r"robot\.L_anterior >= 1\.0")},
        "failure_criteria_in_engine": {
            "corrector_fails": ("if the Newton loop ends with residual > tolerance the step radius is "
                                "halved and the step retried; below 1e-6 the path loop breaks"),
            "stagnation": source_lines(src, r"VENTANA_ESTANCAMIENTO|MIN_DISTANCIA_PROGRESO")[:6],
            "loop_bounds": source_lines(src, r"while \(\(robot\.contador_pasos")},
    }


def verifier_block():
    from src.dataset import verify_maps as vm
    return {
        "source": "src/dataset/verify_maps.py",
        "constants": module_constants(vm),
        "route_resolved_criterion": {
            "success": ("engine produced a non-empty tray.txt AND the last point is within "
                      "arrival_radius of the goal (hypot <= arrival_radius) AND no segment of the "
                      "polyline touches an obstacle"),
            "collision": ("distance from every obstacle CENTER to every SEGMENT tray[i]->tray[i+1] "
                         "(projection clipped to [0,1]; the last point is a degenerate segment) is "
                         "<= r_obstacle + robot_radius. Segment-to-obstacle, NOT point-to-obstacle."),
            "loop_stagnation": "no collision but the trajectory did not end within arrival_radius of the goal",
            "timeout": "HPPM subprocess exceeded --timeout seconds (DEFAULT_TIMEOUT unless overridden)",
            "other_states": ["read_failure", "empty_trajectory", "error: <exception>"],
            "numeric_tolerances": {
                "degenerate_segment_len2": 1e-14,
                "robot_radius_L1": vm.ROBOT_RADIUS, "arrival_radius_L1": vm.ARRIVAL_RADIUS_BASE,
                "note": "both radii scale linearly with L (scaled_constants); L=1 for tiles"},
            "universal_tile": ("a tile counts as universal only if ALL 4*ports_per_side port->center "
                               "routes are 'success' (12 at ports_per_side=3); verification stops at the "
                               "first failing route")},
        "evaluate_trajectory_source": function_source(inspect.getsource(vm), "evaluate_trajectory"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=None, help="checkpoint folder (optional)")
    p.add_argument("--output-dir", default="artifacts/config_dump")
    args = p.parse_args()

    import sys
    sys.path.insert(0, REPO)
    from src.common import domain_scaling as ds
    ds_consts = module_constants(ds)

    sd_src = _read(SET_DIFFUSION)
    ds_scope = vars(ds)
    train_defaults = effective_defaults(sd_src, "e", ds_scope, SET_DIFFUSION_FROZEN["e"])
    sample_defaults = effective_defaults(sd_src, "m", ds_scope, SET_DIFFUSION_FROZEN["m"])
    out = {}
    if args.model_dir:
        out["checkpoint"] = checkpoint_block(args.model_dir)

    out["set_diffusion"] = {
        "source": os.path.relpath(SET_DIFFUSION, REPO),
        "NULL_CLASS": int(re.search(r"^NULL_CLASS\s*=\s*(\d+)", sd_src, re.M).group(1)),
        "TEST_SPLIT_SEED": int(re.search(r"^TEST_SPLIT_SEED\s*=\s*(\d+)", sd_src, re.M).group(1)),
        "train_defaults": train_defaults,
        "sample_defaults": sample_defaults,
        "masking": {
            "kind": "padding mask from n (variable-size sets), NOT random token masking",
            "build_mask_source": function_source(sd_src, "build_mask"),
            "where_applied": ("noise eps and x_t are zeroed on padding; attention ignores padding "
                              "(key_padding_mask=~m); the loss counts masked-in elements only")},
        "loss": {
            "diffusion_loss_source": function_source(sd_src, "diffusion_loss"),
            "normalization": ("sum of squared eps-error over the m=True (real, non-padding) obstacle "
                              "elements, divided by m.sum()*3 -- the number of real obstacles in the "
                              "batch times the 3 coordinates (x, y, r)")},
        "noise_schedule": {"kind": "cosine (Nichol & Dhariwal), s=0.008, betas clamped to [1e-5, 0.999]",
                           "source": function_source(sd_src, "cosine_alphas_cumprod")},
        "ema": {"update_lines": source_lines(sd_src, r"d = min\((?:args\.ema|EMA)|pe\.mul_\(d\)"),
                "decay_default": train_defaults.get("--ema"),
                "note": "decay warm-up min(ema, (1+step)/(10+step)); the saved diffusion_model.pt is the EMA copy"},
        "guidance": {
            "kind": "classifier-free: eps = (1+w)*eps_cond - w*eps_null, null class = NULL_CLASS",
            "w_default": sample_defaults.get("--guidance"),
            "disabled_when": "guidance == 0 or the checkpoint was trained --success-only",
            "cfg_dropout_train": train_defaults.get("--cfg-dropout"),
            "sampler_source": function_source(sd_src, "sample_batch")},
        "normalization": {"source": function_source(sd_src, "normalize")},
    }
    out["domain_scaling"] = {
        "source": "src/common/domain_scaling.py",
        "MIN_RADIUS": ds.MIN_RADIUS, "port_margin": ds.DEFAULT_PORT_MARGIN,
        "max_cluster": ds.DEFAULT_MAX_CLUSTER, "overlap_margin": ds.DEFAULT_OVERLAP_MARGIN,
        "confidence_radius": ds.DEFAULT_CONFIDENCE_RADIUS,
        "ports_per_side": ds.DEFAULT_PORTS_PER_SIDE,
        "max_radius": {"formula": "sqrt(MAX_DENSITY * L^2 / (n * pi))", "MAX_DENSITY": ds.MAX_DENSITY,
                       "source": function_source(inspect.getsource(ds), "max_radius"),
                       "example_n10_L1": ds.max_radius(10, 1.0), "example_n50_L1": ds.max_radius(50, 1.0)},
        "scaled_constants_L1": ds.scaled_constants(1.0),
        "all_constants": ds_consts,
    }
    out["hppm_engine"] = engine_block()
    out["verifier"] = verifier_block()

    os.makedirs(args.output_dir, exist_ok=True)
    jpath = os.path.join(args.output_dir, "config_dump.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    mpath = os.path.join(args.output_dir, "config_dump.md")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(to_markdown(out))
    print(f"Wrote {jpath}\nWrote {mpath}")


def _md_table(d):
    rows = ["| name | value |", "|---|---|"]
    rows += [f"| `{k}` | `{json.dumps(v, default=str)}` |" for k, v in d.items()]
    return "\n".join(rows)


def to_markdown(o):
    L = ["# Effective configuration (appendix)\n",
         "Generated by `src/evaluation/dump_config.py`; every value is read from the code, not retyped.\n"]
    if "checkpoint" in o:
        c = o["checkpoint"]
        L += [f"## Checkpoint\n`{c['checkpoint']}` (step {c['step']}, val loss {c['val_loss']})\n",
              _md_table(c["config"]), ""]
    s = o["set_diffusion"]
    L += ["## Model / training (`set_diffusion.py`)\n",
          f"NULL_CLASS={s['NULL_CLASS']} | TEST_SPLIT_SEED={s['TEST_SPLIT_SEED']}\n",
          "### `train` defaults\n", _md_table(s["train_defaults"]), "",
          "### `sample` defaults\n", _md_table(s["sample_defaults"]), "",
          f"### Masking\n{s['masking']['kind']}. {s['masking']['where_applied']}\n",
          f"### Loss\n{s['loss']['normalization']}\n```python\n{s['loss']['diffusion_loss_source']}\n```\n",
          f"### Noise schedule\n{s['noise_schedule']['kind']}\n```python\n{s['noise_schedule']['source']}\n```\n",
          "### EMA\n" + "\n".join(f"`{ln}`" for ln in s["ema"]["update_lines"]) + f"\n\n{s['ema']['note']}\n",
          f"### Guidance\n{s['guidance']['kind']} (w={s['guidance']['w_default']}, "
          f"cfg dropout {s['guidance']['cfg_dropout_train']}); off when {s['guidance']['disabled_when']}.\n"]
    d = o["domain_scaling"]
    L += ["## Domain scaling (`domain_scaling.py`)\n",
          _md_table({k: d[k] for k in ("MIN_RADIUS", "port_margin", "max_cluster", "overlap_margin",
                                        "confidence_radius", "ports_per_side")}),
          f"\nmax_radius(n, L) = `{d['max_radius']['formula']}` with MAX_DENSITY={d['max_radius']['MAX_DENSITY']}\n",
          "Scaled constants at L=1:\n", _md_table(d["scaled_constants_L1"]), ""]
    e = o["hppm_engine"]
    L += ["## HPPM engine (`HPPMDinamico3.c`)\n", _md_table(e["defines"]), "",
          f"**m1/m2:** {e['slopes_m1_m2']['note']}\n",
          f"**Max steps:** {e['max_steps']}\n",
          f"**Corrector converged:** {e['convergence_criteria']['corrector_converged']}\n",
          f"**Corrector fails:** {e['failure_criteria_in_engine']['corrector_fails']}\n"]
    v = o["verifier"]["route_resolved_criterion"]
    L += ["## Route-resolved criterion (`verify_maps.py`)\n"]
    for k in ("success", "collision", "loop_stagnation", "timeout", "universal_tile"):
        L.append(f"- **{k}:** {v[k]}")
    L += ["", "Numeric tolerances:\n", _md_table(v["numeric_tolerances"]),
          "\n```python\n" + (o["verifier"]["evaluate_trajectory_source"] or "") + "\n```\n"]
    return "\n".join(L)


if __name__ == "__main__":
    main()
