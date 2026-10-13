"""Compatibility with artifacts written before the English rename.

The campaign's checkpoints, reports and summaries were produced with Spanish
dictionary keys and Spanish status values (``capas``, ``cabezas``,
``solo_exito``, ``datos``, ``paso``, ``mejor``, ``hist_n_exito``;
``exito``, ``colision``, ``estancamiento_bucle``, ...). Those artifacts are
published as-is and some of them -- notably the six checkpoints -- cannot be
regenerated cheaply, so every read path runs through the normalizers below.

The code itself writes English keys and English status values from now on.
Anything read from disk is normalized to the English vocabulary on the way in.

Used by:
    src/model/set_diffusion.py      (checkpoints)
    src/evaluation/paper_metrics.py (reports and mosaic reports)
    src/evaluation/dump_config.py   (checkpoints)
    src/pipeline/assembler.py       (checkpoints)
    src/pipeline/mosaic_no_hppm.py  (checkpoints)
"""

# ---------------------------------------------------------------------------
# Keys: Spanish (legacy, on disk) -> English (current, in code)
# ---------------------------------------------------------------------------
LEGACY_KEY_ALIASES = {
    # --- checkpoint config and checkpoint top level ---
    "capas": "layers",
    "cabezas": "heads",
    "solo_exito": "success_only",
    "datos": "data",
    "paso": "step",
    "mejor": "best",
    "hist_n_exito": "hist_n_success",
    "hist_n_exito_total": "hist_n_success_total",
    "paso_size": "step_size",
    "semilla": "seed",
    # --- verification results ---
    "estado": "state",
    "estado_final": "final_state",
    "pasos": "steps",
    "pasos_medio": "mean_steps",
    "longitud": "length",
    "longitud_media": "mean_length",
    "longitud_std": "length_std",
    "conteo": "count",
    "conteo_por_estado": "count_by_status",
    "conteos_por_num_obs": "counts_by_num_obs",
    "fallo_en": "failure_at",
    "fallo_lectura": "read_failure",
    "trayectoria_vacia": "empty_trajectory",
    "exito_sin_reparar": "success_before_repair",
    "mapas_exito": "maps_success",
    "mapas_generados_total": "maps_generated_total",
    # --- rates and counts ---
    "tasa_exito_sobre_total_generado": "success_rate_over_total_generated",
    "tasa_exito_verificada": "verified_success_rate",
    "tasa_exito_sin_reparar": "success_rate_before_repair",
    "tasa_aceptacion_universal": "universal_acceptance_rate",
    "tasa_por_num_obs": "rate_by_num_obs",
    "corridas_hppm_por_exito": "hppm_runs_per_success",
    "corridas_hppm_por_exito_con_rechazo_puro": "hppm_runs_per_success_pure_rejection",
    "corridas_hppm_totales": "hppm_runs_total",
    "corridas_totales": "total_runs",
    "candidatos_intentados": "candidates_attempted",
    "candidatos_incompletos": "candidates_incomplete",
    "candidatos_probados": "candidates_tried",
    "candidatos_por_seg": "candidates_per_sec",
    "puertos_por_borde": "ports_per_side",
    "fallos_por_puerto": "failures_per_port",
    "num_obs_evaluados_media": "mean_num_obs_evaluated",
    "num_obs_exitos_media": "mean_num_obs_success",
    "num_obs_medio_exitos": "mean_num_obs_success",
    "iteraciones_mediana": "median_iterations",
    "std_entre_semillas": "std_across_seeds",
    "n_por_semilla": "n_per_seed",
    "BASELINE_media": "BASELINE_mean",
    # --- repair (Gibbs) ---
    "gibbs_aceptados": "gibbs_accepted",
    "gibbs_pasos": "gibbs_steps",
    "gibbs_tasa_aceptacion": "gibbs_acceptance_rate",
    # --- timing ---
    "tiempo_seg": "time_sec",
    "tiempo_total_seg": "total_time_sec",
    "tiempo_verificacion_cpu_seg": "verification_cpu_time_sec",
    # --- mosaic ---
    "filas": "rows",
    "columnas": "columns",
    "n_obstaculos": "n_obstacles",
    "n_por_celda": "n_per_cell",
    "objetivo_obstaculos": "target_obstacles",
    "reintentos_celda_usados": "cell_retries_used",
    "resuelto_primer_intento": "solved_first_try",
    "resuelto_tras_reparacion": "solved_after_repair",
    "verificacion_mosaico": "mosaic_verification",
    # --- verification scope ---
    "inicio": "start",
    "inicio_fijo_si_no_hay_ruta": "fixed_start_if_no_route",
    "criterio": "criterion",
    "criterio_12_rutas_del_tile": "tile_12_route_criterion",
    "rutas_hppm_por_verificacion": "hppm_routes_per_verification",
    "nota": "note",
}

# ---------------------------------------------------------------------------
# Status values: Spanish (legacy) -> English (current)
# ---------------------------------------------------------------------------
LEGACY_STATUS_ALIASES = {
    "exito": "success",
    "colision": "collision",
    "colisiones": "collisions",
    "estancamiento_bucle": "loop_stagnation",
    "tile_universal": "universal_tile",
    "fallo_lectura": "read_failure",
    "trayectoria_vacia": "empty_trajectory",
    "timeout": "timeout",
}


def normalize_key(key):
    """English key for ``key`` (legacy Spanish keys are translated)."""
    if not isinstance(key, str):
        return key
    return LEGACY_KEY_ALIASES.get(key, key)


def normalize_status(value):
    """English status value for ``value`` if it is a legacy status string."""
    if isinstance(value, str):
        return LEGACY_STATUS_ALIASES.get(value, value)
    return value


def normalize(obj):
    """Recursively translate legacy Spanish keys AND status values.

    Safe on any JSON-shaped structure: dicts, lists, scalars. Use it on
    anything loaded from a checkpoint or a JSON report before reading fields.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # A status name can appear as a KEY too, not only as a value:
            # count_by_status is {"exito": 1273, "colision": 3139, ...} in the
            # reports written by the campaign. Normalize both.
            nk = normalize_status(normalize_key(k))
            out[nk] = normalize(v)
            if isinstance(out[nk], str):
                out[nk] = normalize_status(out[nk])
        return out
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, str):
        return normalize_status(obj)
    return obj


def normalize_config(cfg):
    """Checkpoint ``config`` dict -> English keys."""
    return normalize(cfg)


def normalize_checkpoint(ck):
    """Whole checkpoint payload -> English keys (``model``/``opt`` untouched)."""
    return normalize(ck)


def load_json(path, **kwargs):
    """``json.load`` with legacy keys and status values normalized.

    Use this instead of a bare ``json.load`` for any report written before the
    rename: ``reporte.json``, ``reporte_mosaico.json``, the
    ``resultados_detalle.json`` files and the ``resumen.json`` summaries.
    """
    import json

    with open(path, encoding="utf-8") as f:
        return normalize(json.load(f, **kwargs))


def load_checkpoint(path, map_location="cpu"):
    """``torch.load`` with legacy keys and status values normalized."""
    import torch

    return normalize(torch.load(path, map_location=map_location, weights_only=False))
