#!/usr/bin/env python3
"""flag_audit.py -- acceptance checker for WORK B (configuration freeze).

Compares the argparse surface of a BASELINE git ref against the WORKING TREE
(or an explicit --work directory) and fails on any violation of the freeze
protocol.  Read-only: it never writes inside the repository.

    python3 /tmp/audit/flag_audit.py --base HEAD --work . --whitelist /tmp/audit/whitelist.txt
Exit code = number of FAILs (0 = accepted).
"""
import argparse, ast, glob, json, os, re, subprocess, sys, py_compile

# ---------------------------------------------------------------- AST helpers
def parse(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def safe_parse(src, label):
    """Never crash the audit because of a broken file: report and degrade."""
    try:
        return ast.parse(src)
    except SyntaxError as e:
        fail(f"[S0 syntax] {label}: {e.msg} (line {e.lineno})")
        return ast.Module(body=[], type_ignores=[])

def module_constants(src, extra_ns=None):
    """UPPER_CASE module-level Name = <literal/expression> resolved best-effort.
    extra_ns supplies names imported from other modules (DEFAULT_L, STRICT_... )."""
    tree = safe_parse(src, '<file>')
    raw = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            raw[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            raw[node.target.id] = node.value
    ns = {"str": str, "float": float, "int": int, "max": max, "min": min, "abs": abs, "bool": bool, "len": len}
    if extra_ns:
        ns.update(extra_ns)
    for _ in range(4):                      # resolve chains a few times
        for k, v in raw.items():
            try:
                code = compile(ast.Expression(v), "<c>", "eval")
                ns[k] = eval(code, {"__builtins__": {}}, ns)
            except Exception:
                pass
    return raw, ns

def flags_of(src, extra_ns=None):
    """[(flag, parser_var, dest, action, type_src, nargs, required, default_src)]
    plus the dests of every declared flag."""
    raw, ns = module_constants(src, extra_ns)
    out = {}
    for node in ast.walk(safe_parse(src, '<file>')):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        var = node.func.value.id if isinstance(node.func.value, ast.Name) else "?"
        flag = next((a.value for a in node.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value[:1] == "-"), None)
        if flag is None:
            continue
        kw = {k.arg: k.value for k in node.keywords}
        default = kw.get("default")
        action = kw["action"].value if isinstance(kw.get("action"), ast.Constant) else None
        dsrc = ast.unparse(default) if default is not None else None
        if default is None:
            dval, dtype = (False if action == "store_true" else (True if action == "store_false" else None)), bool if action else type(None)
        else:
            try:
                dval = eval(compile(ast.Expression(default), "<d>", "eval"), {"__builtins__": {}}, ns)
                dtype = type(dval)
            except Exception:
                dval, dtype = "<EXPR>", None
        if flag.startswith("--"):
            dest = kw["dest"].value if isinstance(kw.get("dest"), ast.Constant) else flag[2:].replace("-", "_")
        else:
            dest = kw["dest"].value if isinstance(kw.get("dest"), ast.Constant) else flag.lstrip("-")
        out[flag] = {"var": var, "dest": dest, "action": action,
                     "type": ast.unparse(kw["type"]) if "type" in kw else None,
                     "nargs": ast.unparse(kw["nargs"]) if "nargs" in kw else None,
                     "required": bool(getattr(kw.get("required"), "value", False)),
                     "default_src": dsrc, "value": dval, "vtype": dtype}
    return out, ns, raw

def args_attrs(src):
    """{dest: [lineno]} of every args.<name> / A.<name> attribute read."""
    hits = {}
    for node in ast.walk(safe_parse(src, "<file>")):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("args", "A"):
            hits.setdefault(node.attr, []).append(node.lineno)
    return hits

def parser_dests(src):
    """dests that never come from add_argument: subparsers and set_defaults."""
    out = set()
    for node in ast.walk(safe_parse(src, "<file>")):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_subparsers":
                for kw in node.keywords:
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                        out.add(kw.value.value)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "set_defaults":
                for kw in node.keywords:
                    if kw.arg:
                        out.add(kw.arg)
    return out

def signatures(src):
    """Structural fingerprint: top-level names + every function signature."""
    tree = safe_parse(src, "<file>")
    sig = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig.append(("def", node.name, ast.unparse(node.args)))
        elif isinstance(node, ast.ClassDef):
            sig.append(("class", node.name, [ast.unparse(b) for b in node.bases]))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig.append(("method", node.name + "." + sub.name, ast.unparse(sub.args)))
    return sig

def code_only(src):
    """AST with every bare-string expression (docstrings) removed -> comment-only
    edits and docstring edits both vanish; real code edits do not."""
    tree = safe_parse(src, "<file>")
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if isinstance(body, list):
                setattr(node, field, [s for s in body
                                      if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                              and isinstance(s.value.value, str))])
    return ast.dump(tree, include_attributes=False)

# ------------------------------------------------------------------ reporting
FAIL, WARN, OK = [], [], []
def fail(m): FAIL.append(m)
def warn(m): WARN.append(m)
def ok(m): OK.append(m)

# A whitelisted flag is normally required to survive in EVERY file. These are
# the deliberate per-file exceptions: --num-obs-min/--num-obs-max must stay
# flags in src/model/set_diffusion.py, because paper_metrics passes the per-bin
# bounds to it ("uniform n per bin", the regime of Table 1), but nothing ever
# passes them to the generators, so there they are frozen like any other
# content parameter.
FROZEN_ANYWAY = {
    ("src/dataset/fast_generate.py", "--num-obs-min"),
    ("src/dataset/fast_generate.py", "--num-obs-max"),
    ("src/dataset/repaired_generator.py", "--num-obs-min"),
    ("src/dataset/repaired_generator.py", "--num-obs-max"),
    ("src/dataset/universal_generator.py", "--num-obs-min"),
    ("src/dataset/universal_generator.py", "--num-obs-max"),
}


def py_files(root):
    """Every .py in the repo except the audit tooling itself: tools/ is not part
    of the paper's code and has flags of its own (--whitelist, --work, ...)."""
    return sorted(p for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True)
                  if "/.git/" not in p and "/tools/" not in p.replace(os.sep, "/"))

def git_show(ref, path):
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--work", default=".")
    ap.add_argument("--whitelist", required=True)
    ap.add_argument("--deviations", default=None, help="JSON: {flag: expected_literal_value}")
    ap.add_argument("--code-only", nargs="*", default=[], help="files where only comments/docstrings may change")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    whitelist = {ln.strip() for ln in open(a.whitelist) if ln.strip() and not ln.startswith("#")}
    deviations = json.loads(open(a.deviations).read()) if a.deviations else {}
    work = os.path.abspath(a.work)
    repo = work

    # ---- stage 0: the tree must compile (B and C both) ----
    broken = set()
    for p in py_files(work):
        rel = os.path.relpath(p, work)
        try:
            py_compile.compile(p, doraise=True, cfile="/tmp/audit/_pyc")
        except py_compile.PyCompileError as e:
            broken.add(rel)
            fail(f"[S0 syntax] {rel}: {str(e).strip().splitlines()[0]}  (everything else in this "
                 f"file is skipped)")

    # repo-wide constant namespace (defaults are often imported across modules)
    ns_all = {}
    for pass_ in range(4):
        for p in py_files(work):
            ns_all.update(module_constants(parse(p), ns_all)[1])

    base_flags, work_flags = {}, {}
    consts_work, raw_work, sig_before, sig_after, attrs = {}, {}, {}, {}, {}
    sub_dests = {}
    for p in py_files(work):
        rel = os.path.relpath(p, work)
        src_after = parse(p)
        sig_after[rel] = signatures(src_after)
        attrs[rel] = args_attrs(src_after)
        sub_dests[rel] = parser_dests(src_after)
        if rel in broken:
            base_flags[rel] = {}
            continue
        f, ns, raw = flags_of(src_after, ns_all)
        work_flags[rel] = f
        raw_work[rel] = raw
        consts_work[rel] = {k: ns[k] for k in raw if k in ns}
        src_before = git_show(a.base, rel)
        if src_before is None:
            warn(f"[S1] {rel} is new relative to {a.base} (no baseline to compare)")
            base_flags[rel] = {}
            continue
        sig_before[rel] = signatures(src_before)
        base_flags[rel] = flags_of(src_before, ns_all)[0]

    # ---- stage 1: surviving set == whitelist ----
    surviving = {fl for f in work_flags.values() for fl in f}
    missing, extra = whitelist - surviving, surviving - whitelist
    for fl in sorted(missing):
        fail(f"[S1 whitelist] '{fl}' must survive as a flag but no longer exists")
    for fl in sorted(extra):
        fail(f"[S1 whitelist] '{fl}' is still a flag but is not whitelisted (must be frozen)")

    # ---- stage 2/3/4: frozen flags -> constant with the SAME value and type ----
    for rel, bf in base_flags.items():
        wf = work_flags.get(rel, {})
        attrs_rel, declared_extra = attrs.get(rel, {}), sub_dests.get(rel, set())
        for fl, spec in bf.items():
            dest, spec_v, spec_t = spec["dest"], spec["value"], spec["vtype"]
            if fl in whitelist:
                if fl not in wf and (rel, fl) not in FROZEN_ANYWAY:
                    fail(f"[S2whitelist] {rel}: '{fl}' is whitelisted (must survive) but was frozen in "
                         f"this file")
                continue
            if fl in wf:
                continue                                    # globally: stage 1 already failed
            want = deviations.get(fl, spec_v)
            if spec_v == "<EXPR>":
                warn(f"[S2] {rel}: '{fl}' default is an expression ({spec['default_src']}) -- "
                     f"verify the constant by hand")
            found = [n for n, v in consts_work[rel].items()
                     if n.isupper() and not n.startswith("_") and isinstance(v, type(want)) and v == want]
            if not found:
                elsewhere = [n for n, v in ns_all.items()
                             if n.isupper() and not n.startswith("_") and isinstance(v, type(want)) and v == want]
                if elsewhere:
                    ok(f"[S2 imported] {rel}: '{fl}' -> {want!r} via an existing shared constant "
                       f"({elsewhere[0]})")
            if not found and not elsewhere:
                fail(f"[S2value] {rel}: '{fl}' (dest={dest}) frozen but no UPPER_CASE constant == "
                     f"{want!r} ({type(want).__name__}) exists in the file"
                     + (f"  [authorized deviation: {deviations[fl]!r}]" if fl in deviations else ""))
            if dest and dest in attrs_rel:
                fail(f"[S3orphan] {rel}: 'args.{dest}' still used at lines {attrs_rel[dest]} after "
                     f"'{fl}' was frozen -> substitute the constant")
        # ---- stage 4: orphan args.* (declared by no surviving flag / subparser) ----
        declared = {s["dest"] for s in wf.values()} | declared_extra
        for name, lines in attrs_rel.items():
            if name not in declared:
                fail(f"[S4] {rel}: 'args.{name}' read at lines {lines} but no add_argument/"
                     f"add_subparsers declares dest='{name}' (orphan)")

    # ---- stage 5: required flags have no default -> cannot be frozen ----
    for rel, bf in base_flags.items():
        for fl, spec in bf.items():
            if spec["required"] and fl not in work_flags.get(rel, {}):
                fail(f"[S5] {rel}: '{fl}' was required=True and is gone -- a required flag has no "
                     f"default to freeze, so it must survive")

    # ---- stage 6: flags used by runnable documented commands ----
    doc_runnable = set()
    for doc in ["README.md"] + glob.glob(os.path.join(repo, "docs", "*.md")):
        t = parse(doc)
        for block in re.findall(r"```[a-z]*\n(.*?)```", t, re.S):
            if "python3" in block:
                doc_runnable |= set(re.findall(r"--[a-z][a-z0-9-]+", block))
    for fl in sorted(doc_runnable - surviving):
        fail(f"[S6 docs] '{fl}' appears in a runnable documented command but is no longer a flag")

    # ---- stage 7: no restructuring (B and C) ----
    for rel in sig_after:
        if rel in broken:
            continue
        if rel in sig_before and sig_before[rel] != sig_after[rel]:
            # Only signatures that CHANGED or DISAPPEARED are a violation:
            # adding a new helper is not a restructure.
            gone_or_changed = [x for x in sig_before[rel] if x not in sig_after[rel]]
            added = [x for x in sig_after[rel] if x not in sig_before[rel]]
            if gone_or_changed:
                fail(f"[S7 structure] {rel}: function/class signature changed or removed: "
                     f"{gone_or_changed}")
            if added:
                warn(f"[S7 structure] {rel}: new function(s) added (not a violation): {added}")

    # ---- stage 8: comment-only edits where declared ----
    for rel in a.code_only:
        p = os.path.join(work, rel)
        before = git_show(a.base, rel)
        if before is None:
            warn(f"[S8] {rel} not in {a.base}")
            continue
        if rel in broken:
            continue
        if code_only(before) != code_only(parse(p)):
            fail(f"[S8 comment-only] {rel}: the AST without docstrings changed -> code (not only "
                 f"comments) was touched")

    # ---- stage 9: traceability oracle (paper_metrics _config vs shipped summary.json) ----
    pm = work_flags.get(os.path.join("src", "evaluation", "paper_metrics.py"), {})
    sump = os.path.join(repo, "artifacts", "metrics_seed101", "summary.json")
    if pm and os.path.exists(sump):
        shipped = json.load(open(sump))["_config"]
        cli = {"model_dir": "artifacts/ckpt_tiles_seed101", "hppm_exe": "artifacts/hppm_v2",
               "training_data": "tile-obstacle-maps.npz", "reference_data": "artifacts/test_holdout_dedup.npz",
               "output_dir": "artifacts/metrics_seed101", "tests": ["cost"], "workers": 30, "seed": 42,
               "training_log": "train_seed101.log"}
        got = {}
        for fl, s in pm.items():
            got[s["dest"]] = cli.get(s["dest"], s["value"])
        # Options frozen into constants are no longer flags, so vars(args) no
        # longer sees them. paper_metrics re-injects them into _config through a
        # FROZEN_CONFIG dict; read that map and resolve each constant's value
        # from the module so the oracle follows the code instead of the flags.
        src_pm = open(os.path.join(repo, "src", "evaluation", "paper_metrics.py"),
                      encoding="utf-8").read()
        m = re.search(r"FROZEN_CONFIG\s*=\s*\{(.*?)\n\}", src_pm, re.S)
        if m:
            for key, const in re.findall(r'"([a-z_]+)"\s*:\s*([A-Z][A-Z0-9_]*)', m.group(1)):
                cm = re.search(rf"^{const}\s*=\s*([^#\n]+)", src_pm, re.M)
                if not cm:
                    continue
                val = cm.group(1).strip()
                try:
                    got[key] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    # The constant may be imported, e.g.
                    # PORT_MARGIN = DEFAULT_PORT_MARGIN. Resolve it from the
                    # module that defines it.
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", val):
                        for mod in ("src/common/domain_scaling.py",
                                    "src/common/tile_descriptors.py"):
                            try:
                                mt = open(os.path.join(repo, mod), encoding="utf-8").read()
                            except OSError:
                                continue
                            dm = re.search(rf"^{val}\s*=\s*([^#\n]+)", mt, re.M)
                            if dm:
                                try:
                                    got[key] = ast.literal_eval(dm.group(1).strip())
                                except (ValueError, SyntaxError):
                                    got[key] = dm.group(1).strip()
                                break
                        else:
                            got[key] = val
                    else:
                        got[key] = val
        for k in sorted(set(shipped) | set(got)):
            if k not in got:
                fail(f"[S9 traceability] _config key '{k}' is in the shipped summary.json but no longer "
                     f"produced by any flag/constant of paper_metrics.py")
            elif got[k] != shipped[k]:
                fail(f"[S9 traceability] _config['{k}']: shipped {shipped[k]!r} != {got[k]!r} now")
        if FAIL:
            warn("[S9] if these are frozen constants, paper_metrics must keep writing them into "
                 "_config (vars(args) at paper_metrics.py:818 no longer sees them)")

    rep = {"baseline": a.base, "work": a.work, "whitelist_size": len(whitelist),
           "surviving": sorted(surviving), "missing_whitelist": sorted(missing),
           "unexpected_flags": sorted(extra), "FAIL": FAIL, "WARN": WARN, "OK": OK}
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=2)
    print(f"baseline={a.base}  work={a.work}")
    for m in FAIL: print("FAIL  " + m)
    for m in WARN: print("WARN  " + m)
    print(f"\n{len(FAIL)} FAIL / {len(WARN)} WARN / whitelist={len(whitelist)} surviving={len(surviving)}")
    return min(len(FAIL), 250)

if __name__ == "__main__":
    sys.exit(main())
