#!/usr/bin/env python3
"""
Probe DX-COM's NPU op-coverage boundaries for Conv2d variants.

Builds a single ONNX with one image-like input, an initial 1x1 expand conv
(lifts 3 -> probe_channels), and then N parallel "probe" Conv branches with
different (kernel, dilation, groups) configurations, all concatenated to one
output.  Compiles with dxcom, then parses the compile log to determine which
probes triggered the "not supported in NPU yet" warning.

The warning DX-COM emits is:
  [WARNING] - Depthwise Conv with dilation [D, D] and kernel (K, K) is not
              supported in NPU yet

We match warnings to our probe configurations by (kernel, dilation, depthwise).

Run from the repo root after activating the DX-COM compiler venv:
  source /media/abbeefai/TheExpanse/dx-all-suite/dx-compiler/venv-dx-compiler-local/bin/activate
  python Models/exports/deepx/probe_deepx.py \\
      --out-dir Models/exports/deepx/probes \\
      --calib-src Models/exports/axelera/SceneSegLite_calib
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


PROBE_CHANNELS = 64
PROBE_HW = 96   # >= 73 so the largest probe (k=3 d=36 effective RF 73) still produces output


@dataclass(frozen=True)
class Probe:
    name: str         # unique identifier, used for ONNX node name
    kernel: int
    dilation: int
    depthwise: bool   # if True, groups=PROBE_CHANNELS; else groups=1


def build_probe_set() -> list[Probe]:
    """Round 1 sweep covering the four hypotheses from the analysis."""
    probes: list[Probe] = []

    # A. Depthwise k=3, sweep dilation.  We know d=2 works (it's elsewhere in
    #    the graph); we want to find the cutoff between d=2 and d=12.
    for d in [1, 2, 4, 6, 8, 10, 12, 24, 36]:
        probes.append(Probe(f"dw_k3_d{d}", kernel=3, dilation=d, depthwise=True))

    # B. Depthwise k=5, sweep dilation.  We know d=2 fails; check d=1 and d=3.
    for d in [1, 3]:
        probes.append(Probe(f"dw_k5_d{d}", kernel=5, dilation=d, depthwise=True))

    # C. Depthwise k=N at dilation=1, sweep kernel size.  Tests whether
    #    kernel-size alone (without dilation) hits any cap, which would
    #    determine viability of the "dilation inflation" ONNX surgery
    #    workaround.
    for k in [7, 9, 11, 17, 25, 49, 73]:
        probes.append(Probe(f"dw_k{k}_d1", kernel=k, dilation=1, depthwise=True))

    # D. Dense (groups=1) Conv2d, k=3, sweep dilation.  Tests whether the
    #    "not supported" cliff is depthwise-specific or dilation-specific.
    for d in [2, 6, 12, 24, 36]:
        probes.append(Probe(f"dense_k3_d{d}", kernel=3, dilation=d, depthwise=False))

    return probes


def build_probe_set_round2() -> list[Probe]:
    """Round 2 sweep: narrow the depthwise-dilation cutoff, find the exact
    kernel-size cap.  Avoids huge dense kernels (the simplifier chokes when
    multiple ~1MB-weight ops are in one graph)."""
    probes: list[Probe] = []

    # A. Depthwise k=3 d=3 — bracket the depthwise k=3 cutoff (we have OK at
    #    d=2 and FAIL at d=4; is d=3 OK or FAIL?).
    probes.append(Probe("dw_k3_d3", kernel=3, dilation=3, depthwise=True))

    # B. Depthwise k=5 d=2 — cross-validate the SceneSegLite stage-7 finding.
    probes.append(Probe("dw_k5_d2", kernel=5, dilation=2, depthwise=True))

    # C. Depthwise k=12..16 d=1 — find exact kernel-size cap (we know k=11 OK,
    #    k=17 FAIL; cap message says "must be smaller than 16" so k=15 should
    #    be max).
    for k in [12, 13, 14, 15, 16]:
        probes.append(Probe(f"dw_k{k}_d1", kernel=k, dilation=1, depthwise=True))

    # D. Dense + k=5 + dilation — does removing depthwise unblock the
    #    stage-7-style k=5 d=2 pattern even without the kernel-inflation
    #    trick?
    for d in [2, 3]:
        probes.append(Probe(f"dense_k5_d{d}", kernel=5, dilation=d, depthwise=False))

    return probes


def build_probe_set_round3() -> list[Probe]:
    """Round 3: small dense+large-kernel probe set kept separate from round 2
    to avoid blowing up the simplifier (each dense k=17 conv has ~4.7 MB of
    weights at 64ch; combining many of these in one graph crashes DX-COM)."""
    probes: list[Probe] = []
    for k in [15, 16, 17]:
        probes.append(Probe(f"dense_k{k}_d1", kernel=k, dilation=1, depthwise=False))
    return probes


def build_probe_onnx(probes: list[Probe], out_path: Path) -> None:
    """Build a single ONNX with all probe branches in parallel."""
    in_h = in_w = PROBE_HW
    in_name = "input"
    input_vi = helper.make_tensor_value_info(in_name, TensorProto.FLOAT,
                                             [1, 3, in_h, in_w])

    rng = np.random.RandomState(0)
    initializers: list = []
    nodes: list = []

    # Initial 1x1 expand: 3 -> PROBE_CHANNELS so the probe convs operate on
    # the same channel count as a real EfficientNet-B1 feature map slice.
    expand_w = rng.randn(PROBE_CHANNELS, 3, 1, 1).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(expand_w, name="expand_w"))
    nodes.append(helper.make_node(
        "Conv", inputs=[in_name, "expand_w"], outputs=["expanded"],
        name="expand_3to64",
        kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1],
        dilations=[1, 1], group=1))

    branch_outputs: list[str] = []
    for p in probes:
        groups = PROBE_CHANNELS if p.depthwise else 1
        w_in_ch = 1 if p.depthwise else PROBE_CHANNELS
        w_shape = (PROBE_CHANNELS, w_in_ch, p.kernel, p.kernel)
        w = (rng.randn(*w_shape).astype(np.float32) * 0.01)
        w_name = f"{p.name}_w"
        initializers.append(numpy_helper.from_array(w, name=w_name))

        # Same-padding so output spatial dims match (clean concat).  For even
        # kernels the total pad is odd, so split asymmetrically (smaller pad
        # on the start side, larger on the end side).  ONNX pads order for
        # 2D Conv is [h_begin, w_begin, h_end, w_end].
        total = (p.kernel - 1) * p.dilation
        pad_begin = total // 2
        pad_end = total - pad_begin
        nodes.append(helper.make_node(
            "Conv", inputs=["expanded", w_name], outputs=[f"{p.name}_out"],
            name=f"conv_{p.name}",
            kernel_shape=[p.kernel, p.kernel],
            pads=[pad_begin, pad_begin, pad_end, pad_end],
            strides=[1, 1],
            dilations=[p.dilation, p.dilation],
            group=groups))
        branch_outputs.append(f"{p.name}_out")

    # Concat all probe outputs into one big output tensor so they all stay
    # live (otherwise DX-COM might prune them as dead branches).
    nodes.append(helper.make_node(
        "Concat", inputs=branch_outputs, outputs=["concat_out"],
        name="probe_concat", axis=1))

    out_vi = helper.make_tensor_value_info(
        "concat_out", TensorProto.FLOAT,
        [1, PROBE_CHANNELS * len(probes), in_h, in_w])

    graph = helper.make_graph(nodes=nodes, name="dxcom_op_probe",
                              inputs=[input_vi], outputs=[out_vi],
                              initializer=initializers)
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # Stable, well-supported across tooling
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))


def write_dxcom_config(json_path: Path, calib_dir: Path,
                       n_calib: int) -> None:
    cfg = {
        "inputs": {"input": [1, 3, PROBE_HW, PROBE_HW]},
        "calibration_num": n_calib,
        "calibration_method": "minmax",
        "default_loader": {
            "dataset_path": str(calib_dir),
            "file_extensions": ["jpeg", "jpg", "png"],
            "preprocessings": [
                {"convertColor": {"form": "BGR2RGB"}},
                {"resize": {"width": PROBE_HW, "height": PROBE_HW}},
                {"div": {"x": 255.0}},
                {"transpose": {"axis": [2, 0, 1]}},
                {"expandDim": {"axis": 0}},
            ],
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(cfg, f, indent=2)


def run_dxcom(dxcom: str, onnx_path: Path, cfg_path: Path,
              out_dir: Path, log_path: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [dxcom, "-m", str(onnx_path), "-c", str(cfg_path),
           "-o", str(out_dir)]
    print(f"[INFO] Running: {' '.join(cmd)}")
    print(f"[INFO] Log: {log_path}")
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        logf.write(r.stdout)
        sys.stdout.write(r.stdout)
    return r.returncode


# Two distinct warning forms DX-COM 2.2.0 emits:
#   1) Depthwise/dilation combinations:
#        Depthwise Conv with dilation [D, D] and kernel [K, K] is not supported in NPU yet
#   2) Kernel-size cap (applies to depthwise and dense both):
#        Conv node conv_NAME to CPU: Kernel size [K, K] must be smaller than 16
DW_DILATION_RE = re.compile(
    r"Depthwise Conv with dilation \[(\d+),\s*\d+\] and kernel \[(\d+),\s*\d+\] is not supported")
KSIZE_CAP_RE = re.compile(
    r"Conv node (\S+) to CPU: Kernel size \[(\d+),\s*\d+\] must be smaller than 16")


def parse_unsupported(log_path: Path,
                      probes: list[Probe]) -> dict[str, tuple[str, str]]:
    """Return {probe_name: (verdict, reason)} for each probe.

    verdict is "OK" or "FAIL". reason is "" or a short explanation."""
    # Collect (depthwise, kernel, dilation) signatures flagged unsupported.
    dw_dilation_unsupported: set[tuple[int, int]] = set()
    # Collect per-node kernel-size cap rejections.
    ksize_rejected_nodes: set[str] = set()
    with open(log_path) as f:
        for line in f:
            m = DW_DILATION_RE.search(line)
            if m:
                d = int(m.group(1))
                k = int(m.group(2))
                dw_dilation_unsupported.add((k, d))
                continue
            m = KSIZE_CAP_RE.search(line)
            if m:
                ksize_rejected_nodes.add(m.group(1))

    results: dict[str, tuple[str, str]] = {}
    for p in probes:
        node_name = f"conv_{p.name}"
        if node_name in ksize_rejected_nodes:
            results[p.name] = ("FAIL", f"kernel size {p.kernel} >= 16 cap")
        elif p.depthwise and (p.kernel, p.dilation) in dw_dilation_unsupported:
            results[p.name] = ("FAIL", f"depthwise k={p.kernel} d={p.dilation} unsupported")
        else:
            results[p.name] = ("OK", "")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output dir for probe ONNX + dxcom artifacts")
    p.add_argument("--calib-src", type=Path, required=True,
                   help="Directory of calibration images (the existing "
                        "SceneSegLite_calib/ symlinks are fine)")
    p.add_argument("--n-calib", type=int, default=20,
                   help="Calibration set size (default 20 - we just need to "
                        "get past quantization, not optimize accuracy)")
    p.add_argument("--round", type=int, default=1, choices=[1, 2, 3],
                   help="Probe set to run (1 = initial broad sweep, "
                        "2 = follow-up narrowing of depthwise dilation cutoff "
                        "and kernel-size cap, "
                        "3 = dense + large-kernel kept separate from round 2 "
                        "to avoid simplifier OOM on combined large weights)")
    args = p.parse_args()

    dxcom = shutil.which("dxcom") or shutil.which("dx_com")
    if dxcom is None:
        print("[ERROR] dxcom not found in PATH.  Activate the compiler venv:")
        print("[ERROR]   source /media/abbeefai/TheExpanse/dx-all-suite/dx-compiler/venv-dx-compiler-local/bin/activate")
        sys.exit(1)

    if not args.calib_src.exists():
        print(f"[ERROR] Calibration source does not exist: {args.calib_src}")
        sys.exit(1)

    if args.round == 1:
        probes = build_probe_set()
    elif args.round == 2:
        probes = build_probe_set_round2()
    else:
        probes = build_probe_set_round3()
    print(f"[INFO] Round {args.round}: {len(probes)} probes:")
    for p in probes:
        groups = "depthwise" if p.depthwise else "dense"
        rf = (p.kernel - 1) * p.dilation + 1
        print(f"  {p.name:18s}  kernel={p.kernel:2d} dilation={p.dilation:2d} "
              f"{groups:9s}  effective RF = {rf}x{rf} = {rf*rf}")

    onnx_path = args.out_dir / "probe.onnx"
    cfg_path = args.out_dir / "probe.json"
    log_path = args.out_dir / "probe_compile.log"

    print(f"[INFO] Building probe ONNX: {onnx_path}")
    build_probe_onnx(probes, onnx_path)
    print(f"[INFO] Writing DX-COM config: {cfg_path}")
    write_dxcom_config(cfg_path, args.calib_src, args.n_calib)

    rc = run_dxcom(dxcom, onnx_path, cfg_path, args.out_dir, log_path)
    if rc != 0:
        print(f"[WARN] dxcom returned rc={rc}; check log: {log_path}")
        # Don't bail - warning lines may still be present.

    print()
    print("=" * 90)
    print(f"DX-COM op-coverage probe results "
          f"(channels={PROBE_CHANNELS}, input={PROBE_HW}x{PROBE_HW})")
    print("=" * 90)
    results = parse_unsupported(log_path, probes)
    n_fail = sum(1 for v, _ in results.values() if v == "FAIL")
    print(f"\n{n_fail} of {len(probes)} probes rejected by DX-COM (sent to CPU).\n")

    print(f"{'Probe':18s} {'kernel':>6s} {'dilation':>8s} {'groups':>9s} "
          f"{'eff RF':>8s}  {'NPU':4s}  reason")
    print("-" * 90)
    for p in probes:
        groups = "depthwise" if p.depthwise else "dense"
        rf = (p.kernel - 1) * p.dilation + 1
        verdict, reason = results[p.name]
        print(f"{p.name:18s} {p.kernel:>6d} {p.dilation:>8d} {groups:>9s} "
              f"{rf:>3d}x{rf:<3d}  {verdict:4s}  {reason}")


if __name__ == "__main__":
    main()
