#!/usr/bin/env python3
"""
MemryX MX3 compile flow for AutoSeg-umbrella ONNX models.

Pins the dynamic batch axis with onnxsim, then drives memryx.NeuralCompiler
to produce a .dfp Dataflow Program for the 4-chip MX3 M.2 module.

Example:
    python memryx_flow.py \\
        --onnx Models/exports/onnx_rt/SceneSegLite.onnx \\
        --out-dir Models/exports/memryx \\
        --input-shape 320,640,3
"""
import argparse
import shutil
import sys
from pathlib import Path

import onnx
from onnxsim import simplify

try:
    from memryx import NeuralCompiler
except ImportError:
    print("[ERROR] Could not import memryx. Activate the MemryX venv "
          "(source /media/abbeefai/TheExpanse/venv-mx/bin/activate) "
          "or install via 'pip install --extra-index-url "
          "https://developer.memryx.com/pip memryx'.")
    sys.exit(1)


def pin_batch_axis(src_onnx: Path, dst_onnx: Path, input_name: str,
                   input_shape: tuple[int, int, int, int],
                   use_onnxsim: bool = True) -> None:
    """Overwrite the dynamic batch axis on the ONNX input.

    MemryX's NeuralCompiler requires a static input shape. The Lite-family
    PyTorch export marks dim 0 as 'batch_size'; this folds that to a concrete
    value (typically 1).

    When use_onnxsim is True we run a full simplification pass (folds constants
    and validates the graph). When False we just rewrite the input's dim_value
    in-place — needed when the ONNX uses opset features that the installed
    onnx checker rejects (e.g. PyTorch 2.3.1 emits opset-18 ReduceMean using
    the old 'axes' attribute form, which trips both onnxsim and the official
    checker).
    """
    print(f"[INFO] Loading ONNX: {src_onnx}")
    model = onnx.load(str(src_onnx))

    if use_onnxsim:
        overwrite = {input_name: list(input_shape)}
        print(f"[INFO] Simplifying with overwrite_input_shapes={overwrite}")
        simplified, ok = simplify(model, overwrite_input_shapes=overwrite)
        if not ok:
            print("[ERROR] onnxsim could not validate the simplified model")
            sys.exit(1)
        out = simplified
    else:
        # Locate the named input and rewrite its dims in-place.
        found = False
        for inp in model.graph.input:
            if inp.name != input_name:
                continue
            dims = inp.type.tensor_type.shape.dim
            if len(dims) != len(input_shape):
                print(f"[ERROR] Input '{input_name}' has rank {len(dims)} "
                      f"but --batch-shape has rank {len(input_shape)}")
                sys.exit(1)
            for d, val in zip(dims, input_shape):
                # ONNX's TensorShapeProto.Dimension is a `oneof` of
                # dim_value/dim_param. Setting dim_value automatically
                # clears dim_param; an explicit dim_param assignment
                # would clear the value we just set.
                d.dim_value = int(val)
            found = True
            break
        if not found:
            print(f"[ERROR] Input '{input_name}' not found in graph "
                  f"(available: {[i.name for i in model.graph.input]})")
            sys.exit(1)
        print(f"[INFO] Rewrote input '{input_name}' dims to {list(input_shape)} "
              f"(onnxsim disabled)")
        out = model

    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(out, str(dst_onnx))
    print(f"[INFO] Wrote pinned ONNX: {dst_onnx}")


def compile_dfp(onnx_path: Path, dfp_stem: Path,
                input_shape: list[int] | None,
                num_chips: int, effort: str, autocrop: bool, verbose: int) -> Path:
    """Run the MemryX NeuralCompiler and return the produced .dfp path."""
    # NeuralCompiler writes <dfp_fname>.dfp next to the cwd unless dfp_fname is
    # an absolute path. Pass the absolute stem to land the artifact deterministically.
    dfp_stem = dfp_stem.resolve()

    print(f"[INFO] Compiling DFP: {onnx_path.name} -> {dfp_stem}.dfp")
    print(f"[INFO]   num_chips={num_chips}  effort={effort}  autocrop={autocrop}  "
          f"input_shape={input_shape if input_shape else '(read from ONNX)'}")

    nc_kwargs = dict(
        models=str(onnx_path),
        num_chips=num_chips,
        effort=effort,
        dfp_fname=str(dfp_stem),
        verbose=verbose,
        autocrop=autocrop,
    )
    if input_shape:
        # SDK 2.2 expects a list of int lists in the model's native rank/order
        # (e.g. NCHW for a PyTorch-exported ONNX). The legacy "H,W,C" string
        # format from the ~2023.1 SDK is rejected.
        nc_kwargs["input_shapes"] = [list(input_shape)]
    nc = NeuralCompiler(**nc_kwargs)
    nc.run()

    dfp_path = Path(f"{dfp_stem}.dfp")
    if not dfp_path.exists():
        print(f"[ERROR] Compiler exited but no DFP was written at {dfp_path}")
        sys.exit(1)

    print(f"[INFO] DFP ready: {dfp_path} ({dfp_path.stat().st_size / 1e6:.2f} MB)")
    return dfp_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True, type=Path,
                        help="Path to the FP32 ONNX model")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Output directory for the .dfp artifact")
    parser.add_argument("--name", default=None,
                        help="DFP stem (default: derived from --onnx basename)")
    parser.add_argument("--input-name", default="input",
                        help="ONNX graph input tensor name (default: input)")
    parser.add_argument("--input-shape", default=None,
                        help="Optional comma-separated input shape in the ONNX's native order "
                             "(e.g. '1,3,320,640' for NCHW). Omit to let NeuralCompiler read "
                             "it from the pinned ONNX directly (recommended).")
    parser.add_argument("--batch-shape", default="1,3,320,640",
                        help="NCHW shape to pin via onnxsim (default: 1,3,320,640)")
    parser.add_argument("--no-onnxsim", action="store_true",
                        help="Skip onnxsim simplification; just rewrite the input dim in place. "
                             "Use when the ONNX trips opset checker errors (e.g. older PyTorch "
                             "emitting opset-18 ReduceMean with the legacy attribute form).")
    parser.add_argument("--num-chips", type=int, default=4,
                        help="MX3 M.2 has 4 chips (default: 4)")
    parser.add_argument("--effort", default="Hard",
                        choices=["Lazy", "normal", "Hard"],
                        help="NeuralCompiler effort (default: Hard for deploy DFPs)")
    parser.add_argument("--autocrop", action="store_true",
                        help="Allow the compiler to crop unmappable head/tail ops to host "
                             "(produces a companion post_model_*.onnx that must be loaded via "
                             "MxAccl::connect_post_model). Required for some larger models "
                             "(e.g. DeepLabV3+ that exhausts the 4-chip budget).")
    parser.add_argument("--verbose", type=int, default=1,
                        help="NeuralCompiler verbosity 0-2 (default: 1)")
    parser.add_argument("--keep-intermediate", action="store_true",
                        help="Keep the simplified ONNX alongside the DFP")
    args = parser.parse_args()

    if not args.onnx.exists():
        print(f"[ERROR] ONNX not found: {args.onnx}")
        sys.exit(1)

    name = args.name or args.onnx.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pinned_onnx = args.out_dir / f"{name}_pinned.onnx"
    batch_shape = tuple(int(x) for x in args.batch_shape.split(","))
    pin_batch_axis(args.onnx, pinned_onnx, args.input_name, batch_shape,
                   use_onnxsim=not args.no_onnxsim)

    dfp_stem = args.out_dir / name
    input_shape = ([int(x) for x in args.input_shape.split(",")]
                   if args.input_shape else None)
    compile_dfp(pinned_onnx, dfp_stem, input_shape,
                args.num_chips, args.effort, args.autocrop, args.verbose)

    if not args.keep_intermediate:
        pinned_onnx.unlink(missing_ok=True)
        print(f"[INFO] Removed intermediate: {pinned_onnx}")
    else:
        print(f"[INFO] Kept intermediate: {pinned_onnx}")

    # NeuralCompiler also drops a post_model_*.onnx in the cwd for debug —
    # move it into out_dir so the artifacts stay together.
    for stray in Path.cwd().glob(f"post_model_{name}*.onnx"):
        dst = args.out_dir / stray.name
        shutil.move(str(stray), str(dst))
        print(f"[INFO] Moved compiler artifact: {dst}")


if __name__ == "__main__":
    main()
