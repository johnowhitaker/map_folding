import subprocess
import time
from pathlib import Path

import modal


ROOT = Path(__file__).parent

app = modal.App("map-folding-gpu")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential")
    .add_local_dir(ROOT, "/root/map_folding", copy=True)
)


@app.function(image=image, gpu="A10G:4", timeout=24 * 60 * 60)
def run_gpu(
    source: str = "gpu_folding_v2.cu",
    args: str = "6 18 100000 128 8 4",
    executable: str = "gpu_folding_remote",
    nvcc_flags: str = "",
) -> str:
    workdir = Path("/root/map_folding")
    compile_cmd = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-Xcompiler",
        "-pthread",
        *nvcc_flags.split(),
        "-o",
        executable,
        source,
    ]
    run_cmd = [f"./{executable}", *args.split()]

    def run_and_stream(cmd):
        output = []
        print("$ " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            output.append(line)
        returncode = proc.wait()
        return returncode, "".join(output)

    lines = []
    start = time.perf_counter()
    lines.append("$ " + " ".join(compile_cmd))
    compile_returncode, compile_output = run_and_stream(compile_cmd)
    lines.append(compile_output)
    print(f"compile_returncode={compile_returncode}", flush=True)
    lines.append(f"compile_returncode={compile_returncode}")
    if compile_returncode != 0:
        return "\n".join(lines)

    lines.append("$ " + " ".join(run_cmd))
    run_returncode, run_output = run_and_stream(run_cmd)
    lines.append(run_output)
    print(f"run_returncode={run_returncode}", flush=True)
    wall_s = time.perf_counter() - start
    print(f"modal_wall_s={wall_s:.6f}", flush=True)
    lines.append(f"run_returncode={run_returncode}")
    lines.append(f"modal_wall_s={wall_s:.6f}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(
    source: str = "gpu_folding_v2.cu",
    args: str = "6 18 100000 128 8 4",
    executable: str = "gpu_folding_remote",
    nvcc_flags: str = "",
):
    print(run_gpu.remote(source=source, args=args, executable=executable, nvcc_flags=nvcc_flags))
