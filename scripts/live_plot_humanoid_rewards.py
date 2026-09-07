#!/usr/bin/env python3
"""Serve a live 2x4 reward plot for the eight Humanoid decompositions."""

import argparse
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


DECOMPOSITIONS = (
    "1agent",
    "3agents",
    "5agents",
    "7agents",
    "11agents",
    "13agents",
    "15agents",
    "17x1",
)

METRIC_TAGS = {
    "eval": "eval_average_episode_rewards",
    "train": "train_episode_rewards",
}

METHOD_STYLES = {
    "separate": {
        "label": "HAPPO baseline",
        "short_label": "baseline",
        "color": "tab:orange",
    },
    "critic_to_actor": {
        "label": "Critic-to-actor",
        "short_label": "C2A",
        "color": "tab:blue",
    },
}

ALIGNMENT_MODES = (
    "separate",
    "hard_share",
    "critic_to_actor",
    "actor_to_critic",
    "bidirectional",
    "no_stop",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read TensorBoard event files and show all eight Humanoid reward "
            "curves in one live web page."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/humanoid_decomposition_alignment"),
        help="common HARL result root (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--alignment-modes",
        nargs="+",
        choices=ALIGNMENT_MODES,
        default=("separate", "critic_to_actor"),
        help="methods to overlay in every panel (default: separate critic_to_actor)",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=ALIGNMENT_MODES,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--capacity-mode", default="standard")
    parser.add_argument(
        "--metric",
        choices=("auto", "eval", "train"),
        default="eval",
        help="reward source; auto falls back to training return (default: eval)",
    )
    parser.add_argument(
        "--x-axis",
        choices=("steps", "hours"),
        default="steps",
        help="environment steps or elapsed wall-clock hours",
    )
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--refresh-seconds", type=float, default=10.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--independent-y",
        action="store_true",
        help="let each panel use its own reward scale instead of a shared scale",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="also overwrite this PNG whenever the plot refreshes",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="write one PNG and exit (requires --output)",
    )
    args = parser.parse_args()
    if args.refresh_seconds <= 0:
        parser.error("--refresh-seconds must be positive")
    if args.smooth_window <= 0:
        parser.error("--smooth-window must be positive")
    if args.once and args.output is None:
        parser.error("--once requires --output")
    # Backward compatibility with commands issued before multi-method plotting.
    if args.alignment_mode is not None:
        args.alignment_modes = (args.alignment_mode,)
    return args


def find_latest_run(results_dir, decomposition, seed, alignment_mode, capacity_mode):
    """Return the newest matching seed run, using HARL's timestamped directory."""
    experiment = "humanoid_{}_{}_{}".format(
        decomposition, alignment_mode, capacity_mode
    )
    seed_prefix = "seed-{:05d}-".format(seed)
    candidates = []
    for path in results_dir.rglob(seed_prefix + "*"):
        if path.is_dir() and path.parent.name == experiment:
            candidates.append(path)
    if not candidates:
        return None
    # HARL timestamps are lexicographically sortable; mtime breaks equal-name ties.
    return max(candidates, key=lambda path: (path.name, path.stat().st_mtime))


def _matching_tag(tags, target):
    if target in tags:
        return target
    matches = [tag for tag in tags if target in tag]
    if not matches:
        return None
    return min(matches, key=lambda tag: (len(tag), tag))


def read_scalar_events(log_dir, target):
    """Read and merge one scalar from TensorBoardX's nested event directories."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is required: python -m pip install tensorboard"
        ) from exc

    event_parents = sorted({
        path.parent for path in log_dir.rglob("events.out.tfevents.*")
    })
    matching_parents = [parent for parent in event_parents if target in parent.name]
    if matching_parents:
        event_parents = matching_parents
    merged = {}
    errors = []
    for parent in event_parents:
        try:
            accumulator = event_accumulator.EventAccumulator(
                str(parent),
                size_guidance={event_accumulator.SCALARS: 0},
            )
            accumulator.Reload()
            tag = _matching_tag(accumulator.Tags().get("scalars", []), target)
            if tag is None:
                continue
            for event in accumulator.Scalars(tag):
                # Keep the newest event if a writer emitted a step more than once.
                previous = merged.get(event.step)
                point = (float(event.wall_time), float(event.value))
                if previous is None or point[0] >= previous[0]:
                    merged[int(event.step)] = point
        except Exception as exc:  # A writer may be appending while we reload.
            errors.append("{}: {}".format(parent, exc))
    points = [
        (step, wall_time, value)
        for step, (wall_time, value) in sorted(merged.items())
    ]
    return points, errors


def read_eval_progress(path):
    """Read HARL's flushed ``step,reward`` evaluation progress file."""
    if not path.exists():
        return [], []
    points = {}
    errors = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], ["{}: {}".format(path, exc)]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            step_text, reward_text = line.split(",", 1)
            step = int(step_text)
            reward = float(reward_text)
            points[step] = (path.stat().st_mtime, reward)
        except (OSError, TypeError, ValueError) as exc:
            # Ignore only a final partial line while the trainer is appending.
            if line_number != len(lines):
                errors.append("{}:{}: {}".format(path, line_number, exc))
    return [
        (step, wall_time, reward)
        for step, (wall_time, reward) in sorted(points.items())
    ], errors


def load_decomposition(args, decomposition, alignment_mode):
    run_dir = find_latest_run(
        args.results_dir,
        decomposition,
        args.seed,
        alignment_mode,
        args.capacity_mode,
    )
    if run_dir is None:
        return {
            "decomposition": decomposition,
            "alignment_mode": alignment_mode,
            "run_dir": None,
            "metric": None,
            "points": [],
            "errors": [],
        }

    metrics = ("eval", "train") if args.metric == "auto" else (args.metric,)
    all_errors = []
    selected_metric = metrics[0]
    points = []
    for metric in metrics:
        selected_metric = metric
        if metric == "eval" and args.x_axis == "steps":
            points, errors = read_eval_progress(run_dir / "progress.txt")
        else:
            points, errors = read_scalar_events(
                run_dir / "logs", METRIC_TAGS[metric]
            )
        all_errors.extend(errors)
        if points:
            break
    return {
        "decomposition": decomposition,
        "alignment_mode": alignment_mode,
        "run_dir": run_dir,
        "metric": selected_metric,
        "points": points,
        "errors": all_errors,
    }


def moving_average(values, window):
    if window <= 1 or len(values) < 2:
        return list(values)
    result = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def method_style(alignment_mode, index):
    if alignment_mode in METHOD_STYLES:
        return METHOD_STYLES[alignment_mode]
    colors = ("tab:green", "tab:red", "tab:purple", "tab:brown")
    label = alignment_mode.replace("_", " ")
    return {
        "label": label,
        "short_label": label,
        "color": colors[index % len(colors)],
    }


def render_plot(args):
    # Keep the monitor usable on headless training servers.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {
        decomposition: [
            load_decomposition(args, decomposition, alignment_mode)
            for alignment_mode in args.alignment_modes
        ]
        for decomposition in DECOMPOSITIONS
    }
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(18, 8.5),
        sharey=not args.independent_y,
        constrained_layout=True,
    )
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    figure.suptitle(
        "Humanoid reward monitor | seed={} | {} | updated {}".format(
            args.seed, args.capacity_mode, now
        ),
        fontsize=14,
    )

    from matplotlib.lines import Line2D

    legend_handles = []
    for mode_index, alignment_mode in enumerate(args.alignment_modes):
        style = method_style(alignment_mode, mode_index)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=2.4,
                label=style["label"],
            )
        )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=max(1, len(legend_handles)),
        frameon=False,
    )

    for index, (axis, decomposition) in enumerate(zip(axes.flat, DECOMPOSITIONS)):
        items = series[decomposition]
        axis.grid(True, alpha=0.25, linewidth=0.7)
        axis.set_title(decomposition, fontweight="bold")
        status_lines = []
        has_points = False

        for mode_index, item in enumerate(items):
            alignment_mode = item["alignment_mode"]
            style = method_style(alignment_mode, mode_index)
            points = item["points"]
            metric = item["metric"]
            if not points:
                if item["run_dir"] is None:
                    state = "run not found"
                else:
                    state = "waiting for {}".format(metric)
                status_lines.append("{}: {}".format(style["short_label"], state))
                continue

            has_points = True
            steps = [point[0] for point in points]
            wall_times = [point[1] for point in points]
            rewards = [point[2] for point in points]
            if args.x_axis == "hours":
                first_wall_time = wall_times[0]
                x_values = [
                    (value - first_wall_time) / 3600.0 for value in wall_times
                ]
            else:
                x_values = [step / 1_000_000.0 for step in steps]

            axis.plot(
                x_values,
                rewards,
                color=style["color"],
                alpha=0.20,
                linewidth=1.0,
            )
            smoothed = moving_average(rewards, args.smooth_window)
            axis.plot(
                x_values,
                smoothed,
                color=style["color"],
                linewidth=2.2,
            )
            axis.scatter(
                [x_values[-1]],
                [rewards[-1]],
                color=style["color"],
                s=18,
                zorder=3,
            )
            metric_label = "eval" if metric == "eval" else "train"
            status_lines.append(
                "{}: {} n={} last={:.1f} step={:,}".format(
                    style["short_label"],
                    metric_label,
                    len(points),
                    rewards[-1],
                    steps[-1],
                )
            )

        if not has_points:
            axis.text(
                0.5,
                0.5,
                "\n".join(status_lines),
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="0.45",
            )
            axis.set_xticks([])
            if args.independent_y:
                axis.set_yticks([])
        else:
            axis.text(
                0.02,
                0.98,
                "\n".join(status_lines),
                ha="left",
                va="top",
                transform=axis.transAxes,
                fontsize=8.2,
            )
        if index % 4 == 0:
            axis.set_ylabel("Episode return")
        if index >= 4:
            axis.set_xlabel(
                "Elapsed hours"
                if args.x_axis == "hours"
                else "Environment steps (millions)"
            )

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    png = buffer.getvalue()
    status = {
        "updated": now,
        "runs": [
            {
                "decomposition": item["decomposition"],
                "alignment_mode": item["alignment_mode"],
                "run_dir": str(item["run_dir"]) if item["run_dir"] else None,
                "metric": item["metric"],
                "points": len(item["points"]),
                "errors": item["errors"],
            }
            for decomposition in DECOMPOSITIONS
            for item in series[decomposition]
        ],
    }
    return png, status


def write_output(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(str(temporary), str(path))


class PlotCache:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.rendered_at = 0.0
        self.png = b""
        self.status = {}

    def get(self):
        with self.lock:
            if (
                not self.png
                or time.monotonic() - self.rendered_at >= self.args.refresh_seconds
            ):
                self.png, self.status = render_plot(self.args)
                self.rendered_at = time.monotonic()
                if self.args.output is not None:
                    write_output(self.args.output, self.png)
            return self.png, self.status


def make_handler(cache):
    refresh_ms = max(1000, int(cache.args.refresh_seconds * 1000))
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Humanoid reward monitor</title>
<style>
body { margin: 0; background: #f4f5f7; font-family: sans-serif; }
main { padding: 12px; text-align: center; }
img { max-width: 100%; height: auto; background: white; }
#status { color: #555; font-size: 13px; margin: 5px; }
</style></head><body><main>
<div id="status">Loading…</div><img id="plot" alt="Eight Humanoid reward plots">
<script>
const plot = document.getElementById('plot');
const status = document.getElementById('status');
function refresh() {
  const started = Date.now();
  plot.onload = () => { status.textContent = 'Live · refreshed ' + new Date().toLocaleTimeString(); };
  plot.onerror = () => { status.textContent = 'Refresh failed; retrying…'; };
  plot.src = '/plot.png?t=' + started;
}
refresh(); setInterval(refresh, REFRESH_MS);
</script></main></body></html>""".replace("REFRESH_MS", str(refresh_ms))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path).path
            try:
                if route == "/plot.png":
                    png, _ = cache.get()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(png)))
                    self.end_headers()
                    self.wfile.write(png)
                elif route == "/status.json":
                    _, status_data = cache.get()
                    payload = json.dumps(status_data, indent=2).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif route in ("/", "/index.html"):
                    payload = html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_error(404)
            except Exception as exc:
                self.send_error(500, str(exc))

        def log_message(self, message, *values):
            return

    return Handler


def main():
    args = parse_args()
    args.results_dir = args.results_dir.expanduser().resolve()
    if args.output is not None:
        args.output = args.output.expanduser().resolve()

    cache = PlotCache(args)
    if args.once:
        png, status = cache.get()
        print("wrote {}".format(args.output))
        print(json.dumps(status, indent=2))
        return

    server = ThreadingHTTPServer((args.host, args.port), make_handler(cache))
    print("Reading results from {}".format(args.results_dir))
    print("Open http://{}:{}".format(args.host, args.port))
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
