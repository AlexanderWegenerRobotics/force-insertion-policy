import argparse
import html
import json
import math
from pathlib import Path


def load_history(history_path: Path) -> dict:
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
    ]


def svg_footer() -> list[str]:
    return ["</svg>"]


def write_text(lines: list[str], x: float, y: float, text: str, size: int = 12, anchor: str = "start") -> None:
    lines.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="#111827" text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def format_loss(value: float) -> str:
    abs_value = abs(value)
    if abs_value != 0.0 and (abs_value < 1e-3 or abs_value >= 100.0):
        return f"{value:.1e}"
    return f"{value:.4f}"


def scale_linear(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def scale_log(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    src_min = max(src_min, 1e-12)
    value = max(value, 1e-12)
    lo = math.log10(src_min)
    hi = math.log10(max(src_max, src_min * 1.01))
    ratio = (math.log10(value) - lo) / (hi - lo)
    return dst_min + ratio * (dst_max - dst_min)


def draw_axes(
    lines: list[str],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    x_label: str,
    y_label: str,
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
) -> None:
    lines.append(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'fill="none" stroke="#9ca3af" stroke-width="1.2"/>'
    )

    for tick_y, label in y_ticks:
        lines.append(
            f'<line x1="{x:.2f}" y1="{tick_y:.2f}" x2="{x + w:.2f}" y2="{tick_y:.2f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        write_text(lines, x - 10, tick_y + 4, label, size=10, anchor="end")

    for tick_x, label in x_ticks:
        lines.append(
            f'<line x1="{tick_x:.2f}" y1="{y:.2f}" x2="{tick_x:.2f}" y2="{y + h:.2f}" '
            f'stroke="#f3f4f6" stroke-width="1"/>'
        )
        write_text(lines, tick_x, y + h + 20, label, size=10, anchor="middle")

    write_text(lines, x + w / 2, y - 12, title, size=14, anchor="middle")
    write_text(lines, x + w / 2, y + h + 42, x_label, size=12, anchor="middle")
    write_text(lines, x - 52, y + h / 2, y_label, size=12, anchor="middle")


def draw_legend(lines: list[str], x: float, y: float, color: str, label: str) -> None:
    box_width = max(112, 16 + len(label) * 6)
    lines.append(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{box_width}" height="22" '
        f'fill="#ffffff" stroke="#d1d5db" stroke-width="0.8"/>'
    )
    lines.append(
        f'<line x1="{x + 8:.2f}" y1="{y + 11:.2f}" x2="{x + 26:.2f}" y2="{y + 11:.2f}" '
        f'stroke="{color}" stroke-width="2"/>'
    )
    write_text(lines, x + 32, y + 15, label, size=10)


def loss_range(values: list[float], y_scale: str) -> tuple[float, float]:
    if y_scale == "log":
        positive = [v for v in values if v > 0.0]
        if not positive:
            raise ValueError("Log scale requires at least one positive loss value.")
        ymin = min(positive)
        ymax = max(positive)
        return max(ymin * 0.8, 1e-12), ymax * 1.2

    ymin = min(values)
    ymax = max(values)
    if ymax <= ymin:
        ymax = ymin + 1e-6
    pad = 0.05 * (ymax - ymin)
    return ymin - pad, ymax + pad


def make_y_ticks(ymin: float, ymax: float, y: float, h: float, y_scale: str) -> list[tuple[float, str]]:
    ticks = []
    if y_scale == "log":
        lo = math.floor(math.log10(max(ymin, 1e-12)))
        hi = math.ceil(math.log10(max(ymax, ymin * 1.01)))
        powers = list(range(lo, hi + 1))
        if len(powers) > 6:
            step = math.ceil(len(powers) / 6)
            powers = powers[::step]
        for power in powers:
            value = 10.0 ** power
            tick_y = scale_log(value, ymin, ymax, y + h, y)
            ticks.append((tick_y, format_loss(value)))
        return ticks

    for i in range(5):
        frac = i / 4 if 4 else 0.0
        value = ymin + frac * (ymax - ymin)
        tick_y = scale_linear(value, ymin, ymax, y + h, y)
        ticks.append((tick_y, format_loss(value)))
    return ticks


def draw_loss_panel(
    lines: list[str],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    epochs: list[int],
    values: list[float],
    legend_label: str,
    color: str = "#2563eb",
    y_scale: str = "linear",
) -> None:
    if not values:
        raise ValueError(f"No values available for panel '{title}'.")

    xmin = epochs[0]
    xmax = epochs[-1]
    ymin, ymax = loss_range(values, y_scale)

    x_ticks = []
    for i in range(6):
        frac = i / 5 if 5 else 0.0
        epoch = round(xmin + frac * (xmax - xmin))
        tick_x = scale_linear(epoch, xmin, xmax, x, x + w)
        x_ticks.append((tick_x, str(epoch)))

    draw_axes(
        lines,
        x=x,
        y=y,
        w=w,
        h=h,
        title=title,
        x_label="Epoch",
        y_label="Loss" if y_scale == "linear" else "Loss (log)",
        x_ticks=x_ticks,
        y_ticks=make_y_ticks(ymin, ymax, y, h, y_scale),
    )

    pts = []
    for epoch, value in zip(epochs, values):
        px = scale_linear(epoch, xmin, xmax, x, x + w)
        if y_scale == "log":
            py = scale_log(value, ymin, ymax, y + h, y)
        else:
            py = scale_linear(value, ymin, ymax, y + h, y)
        pts.append(f"{px:.2f},{py:.2f}")
    lines.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')

    draw_legend(lines, x + w - 150, y + 10, color, legend_label)


def plot_both(history: dict, output_path: Path, legend_label: str, y_scale: str) -> None:
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_epochs = history.get("val_epochs", [])
    if not train_loss:
        raise ValueError("history.json is missing train_loss.")
    if not val_loss or len(val_loss) != len(val_epochs):
        raise ValueError("history.json is missing consistent val_loss/val_epochs.")

    width = 1100
    height = 420
    panel_w = 430
    panel_h = 260
    left_x = 90
    right_x = 590
    panel_y = 70

    lines = svg_header(width, height, "Transformer Training History")
    draw_loss_panel(
        lines,
        x=left_x,
        y=panel_y,
        w=panel_w,
        h=panel_h,
        title="Training Loss",
        epochs=list(range(1, len(train_loss) + 1)),
        values=train_loss,
        legend_label=legend_label,
        y_scale=y_scale,
    )
    draw_loss_panel(
        lines,
        x=right_x,
        y=panel_y,
        w=panel_w,
        h=panel_h,
        title="Validation Loss",
        epochs=val_epochs,
        values=val_loss,
        legend_label=legend_label,
        color="#0f766e",
        y_scale=y_scale,
    )
    lines.extend(svg_footer())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_single(history: dict, output_path: Path, metric: str, legend_label: str, title: str, y_scale: str) -> None:
    if metric == "train":
        epochs = list(range(1, len(history.get("train_loss", [])) + 1))
        values = history.get("train_loss", [])
        color = "#2563eb"
    else:
        epochs = history.get("val_epochs", [])
        values = history.get("val_loss", [])
        color = "#0f766e"
    if not values:
        raise ValueError(f"history.json is missing {metric} loss.")

    width = 560
    height = 420
    lines = svg_header(width, height, title)
    draw_loss_panel(
        lines,
        x=90,
        y=70,
        w=400,
        h=260,
        title=title,
        epochs=epochs,
        values=values,
        legend_label=legend_label,
        color=color,
        y_scale=y_scale,
    )
    lines.extend(svg_footer())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def default_legend_label(history: dict) -> str:
    model_type = history.get("model_type", "transformer")
    seq_len = history.get("seq_len")
    hidden_dim = history.get("hidden_dim")
    num_layers = history.get("num_layers")
    num_heads = history.get("num_heads")

    if model_type in {"transformer", "temporal_transformer"}:
        return f"transformer s{seq_len} h{hidden_dim} l{num_layers} h{num_heads}"
    if hidden_dim is not None:
        return f"{model_type} s{seq_len} h{hidden_dim} l{num_layers}"
    return str(model_type)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        history_path = checkpoint_dir / "history.json"
        output_path = Path(args.output) if args.output else checkpoint_dir / "history_plots.svg"
        return history_path, output_path

    history_path = Path(args.history)
    output_path = Path(args.output) if args.output else history_path.with_name("history_plots.svg")
    return history_path, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Transformer training history.")
    parser.add_argument("--checkpoint_dir", default="checkpoints/transformer_s20_h256_l2_h4")
    parser.add_argument("--history", default="checkpoints/transformer/history.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metric", choices=["train", "val", "both"], default="both")
    parser.add_argument("--y-scale", choices=["linear", "log"], default="log")
    parser.add_argument("--title", default=None)
    parser.add_argument("--legend-label", default=None)
    args = parser.parse_args()

    history_path, output_path = resolve_paths(args)
    history = load_history(history_path)
    legend_label = args.legend_label or default_legend_label(history)

    if args.metric == "both":
        plot_both(history, output_path, legend_label, args.y_scale)
    elif args.metric == "train":
        plot_single(history, output_path, "train", legend_label, args.title or "Training Loss", args.y_scale)
    else:
        plot_single(history, output_path, "val", legend_label, args.title or "Validation Loss", args.y_scale)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
