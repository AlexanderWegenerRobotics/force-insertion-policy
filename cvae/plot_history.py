import argparse
import html
import json
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


def scale_linear(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def polyline_points(values: list[float], x0: float, y0: float, width: float, height: float, vmin: float, vmax: float) -> str:
    points = []
    n = len(values)
    for i, value in enumerate(values):
        x = x0 if n <= 1 else x0 + (i / (n - 1)) * width
        y = scale_linear(value, vmin, vmax, y0 + height, y0)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


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
    lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="none" stroke="#9ca3af" stroke-width="1.2"/>')

    for tick_y, label in y_ticks:
        lines.append(f'<line x1="{x:.2f}" y1="{tick_y:.2f}" x2="{x + w:.2f}" y2="{tick_y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        write_text(lines, x - 10, tick_y + 4, label, size=10, anchor="end")

    for tick_x, label in x_ticks:
        lines.append(f'<line x1="{tick_x:.2f}" y1="{y:.2f}" x2="{tick_x:.2f}" y2="{y + h:.2f}" stroke="#f3f4f6" stroke-width="1"/>')
        write_text(lines, tick_x, y + h + 20, label, size=10, anchor="middle")

    write_text(lines, x + w / 2, y - 12, title, size=14, anchor="middle")
    write_text(lines, x + w / 2, y + h + 42, x_label, size=12, anchor="middle")
    write_text(lines, x - 44, y + h / 2, y_label, size=12, anchor="middle")


def draw_legend(lines: list[str], x: float, y: float, color: str, label: str) -> None:
    lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="78" height="22" fill="#ffffff" stroke="#d1d5db" stroke-width="0.8"/>')
    lines.append(f'<line x1="{x + 8:.2f}" y1="{y + 11:.2f}" x2="{x + 26:.2f}" y2="{y + 11:.2f}" stroke="{color}" stroke-width="2"/>')
    write_text(lines, x + 32, y + 15, label, size=10)


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
    color: str = "#4f46e5",
) -> None:
    if not values:
        raise ValueError(f"No values available for panel '{title}'.")

    xmin = epochs[0]
    xmax = epochs[-1]
    ymin = min(values)
    ymax = max(values)
    if ymax <= ymin:
        ymax = ymin + 1e-6
    pad = 0.05 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    x_ticks = []
    for i in range(6):
        frac = i / 5 if 5 else 0.0
        epoch = round(xmin + frac * (xmax - xmin))
        tick_x = scale_linear(epoch, xmin, xmax, x, x + w)
        x_ticks.append((tick_x, str(epoch)))

    y_ticks = []
    for i in range(5):
        frac = i / 4 if 4 else 0.0
        value = ymin + frac * (ymax - ymin)
        tick_y = scale_linear(value, ymin, ymax, y + h, y)
        y_ticks.append((tick_y, f"{value:.4f}"))

    draw_axes(
        lines,
        x=x,
        y=y,
        w=w,
        h=h,
        title=title,
        x_label="Epoch",
        y_label="Loss",
        x_ticks=x_ticks,
        y_ticks=y_ticks,
    )

    pts = []
    for epoch, value in zip(epochs, values):
        px = scale_linear(epoch, xmin, xmax, x, x + w)
        py = scale_linear(value, ymin, ymax, y + h, y)
        pts.append(f"{px:.2f},{py:.2f}")
    lines.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.85"/>')

    draw_legend(lines, x + w - 90, y + 10, color, legend_label)


def plot_both(history: dict, output_path: Path, legend_label: str) -> None:
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

    lines = svg_header(width, height, "Training And Validation Loss")
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
    )
    lines.extend(svg_footer())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_single(history: dict, output_path: Path, metric: str, legend_label: str, title: str) -> None:
    if metric == "train":
        epochs = list(range(1, len(history.get("train_loss", [])) + 1))
        values = history.get("train_loss", [])
    else:
        epochs = history.get("val_epochs", [])
        values = history.get("val_loss", [])
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
    )
    lines.extend(svg_footer())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def default_legend_label(history: dict) -> str:
    hidden_dim = history.get("hidden_dim")
    return f"cvae (N={hidden_dim})" if hidden_dim is not None else "cvae"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CVAE training history using the notebook reference layout.")
    parser.add_argument("--history", default="checkpoints/cvae/history.json")
    parser.add_argument("--output", default="checkpoints/cvae/history_plots.svg")
    parser.add_argument("--metric", choices=["train", "val", "both"], default="both")
    parser.add_argument("--title", default=None)
    parser.add_argument("--legend-label", default=None)
    args = parser.parse_args()

    history = load_history(Path(args.history))
    legend_label = args.legend_label or default_legend_label(history)
    output_path = Path(args.output)

    if args.metric == "both":
        plot_both(history, output_path, legend_label)
    elif args.metric == "train":
        plot_single(history, output_path, "train", legend_label, args.title or "Training Loss")
    else:
        plot_single(history, output_path, "val", legend_label, args.title or "Validation Loss")

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
