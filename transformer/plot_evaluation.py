import argparse
import csv
import html
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - only needed for PNG/JPG output.
    Image = None
    ImageDraw = None
    ImageFont = None


CHANNELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
CHANNEL_TITLES = {
    "Fx": "Fff_fx",
    "Fy": "Fff_fy",
    "Fz": "Fff_fz",
    "Tx": "Fff_tx",
    "Ty": "Fff_ty",
    "Tz": "Fff_tz",
}


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sample_rows(rows: list[dict[str, str]], max_points: int) -> list[dict[str, str]]:
    if len(rows) <= max_points:
        return rows
    if max_points <= 1:
        return [rows[0]]
    idxs = [round(i * (len(rows) - 1) / (max_points - 1)) for i in range(max_points)]
    return [rows[i] for i in idxs]


def window_rows(rows: list[dict[str, str]], start_row: int, num_rows: int) -> list[dict[str, str]]:
    start = max(0, start_row)
    if num_rows <= 0:
        return rows[start:]
    return rows[start:start + num_rows]


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
    ]


def svg_footer() -> list[str]:
    return ["</svg>"]


def write_text(
    lines: list[str],
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    color: str = "#111827",
    anchor: str = "start",
    weight: str = "normal",
) -> None:
    lines.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">'
        f"{html.escape(text)}</text>"
    )


def scale_linear(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if vmax <= vmin:
        return [vmin]
    return [vmin + (i / (count - 1)) * (vmax - vmin) for i in range(count)]


def output_format(output_path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    suffix = output_path.suffix.lower().lstrip(".")
    if suffix in {"png", "jpg", "jpeg", "svg"}:
        return suffix
    return "svg"


def polyline_points(times: list[float], values: list[float], x: float, y: float, w: float, h: float, vmin: float, vmax: float) -> str:
    tmin = times[0]
    tmax = times[-1]
    pts = []
    for t, value in zip(times, values):
        px = scale_linear(t, tmin, tmax, x, x + w)
        py = scale_linear(value, vmin, vmax, y + h, y)
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def draw_panel(
    lines: list[str],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    times: list[float],
    target: list[float],
    pred: list[float],
    show_x_ticks: bool,
    show_legend: bool,
) -> None:
    ymin = min(min(target), min(pred))
    ymax = max(max(target), max(pred))
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="none" stroke="#111827" stroke-width="1"/>')
    write_text(lines, x + w / 2, y - 10, title, size=15, anchor="middle")

    for tick in nice_ticks(ymin, ymax, 5):
        tick_y = scale_linear(tick, ymin, ymax, y + h, y)
        lines.append(f'<line x1="{x:.2f}" y1="{tick_y:.2f}" x2="{x + w:.2f}" y2="{tick_y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        write_text(lines, x - 10, tick_y + 4, f"{tick:.2f}", size=11, color="#374151", anchor="end")

    tmin = times[0]
    tmax = times[-1]
    for tick in nice_ticks(tmin, tmax, 6):
        tick_x = scale_linear(tick, tmin, tmax, x, x + w)
        lines.append(f'<line x1="{tick_x:.2f}" y1="{y:.2f}" x2="{tick_x:.2f}" y2="{y + h:.2f}" stroke="#f3f4f6" stroke-width="1"/>')
        if show_x_ticks:
            write_text(lines, tick_x, y + h + 22, f"{tick:.1f}", size=11, color="#374151", anchor="middle")

    lines.append(
        f'<polyline points="{polyline_points(times, target, x, y, w, h, ymin, ymax)}" '
        f'fill="none" stroke="#1f77b4" stroke-width="1.5" opacity="0.9"/>'
    )
    lines.append(
        f'<polyline points="{polyline_points(times, pred, x, y, w, h, ymin, ymax)}" '
        f'fill="none" stroke="#ff7f0e" stroke-width="1.5" opacity="0.9"/>'
    )

    write_text(lines, x - 58, y + h / 2, "[N] or [Nm]", size=12, anchor="middle")

    if show_legend:
        legend_x = x + w - 120
        legend_y = y + 12
        lines.append(f'<rect x="{legend_x:.2f}" y="{legend_y:.2f}" width="104" height="42" fill="#ffffff" stroke="#d1d5db" stroke-width="0.8"/>')
        lines.append(f'<line x1="{legend_x + 10:.2f}" y1="{legend_y + 14:.2f}" x2="{legend_x + 34:.2f}" y2="{legend_y + 14:.2f}" stroke="#1f77b4" stroke-width="1.5"/>')
        write_text(lines, legend_x + 40, legend_y + 18, "Ground truth", size=10)
        lines.append(f'<line x1="{legend_x + 10:.2f}" y1="{legend_y + 31:.2f}" x2="{legend_x + 34:.2f}" y2="{legend_y + 31:.2f}" stroke="#ff7f0e" stroke-width="1.5"/>')
        write_text(lines, legend_x + 40, legend_y + 35, "Predicted", size=10)


def make_open_loop_grid(
    csv_path: Path,
    output_path: Path,
    *,
    title: str,
    dt: float,
    max_points: int,
    start_row: int,
    num_rows: int,
) -> None:
    rows = sample_rows(window_rows(read_csv_rows(csv_path), start_row, num_rows), max_points)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")

    times = [float(r["timestep"]) * dt for r in rows]
    width = 1800
    height = 760
    left = 118
    top = 92
    panel_w = 450
    panel_h = 220
    gap_x = 80
    gap_y = 64

    lines = svg_header(width, height, title)
    write_text(lines, width / 2, 28, title, size=18, anchor="middle")

    for idx, channel in enumerate(CHANNELS):
        row = idx // 3
        col = idx % 3
        x = left + col * (panel_w + gap_x)
        y = top + row * (panel_h + gap_y)
        target = [float(r[f"target_{channel}"]) for r in rows]
        pred = [float(r[f"pred_{channel}"]) for r in rows]
        draw_panel(
            lines,
            x=x,
            y=y,
            w=panel_w,
            h=panel_h,
            title=CHANNEL_TITLES[channel],
            times=times,
            target=target,
            pred=pred,
            show_x_ticks=row == 1,
            show_legend=idx == 0,
        )

    write_text(lines, width / 2, height - 34, "Time [s]", size=14, anchor="middle")
    lines.extend(svg_footer())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def load_font(size: int, bold: bool = False):
    if ImageFont is None:
        return None
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_raster_text(
    draw,
    xy: tuple[float, float],
    text: str,
    *,
    size: int,
    fill: tuple[int, int, int] = (17, 24, 39),
    anchor: str = "la",
    bold: bool = False,
) -> None:
    draw.text(xy, text, fill=fill, font=load_font(size, bold=bold), anchor=anchor)


def draw_rotated_text(
    image,
    xy: tuple[int, int],
    text: str,
    *,
    size: int,
    fill: tuple[int, int, int] = (17, 24, 39),
) -> None:
    font = load_font(size)
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0] + 8
    th = bbox[3] - bbox[1] + 8
    label = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((4, 4), text, fill=fill, font=font)
    image.alpha_composite(label.rotate(90, expand=True), xy)


def draw_raster_panel(
    draw,
    image,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    times: list[float],
    target: list[float],
    pred: list[float],
    show_x_ticks: bool,
    show_legend: bool,
) -> None:
    ymin = min(min(target), min(pred))
    ymax = max(max(target), max(pred))
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    draw.rectangle((x, y, x + w, y + h), outline=(17, 24, 39), width=1)
    draw_raster_text(draw, (x + w / 2, y - 24), title, size=17, anchor="ma")

    for tick in nice_ticks(ymin, ymax, 5):
        tick_y = scale_linear(tick, ymin, ymax, y + h, y)
        draw.line((x, tick_y, x + w, tick_y), fill=(229, 231, 235), width=1)
        draw_raster_text(draw, (x - 10, tick_y - 7), f"{tick:.2f}", size=12, fill=(55, 65, 81), anchor="ra")

    tmin = times[0]
    tmax = times[-1]
    for tick in nice_ticks(tmin, tmax, 6):
        tick_x = scale_linear(tick, tmin, tmax, x, x + w)
        draw.line((tick_x, y, tick_x, y + h), fill=(243, 244, 246), width=1)
        if show_x_ticks:
            draw_raster_text(draw, (tick_x, y + h + 10), f"{tick:.1f}", size=12, fill=(55, 65, 81), anchor="ma")

    def points(values: list[float]) -> list[tuple[float, float]]:
        return [
            (
                scale_linear(t, tmin, tmax, x, x + w),
                scale_linear(v, ymin, ymax, y + h, y),
            )
            for t, v in zip(times, values)
        ]

    draw.line(points(target), fill=(31, 119, 180), width=2, joint="curve")
    draw.line(points(pred), fill=(255, 127, 14), width=2, joint="curve")
    draw_rotated_text(image, (x - 72, y + h // 2 - 46), "[N] or [Nm]", size=13)

    if show_legend:
        legend_x = x + w - 138
        legend_y = y + 12
        draw.rectangle((legend_x, legend_y, legend_x + 122, legend_y + 46), fill=(255, 255, 255), outline=(209, 213, 219), width=1)
        draw.line((legend_x + 10, legend_y + 15, legend_x + 38, legend_y + 15), fill=(31, 119, 180), width=2)
        draw_raster_text(draw, (legend_x + 45, legend_y + 7), "Ground truth", size=11)
        draw.line((legend_x + 10, legend_y + 33, legend_x + 38, legend_y + 33), fill=(255, 127, 14), width=2)
        draw_raster_text(draw, (legend_x + 45, legend_y + 25), "Predicted", size=11)


def make_open_loop_grid_raster(
    csv_path: Path,
    output_path: Path,
    *,
    title: str,
    dt: float,
    max_points: int,
    start_row: int,
    num_rows: int,
    image_format: str,
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for PNG/JPG output. Install pillow or write SVG instead.")

    rows = sample_rows(window_rows(read_csv_rows(csv_path), start_row, num_rows), max_points)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")

    times = [float(r["timestep"]) * dt for r in rows]
    width = 1800
    height = 760
    left = 118
    top = 92
    panel_w = 450
    panel_h = 220
    gap_x = 80
    gap_y = 64

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw_raster_text(draw, (width / 2, 16), title, size=18, anchor="ma")

    for idx, channel in enumerate(CHANNELS):
        row = idx // 3
        col = idx % 3
        x = left + col * (panel_w + gap_x)
        y = top + row * (panel_h + gap_y)
        target = [float(r[f"target_{channel}"]) for r in rows]
        pred = [float(r[f"pred_{channel}"]) for r in rows]
        draw_raster_panel(
            draw,
            image,
            x=x,
            y=y,
            w=panel_w,
            h=panel_h,
            title=CHANNEL_TITLES[channel],
            times=times,
            target=target,
            pred=pred,
            show_x_ticks=row == 1,
            show_legend=idx == 0,
        )

    draw_raster_text(draw, (width / 2, height - 42), "Time [s]", size=15, anchor="ma")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_format in {"jpg", "jpeg"}:
        image.convert("RGB").save(output_path, quality=95)
    else:
        image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Transformer open-loop predictions against ground truth.")
    parser.add_argument("--csv", default=None, help="Path to pointwise_comparison.csv.")
    parser.add_argument("--checkpoint_dir", default="checkpoints/transformer_s20_h256_l2_h4")
    parser.add_argument("--evaluation_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--format", choices=["auto", "svg", "png", "jpg", "jpeg"], default="auto")
    parser.add_argument("--dt", type=float, default=0.005, help="Seconds per timestep. Default is 200 Hz data.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=2500, help="Contiguous rows to plot. Use 0 for all rows.")
    parser.add_argument("--max-points", type=int, default=2500)
    parser.add_argument("--title", default="Open-loop validation - val split")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        suffix = "png" if args.format == "png" else "svg"
        default_output = csv_path.with_name(f"open_loop_timeseries_grid.{suffix}")
    else:
        eval_dir = Path(args.evaluation_dir) if args.evaluation_dir else Path(args.checkpoint_dir) / "evaluation_val"
        csv_path = eval_dir / "pointwise_comparison.csv"
        suffix = "png" if args.format == "png" else "svg"
        default_output = eval_dir / f"open_loop_timeseries_grid.{suffix}"

    output_path = Path(args.output) if args.output else default_output
    fmt = output_format(output_path, args.format)
    if fmt == "svg":
        make_open_loop_grid(
            csv_path,
            output_path,
            title=args.title,
            dt=args.dt,
            max_points=args.max_points,
            start_row=args.start_row,
            num_rows=args.num_rows,
        )
    else:
        make_open_loop_grid_raster(
            csv_path,
            output_path,
            title=args.title,
            dt=args.dt,
            max_points=args.max_points,
            start_row=args.start_row,
            num_rows=args.num_rows,
            image_format=fmt,
        )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
