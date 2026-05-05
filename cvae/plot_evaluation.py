import argparse
import csv
import html
from pathlib import Path


CHANNELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def read_csv_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sample_rows(rows, max_points: int):
    if len(rows) <= max_points:
        return rows
    if max_points <= 1:
        return [rows[0]]
    idxs = [round(i * (len(rows) - 1) / (max_points - 1)) for i in range(max_points)]
    return [rows[i] for i in idxs]


def svg_header(width: int, height: int, title: str):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<title>{html.escape(title)}</title>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fffdf8" />',
    ]


def svg_footer():
    return ["</svg>"]


def scale_linear(value, src_min, src_max, dst_min, dst_max):
    if src_max <= src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def polyline_points(values, x0, y0, width, height, vmin, vmax):
    points = []
    n = len(values)
    for i, v in enumerate(values):
        x = x0 if n == 1 else x0 + (i / (n - 1)) * width
        y = scale_linear(v, vmin, vmax, y0 + height, y0)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_text(lines, x, y, text, size=14, color="#111827", anchor="start", weight="normal"):
    lines.append(
        f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" font-size="{size}" '
        f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{html.escape(text)}</text>'
    )


def draw_axes(lines, x, y, w, h, label):
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#d1d5db" stroke-width="1"/>')
    for frac in (0.25, 0.5, 0.75):
        yy = y + h * frac
        lines.append(f'<line x1="{x}" y1="{yy}" x2="{x + w}" y2="{yy}" stroke="#e5e7eb" stroke-width="1"/>')
    write_text(lines, x + 6, y + 18, label, size=14, weight="bold")


def make_time_series_svg(rows, output_path: Path, max_points: int):
    sampled = sample_rows(rows, max_points)
    width = 1500
    panel_h = 220
    margin = 70
    height = margin + panel_h * len(CHANNELS) + 40
    lines = svg_header(width, height, "Open-Loop CVAE vs Collected Data Time Series")

    write_text(lines, width / 2, 34, "Open-Loop CVAE vs Collected Data: Time-Series Comparison", size=24, anchor="middle", weight="bold")

    for idx, channel in enumerate(CHANNELS):
        x = 90
        y = margin + idx * panel_h
        w = width - 150
        h = 150
        draw_axes(lines, x, y, w, h, channel)

        target = [float(r[f"target_{channel}"]) for r in sampled]
        posterior = [float(r[f"posterior_mean_{channel}"]) for r in sampled]
        prior = [float(r[f"prior_mean_{channel}"]) for r in sampled]
        prior_std = [float(r[f"prior_std_{channel}"]) for r in sampled]

        low = min(min(target), min(posterior), min(prior), min(p - s for p, s in zip(prior, prior_std)))
        high = max(max(target), max(posterior), max(prior), max(p + s for p, s in zip(prior, prior_std)))
        if high <= low:
            high = low + 1.0

        upper = [p + s for p, s in zip(prior, prior_std)]
        lower = [p - s for p, s in zip(prior, prior_std)]
        upper_pts = []
        lower_pts = []
        n = len(sampled)
        for i in range(n):
            xx = x if n == 1 else x + (i / (n - 1)) * w
            upper_y = scale_linear(upper[i], low, high, y + h, y)
            lower_y = scale_linear(lower[i], low, high, y + h, y)
            upper_pts.append(f"{xx:.2f},{upper_y:.2f}")
            lower_pts.append(f"{xx:.2f},{lower_y:.2f}")
        band_pts = " ".join(upper_pts + list(reversed(lower_pts)))
        lines.append(f'<polygon points="{band_pts}" fill="#fecaca" opacity="0.45"/>')

        lines.append(
            f'<polyline points="{polyline_points(target, x, y, w, h, low, high)}" '
            f'fill="none" stroke="#111827" stroke-width="1.5"/>'
        )
        lines.append(
            f'<polyline points="{polyline_points(posterior, x, y, w, h, low, high)}" '
            f'fill="none" stroke="#2563eb" stroke-width="1.2" opacity="0.9"/>'
        )
        lines.append(
            f'<polyline points="{polyline_points(prior, x, y, w, h, low, high)}" '
            f'fill="none" stroke="#dc2626" stroke-width="1.2" opacity="0.9"/>'
        )

        write_text(lines, x + w + 8, y + 14, f"min {low:.3f}", size=11, color="#6b7280")
        write_text(lines, x + w + 8, y + h, f"max {high:.3f}", size=11, color="#6b7280")

    legend_y = height - 18
    lines.append('<line x1="120" y1="{0}" x2="155" y2="{0}" stroke="#111827" stroke-width="2"/>'.format(legend_y))
    write_text(lines, 165, legend_y + 4, "target", size=12)
    lines.append('<line x1="260" y1="{0}" x2="295" y2="{0}" stroke="#2563eb" stroke-width="2"/>'.format(legend_y))
    write_text(lines, 305, legend_y + 4, "posterior mean", size=12)
    lines.append('<line x1="450" y1="{0}" x2="485" y2="{0}" stroke="#dc2626" stroke-width="2"/>'.format(legend_y))
    write_text(lines, 495, legend_y + 4, "prior mean", size=12)
    lines.append(f'<rect x="620" y="{legend_y - 8}" width="28" height="12" fill="#fecaca" opacity="0.45"/>')
    write_text(lines, 658, legend_y + 4, "prior std band", size=12)

    lines.extend(svg_footer())
    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_scatter_svg(rows, output_path: Path, max_points: int):
    sampled = sample_rows(rows, max_points)
    width = 1500
    height = 980
    lines = svg_header(width, height, "Open-Loop CVAE Prior Mean vs Target")
    write_text(lines, width / 2, 34, "Open-Loop CVAE Prior Mean vs Target", size=24, anchor="middle", weight="bold")

    cols = 3
    panel_w = 420
    panel_h = 360
    x_gap = 40
    y_gap = 50
    left_margin = 70
    top_margin = 70

    for i, channel in enumerate(CHANNELS):
        row = i // cols
        col = i % cols
        x = left_margin + col * (panel_w + x_gap)
        y = top_margin + row * (panel_h + y_gap)
        w = panel_w
        h = panel_h
        draw_axes(lines, x, y, w, h, channel)

        target = [float(r[f"target_{channel}"]) for r in sampled]
        prior = [float(r[f"prior_mean_{channel}"]) for r in sampled]
        lo = min(min(target), min(prior))
        hi = max(max(target), max(prior))
        if hi <= lo:
            hi = lo + 1.0

        for t, p in zip(target, prior):
            px = scale_linear(t, lo, hi, x, x + w)
            py = scale_linear(p, lo, hi, y + h, y)
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.6" fill="#dc2626" opacity="0.22"/>')

        x1 = scale_linear(lo, lo, hi, x, x + w)
        y1 = scale_linear(lo, lo, hi, y + h, y)
        x2 = scale_linear(hi, lo, hi, x, x + w)
        y2 = scale_linear(hi, lo, hi, y + h, y)
        lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#111827" stroke-width="1.2" stroke-dasharray="6 4"/>')

    lines.extend(svg_footer())
    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_tz_focus_svg(rows, output_path: Path, max_points: int):
    sampled = sample_rows(rows, max_points)
    width = 1500
    height = 900
    lines = svg_header(width, height, "Open-Loop CVAE Tz Focus")
    write_text(lines, width / 2, 34, "Open-Loop CVAE Tz Focus", size=24, anchor="middle", weight="bold")

    x = 90
    y = 80
    w = width - 180
    h = 280
    draw_axes(lines, x, y, w, h, "Tz time series")

    channel = "Tz"
    target = [float(r[f"target_{channel}"]) for r in sampled]
    posterior = [float(r[f"posterior_mean_{channel}"]) for r in sampled]
    prior = [float(r[f"prior_mean_{channel}"]) for r in sampled]
    prior_std = [float(r[f"prior_std_{channel}"]) for r in sampled]

    low = min(min(target), min(posterior), min(prior), min(p - s for p, s in zip(prior, prior_std)))
    high = max(max(target), max(posterior), max(prior), max(p + s for p, s in zip(prior, prior_std)))
    if high <= low:
        high = low + 1.0

    upper = [p + s for p, s in zip(prior, prior_std)]
    lower = [p - s for p, s in zip(prior, prior_std)]
    upper_pts = []
    lower_pts = []
    n = len(sampled)
    for i in range(n):
        xx = x if n == 1 else x + (i / (n - 1)) * w
        upper_y = scale_linear(upper[i], low, high, y + h, y)
        lower_y = scale_linear(lower[i], low, high, y + h, y)
        upper_pts.append(f"{xx:.2f},{upper_y:.2f}")
        lower_pts.append(f"{xx:.2f},{lower_y:.2f}")
    lines.append(f'<polygon points="{" ".join(upper_pts + list(reversed(lower_pts)))}" fill="#fecaca" opacity="0.45"/>')
    lines.append(f'<polyline points="{polyline_points(target, x, y, w, h, low, high)}" fill="none" stroke="#111827" stroke-width="1.7"/>')
    lines.append(f'<polyline points="{polyline_points(posterior, x, y, w, h, low, high)}" fill="none" stroke="#2563eb" stroke-width="1.3"/>')
    lines.append(f'<polyline points="{polyline_points(prior, x, y, w, h, low, high)}" fill="none" stroke="#dc2626" stroke-width="1.3"/>')

    sx = 90
    sy = 450
    sw = width - 180
    sh = 320
    draw_axes(lines, sx, sy, sw, sh, "Tz prior mean vs target")
    scatter_rows = sample_rows(rows, min(20000, len(rows)))
    target_scatter = [float(r[f"target_{channel}"]) for r in scatter_rows]
    prior_scatter = [float(r[f"prior_mean_{channel}"]) for r in scatter_rows]
    lo = min(min(target_scatter), min(prior_scatter))
    hi = max(max(target_scatter), max(prior_scatter))
    if hi <= lo:
        hi = lo + 1.0
    for t, p in zip(target_scatter, prior_scatter):
        px = scale_linear(t, lo, hi, sx, sx + sw)
        py = scale_linear(p, lo, hi, sy + sh, sy)
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.8" fill="#dc2626" opacity="0.22"/>')
    x1 = scale_linear(lo, lo, hi, sx, sx + sw)
    y1 = scale_linear(lo, lo, hi, sy + sh, sy)
    x2 = scale_linear(hi, lo, hi, sx, sx + sw)
    y2 = scale_linear(hi, lo, hi, sy + sh, sy)
    lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#111827" stroke-width="1.2" stroke-dasharray="6 4"/>')

    lines.extend(svg_footer())
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="checkpoints/cvae/evaluation_val/pointwise_comparison.csv",
        help="Path to pointwise_comparison.csv generated by cvae.evaluate",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_timeseries_points", type=int, default=5000)
    parser.add_argument("--max_scatter_points", type=int, default=20000)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(csv_path)

    make_time_series_svg(rows, output_dir / "timeseries_all_channels.svg", args.max_timeseries_points)
    make_scatter_svg(rows, output_dir / "scatter_prior_vs_target.svg", args.max_scatter_points)
    make_tz_focus_svg(rows, output_dir / "tz_focus.svg", args.max_timeseries_points)

    summary = (
        "Generated SVG plots from pointwise_comparison.csv\n"
        "timeseries_all_channels.svg: all channels over time\n"
        "scatter_prior_vs_target.svg: deployable prior mean against target\n"
        "tz_focus.svg: detailed Tz view\n"
    )
    (output_dir / "README.txt").write_text(summary, encoding="utf-8")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
