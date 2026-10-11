from datetime import datetime


def make_markdown_report(site_name, threshold, changed_pixels, total_pixels,
                         changed_percentage, notes="Synthetic or user-provided images",
                         metrics=None):
    metric_lines = ""
    if metrics:
        metric_lines = "\n".join(f"- {key}: {value}" for key, value in metrics.items())
    return f"""# GeoBuild AI — Analysis Report

- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- Site: {site_name}
- Method: grayscale absolute pixel difference + threshold
- Threshold: {threshold}
- Flagged pixels: {changed_pixels:,}
- Total pixels: {total_pixels:,}
- Flagged image pixels: {changed_percentage:.2f}%
- Input: {notes}

## Evaluation metrics
{metric_lines if metric_lines else "No labelled ground-truth evaluation supplied."}

## Interpretation and limitations

Flagged image-pixel share is not construction completion or real-world area.
Results are sensitive to alignment, shadows, illumination, clouds, seasonality and sensor differences.
"""
