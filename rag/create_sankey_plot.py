"""Create a Sankey plot for prompt and RAG output classifications."""

from __future__ import annotations

import json
import sys
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

PROJECT_ROOT = Path("/home/alic/RAG")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.analyze_multiple_responses_with_F1 import (  # noqa: E402
    check_answer,
    get_base_code,
    is_code_valid,
    norm_code,
)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "rag" / "results"

INPUT_JSON = ( RESULTS_DIR / "sankey_plots")

OUT_PNG = (
    RESULTS_DIR / "sankey_plots")


# -----------------------------------------------------------------------------
# Patterns
# -----------------------------------------------------------------------------

ALL_CODES_PATTERN = re.compile(
    r"\b[A-Z](?![A-Z])\d{1,2}(?:\.\d{1,2})?\b",
    re.IGNORECASE,
)


def find_all_codes(text: Optional[str]) -> List[str]:
    """
    Find all ICD-like codes in a text.

    Args:
        text (str | None): Input text to search in.

    Returns:
        list[str]: List of found codes in uppercase.
    """
    if not text:
        return []

    return [
        match.group(0).upper()
        for match in ALL_CODES_PATTERN.finditer(text)
    ]


def classify_prompt_from_examples(
    true_answer: str,
    qa_list: List[Dict[str, str]],
) -> str:
    """
    Classify the prompt based on the example questions and answers.

    Args:
        true_answer (str): Ground-truth code.
        qa_list (list[dict[str, str]]): List of question-answer dictionaries.

    Returns:
        str: Prompt class.
    """
    examples = qa_list[:-1] if qa_list else []

    codes = set()

    for example in examples:
        answer = (example.get("answer", "") or "").strip()

        if is_code_valid(answer):
            codes.add(norm_code(answer))

        for code in find_all_codes(example.get("question", "")):
            if is_code_valid(code):
                codes.add(norm_code(code))

    if not codes:
        return "prompt_none"

    if not is_code_valid(true_answer):
        return "prompt_wrong"

    ground_truth_code = norm_code(true_answer)
    ground_truth_base = get_base_code(ground_truth_code)

    if ground_truth_code in codes:
        return "prompt_exact"

    if any(get_base_code(code) == ground_truth_base for code in codes):
        return "prompt_partial"

    return "prompt_wrong"


def classify_output(true_answer: str, generated_answer: str) -> str:
    """
    Classify the generated output based on the answer type.

    Args:
        true_answer (str): Ground-truth code.
        generated_answer (str): Generated model answer.

    Returns:
        str: Output class.
    """
    category = check_answer(true_answer, generated_answer)

    if category == "exact match":
        return "out_exact"

    if category == "partial match code":
        return "out_partial"

    if category in ("false code", "false bool"):
        return "out_wrong"

    return "out_invalid"


def _stack_positions(
    sizes: List[float],
    total_height: float = 0.9,
    gap: float = 0.02,
) -> List[Tuple[float, float]]:
    """
    Calculate vertical positions for stacked blocks.

    Args:
        sizes (list[float]): List of block sizes.
        total_height (float): Total height available for all blocks.
        gap (float): Gap between blocks.

    Returns:
        list[tuple[float, float]]: Start and end positions for each block.
    """
    sizes = np.asarray(sizes, dtype=float)

    if sizes.sum() <= 0:
        return [(0.5, 0.5) for _ in sizes]

    relative_sizes = sizes / sizes.sum()
    total_gap = gap * max(0, len(sizes) - 1)
    scale = max(0.0, total_height - total_gap)
    heights = relative_sizes * scale

    positions = []
    y_position = (1.0 - total_height) / 2.0

    for height in heights:
        positions.append((y_position, y_position + height))
        y_position += height + gap

    return positions


def _ribbon_path(
    x0: float,
    y0a: float,
    y0b: float,
    x1: float,
    y1a: float,
    y1b: float,
    curvature: float = 0.35,
) -> MplPath:
    """
    Create a curved ribbon path between two stacked blocks.

    Args:
        x0 (float): Left x position.
        y0a (float): Lower left y position.
        y0b (float): Upper left y position.
        x1 (float): Right x position.
        y1a (float): Lower right y position.
        y1b (float): Upper right y position.
        curvature (float): Curvature strength.

    Returns:
        matplotlib.path.Path: Matplotlib path object.
    """
    ctrl1_up = (x0 + curvature * (x1 - x0), y0b)
    ctrl2_up = (x1 - curvature * (x1 - x0), y1b)
    ctrl1_down = (x1 - curvature * (x1 - x0), y1a)
    ctrl2_down = (x0 + curvature * (x1 - x0), y0a)

    vertices = [
        (x0, y0b),
        ctrl1_up,
        ctrl2_up,
        (x1, y1b),
        (x1, y1a),
        ctrl1_down,
        ctrl2_down,
        (x0, y0a),
        (x0, y0b),
    ]

    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]

    return MplPath(vertices, codes)


def _draw_sankey_matplotlib(
    flows_mat,
    left_labels,
    right_labels,
    out_png: str | Path,
    fig_w: int = 14,
    fig_h: int = 8,
    dpi: int = 200,
    colors=None,
    curvature: float = 0.35,
    show_percent: bool = True,
    percent_decimals: int = 1,
) -> None:
    """
    Draw a Sankey-style plot with Matplotlib.

    Args:
        flows_mat: Flow matrix.
        left_labels: Labels for the left side.
        right_labels: Labels for the right side.
        out_png (str | Path): Output PNG path.
        fig_w (int): Figure width.
        fig_h (int): Figure height.
        dpi (int): Image resolution.
        colors: Colors for the output classes.
        curvature (float): Curvature of the ribbons.
        show_percent (bool): Whether to show percentages.
        percent_decimals (int): Number of decimals for percentages.
    """
    del left_labels
    del right_labels

    flows = np.array(flows_mat, dtype=float)

    num_left, num_right = flows.shape
    left_sums = flows.sum(axis=1)
    right_sums = flows.sum(axis=0)
    total = float(flows.sum()) if flows.sum() > 0 else 0.0

    left_pos = _stack_positions(left_sums, total_height=0.9, gap=0.02)
    right_pos = _stack_positions(right_sums, total_height=0.9, gap=0.02)

    left_segments = []
    for left_index in range(num_left):
        y0, y1 = left_pos[left_index]
        height = max(0.0, y1 - y0)

        if left_sums[left_index] <= 0 or height <= 0:
            left_segments.append([(y0, y0) for _ in range(num_right)])
            continue

        relative_values = flows[left_index, :] / left_sums[left_index]
        segment_heights = relative_values * height
        cumulative = [y0]

        for height_part in segment_heights:
            cumulative.append(cumulative[-1] + height_part)

        left_segments.append(
            [
                (cumulative[index], cumulative[index + 1])
                for index in range(num_right)
            ]
        )

    right_segments = []
    for right_index in range(num_right):
        y0, y1 = right_pos[right_index]
        height = max(0.0, y1 - y0)

        if right_sums[right_index] <= 0 or height <= 0:
            right_segments.append([(y0, y0) for _ in range(num_left)])
            continue

        relative_values = flows[:, right_index] / right_sums[right_index]
        segment_heights = relative_values * height
        cumulative = [y0]

        for height_part in segment_heights:
            cumulative.append(cumulative[-1] + height_part)

        right_segments.append(
            [
                (cumulative[index], cumulative[index + 1])
                for index in range(num_left)
            ]
        )

    if colors is None:
        colors = [
            "#4CAF50",
            "#FFC107",
            "#FF7043",
            "#9E9E9E",
        ]

    colors = list(colors) + ["#999999"] * max(0, num_right - len(colors))

    left_block_colors = [
        colors[0],
        colors[1],
        colors[2],
        colors[3],
    ]

    right_block_colors = [
        colors[0],
        colors[1],
        colors[2],
        colors[3],
    ]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_left = 0.10
    x_right = 0.90
    block_width = 0.065
    percent_fontsize = 20

    header_fontsize = 28
    header_y = 0.985

    ax.text(
        x_left,
        header_y,
        "Prompt",
        ha="center",
        va="bottom",
        fontsize=header_fontsize,
        fontweight="bold",
    )
    ax.text(
        x_right,
        header_y,
        "RAG",
        ha="center",
        va="bottom",
        fontsize=header_fontsize,
        fontweight="bold",
    )

    for left_index in range(num_left):
        for right_index in range(num_right):
            value = flows[left_index, right_index]

            if value <= 0:
                continue

            y0a, y0b = left_segments[left_index][right_index]
            y1a, y1b = right_segments[right_index][left_index]

            path = _ribbon_path(
                x_left,
                y0a,
                y0b,
                x_right,
                y1a,
                y1b,
                curvature=curvature,
            )

            ax.add_patch(
                PathPatch(
                    path,
                    facecolor=colors[right_index],
                    edgecolor="none",
                    alpha=0.65,
                )
            )

    for left_index, (y0, y1) in enumerate(left_pos):
        face_color = left_block_colors[left_index]

        ax.add_patch(
            plt.Rectangle(
                (x_left - block_width / 2.0, y0),
                block_width,
                y1 - y0,
                facecolor=face_color,
                edgecolor="#333333",
                lw=0.8,
                alpha=0.70,
                zorder=3,
            )
        )

        if show_percent and total > 0 and left_sums[left_index] > 0:
            percentage = 100.0 * float(left_sums[left_index]) / total

            ax.text(
                x_left,
                (y0 + y1) / 2,
                f"{percentage:.{percent_decimals}f}%",
                ha="center",
                va="center",
                fontsize=percent_fontsize,
                fontweight="bold",
                color="black",
                zorder=4,
            )

    for right_index, (y0, y1) in enumerate(right_pos):
        face_color = right_block_colors[right_index]

        ax.add_patch(
            plt.Rectangle(
                (x_right - block_width / 2.0, y0),
                block_width,
                y1 - y0,
                facecolor=face_color,
                edgecolor="#333333",
                lw=0.8,
                alpha=0.70,
                zorder=3,
            )
        )

        if show_percent and total > 0 and right_sums[right_index] > 0:
            percentage = 100.0 * float(right_sums[right_index]) / total

            ax.text(
                x_right,
                (y0 + y1) / 2,
                f"{percentage:.{percent_decimals}f}%",
                ha="center",
                va="center",
                fontsize=percent_fontsize,
                fontweight="bold",
                color="black",
                zorder=4,
            )

    ax.set_title(
        "Llama-3.3-70B PEFT",
        fontsize=30,
        pad=10,
        fontweight="bold",
    )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.subplots_adjust(top=0.90, left=0.02, right=0.98, bottom=0.02)

    fig.savefig(
        out_png,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def _build_flows_from_json(
    data: List[Dict],
) -> Tuple[List[List[int]], List[str], List[str]]:
    """
    Build a 4x4 flow matrix from the JSON data.

    Args:
        data (list[dict]): List of JSON entries.

    Returns:
        tuple[list[list[int]], list[str], list[str]]: Flow matrix, left labels,
        and right labels.
    """
    prompt_order = [
        "prompt_exact",
        "prompt_partial",
        "prompt_wrong",
        "prompt_none",
    ]
    output_order = [
        "out_exact",
        "out_partial",
        "out_wrong",
        "out_invalid",
    ]

    label_map_left = {
        "prompt_exact": "Prompt: Exact",
        "prompt_partial": "Partial match",
        "prompt_wrong": "Mismatch",
        "prompt_none": "Malformed",
    }
    label_map_right = {
        "out_exact": "Exact match",
        "out_partial": "Partial match",
        "out_wrong": "Mismatch",
        "out_invalid": "Malformed",
    }

    flows = Counter()

    for entry in data:
        qa = entry.get("qa", []) or []
        ground_truth = entry.get("true_answer", "") or ""
        generated = entry.get("generated_answer", "") or ""

        prompt_category = classify_prompt_from_examples(ground_truth, qa)
        output_category = classify_output(ground_truth, generated)

        flows[(prompt_category, output_category)] += 1

    matrix = []

    for prompt in prompt_order:
        row = [
            int(flows.get((prompt, output), 0))
            for output in output_order
        ]
        matrix.append(row)

    left_labels = [label_map_left[prompt] for prompt in prompt_order]
    right_labels = [label_map_right[output] for output in output_order]

    return matrix, left_labels, right_labels


def build_sankey_from_json(
    input_json_path: str | Path,
    out_png_path: str | Path,
    fig_w: int = 14,
    fig_h: int = 8,
    dpi: int = 200,
) -> None:
    """
    Build a Sankey plot from a JSON file.

    Args:
        input_json_path (str | Path): Path to the input JSON file.
        out_png_path (str | Path): Path to the output PNG file.
        fig_w (int): Figure width.
        fig_h (int): Figure height.
        dpi (int): Image resolution.
    """
    input_json_path = Path(input_json_path)

    with open(input_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    matrix, left_labels, right_labels = _build_flows_from_json(data)

    _draw_sankey_matplotlib(
        flows_mat=matrix,
        left_labels=left_labels,
        right_labels=right_labels,
        out_png=out_png_path,
        fig_w=fig_w,
        fig_h=fig_h,
        dpi=dpi,
        colors=["#4CAF50", "#FFC107", "#FF7043", "#9E9E9E"],
        curvature=0.35,
        show_percent=True,
        percent_decimals=1,
    )

    print(f"[OK] Sankey PNG saved: {out_png_path}")


def main() -> None:
    """Build the Sankey plot from the configured JSON response file."""
    build_sankey_from_json(
        input_json_path=INPUT_JSON,
        out_png_path=OUT_PNG,
        fig_w=14,
        fig_h=9,
        dpi=300,
    )


if __name__ == "__main__":
    main()