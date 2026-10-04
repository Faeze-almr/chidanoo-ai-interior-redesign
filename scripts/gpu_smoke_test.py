"""Final GPU smoke test for the three canonical workflows.

This script does not create masks or modify benchmark inputs. The room_04
mask defaults to the frozen refined-bed validation fixture and can be
overridden. The fixture is test evidence only, not product mask logic.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.errors import GenerationDomainError  # noqa: E402
from app.core.ip_adapter import reference_adapter_manager  # noqa: E402
from app.core.pipeline import (  # noqa: E402
    generate_reference_result,
    generate_region_result,
    generate_room_result,
)


SMOKE_CASES = (
    "room_08_full",
    "room_16_full",
    "room_04_targeted",
    "reference",
    "full_after_reference",
)


def _parse_cases(value: str) -> tuple[str, ...]:
    requested = tuple(case.strip() for case in value.split(",") if case.strip())
    invalid = sorted(set(requested) - set(SMOKE_CASES))
    if not requested or invalid:
        valid = ", ".join(SMOKE_CASES)
        detail = f"; unknown: {', '.join(invalid)}" if invalid else ""
        raise argparse.ArgumentTypeError(f"cases must be a comma-separated subset of: {valid}{detail}")
    return requested


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    benchmark = WORKSPACE_ROOT / "benchmark"
    parser = argparse.ArgumentParser(description="Run the final five-case CUDA smoke suite.")
    parser.add_argument("--room08", type=Path, default=benchmark / "inputs/core/room_08.jpg")
    parser.add_argument("--room16", type=Path, default=benchmark / "inputs/core/room_16.jpg")
    parser.add_argument("--room04", type=Path, default=benchmark / "inputs/core/room_04.jpg")
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=benchmark / "inputs/core/room_01.jpg",
    )
    parser.add_argument(
        "--room04-mask",
        type=Path,
        default=PROJECT_ROOT / "test_fixtures/room_04_furniture_mask.png",
    )
    parser.add_argument(
        "--cases",
        type=_parse_cases,
        default=SMOKE_CASES,
        help=(
            "Optional comma-separated case subset. Defaults to all five. "
            "Quick Reference debug: --cases reference,full_after_reference"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/gpu_smoke")
    return parser.parse_args(argv)


def save_result(name: str, result, output_dir: Path) -> dict:
    output_path = output_dir / f"{name}.png"
    result.image.save(output_path, format="PNG")
    record = result.metadata.to_dict()
    record["case"] = name
    record["output_path"] = str(output_path)
    return record


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    full_cases = [
        ("room_08_full", args.room08),
        ("room_16_full", args.room16),
    ]
    for name, image_path in full_cases:
        if name not in args.cases:
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            result = generate_room_result(image, style_key="modern")
            records.append(save_result(name, result, args.output_dir))
        except GenerationDomainError as exc:
            logging.exception("Smoke case %s failed", name)
            records.append({"case": name, "status": "failed", "error": exc.user_message})
        except Exception:
            logging.exception("Unexpected smoke failure for %s", name)
            records.append({"case": name, "status": "failed", "error": "Unexpected smoke-test failure."})

    if "room_04_targeted" in args.cases:
        try:
            image = Image.open(args.room04).convert("RGB")
            mask = Image.open(args.room04_mask)
            result = generate_region_result(
                image,
                mask,
                style_key="modern",
                target_description="bed",
            )
            records.append(save_result("room_04_targeted", result, args.output_dir))
        except GenerationDomainError as exc:
            logging.exception("Smoke case room_04_targeted failed")
            records.append(
                {"case": "room_04_targeted", "status": "failed", "error": exc.user_message}
            )
        except Exception:
            logging.exception("Unexpected smoke failure for room_04_targeted")
            records.append(
                {
                    "case": "room_04_targeted",
                    "status": "failed",
                    "error": "Unexpected smoke-test failure.",
                }
            )

    # Reference feature case: the adapter is loaded lazily for this operation
    # and must be absent immediately afterward.
    if "reference" in args.cases:
        try:
            image = Image.open(args.room08).convert("RGB")
            reference = Image.open(args.reference_image).convert("RGB")
            result = generate_reference_result(
                image,
                reference,
            )
            record = save_result("room_08_reference", result, args.output_dir)
            record["adapter_loaded_after_case"] = reference_adapter_manager.is_loaded
            record["cleanup_memory_allocated"] = (
                reference_adapter_manager.last_cleanup_allocated
            )
            record["cleanup_memory_reserved"] = reference_adapter_manager.last_cleanup_reserved
            records.append(record)
        except GenerationDomainError as exc:
            logging.exception("Smoke case room_08_reference failed")
            records.append(
                {"case": "room_08_reference", "status": "failed", "error": exc.user_message}
            )
        except Exception:
            logging.exception("Unexpected smoke failure for room_08_reference")
            records.append(
                {
                    "case": "room_08_reference",
                    "status": "failed",
                    "error": "Unexpected smoke-test failure.",
                }
            )

    # Required isolation proof: Full Redesign must load a clean canonical
    # pipeline after the disposable Reference pipeline has been deleted.
    if "full_after_reference" in args.cases:
        if "reference" in args.cases and not reference_adapter_manager.cleanup_allows_full_reload:
            records.append(
                {
                    "case": "room_08_full_after_reference",
                    "status": "failed",
                    "error": "Skipped because Reference cleanup left abnormal CUDA allocation.",
                    "cleanup_memory_allocated": (
                        reference_adapter_manager.last_cleanup_allocated
                    ),
                    "cleanup_memory_reserved": reference_adapter_manager.last_cleanup_reserved,
                }
            )
        else:
            try:
                image = Image.open(args.room08).convert("RGB")
                adapter_loaded_before = reference_adapter_manager.is_loaded
                result = generate_room_result(image, style_key="modern")
                record = save_result("room_08_full_after_reference", result, args.output_dir)
                record["adapter_loaded_before_case"] = adapter_loaded_before
                record["adapter_loaded_after_case"] = reference_adapter_manager.is_loaded
                records.append(record)
            except GenerationDomainError as exc:
                logging.exception("Smoke case room_08_full_after_reference failed")
                records.append(
                    {
                        "case": "room_08_full_after_reference",
                        "status": "failed",
                        "error": exc.user_message,
                    }
                )
            except Exception:
                logging.exception("Unexpected smoke failure for room_08_full_after_reference")
                records.append(
                    {
                        "case": "room_08_full_after_reference",
                        "status": "failed",
                        "error": "Unexpected smoke-test failure.",
                    }
                )

    summary_path = args.output_dir / "smoke_results.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    statuses_ok = all(record.get("status") == "succeeded" for record in records)
    adapter_state_ok = all(
        not value
        for record in records
        for key, value in record.items()
        if key.startswith("adapter_loaded_")
    )
    return 0 if statuses_ok and adapter_state_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
