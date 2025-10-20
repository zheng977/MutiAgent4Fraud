import argparse
import csv
import glob
import os
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_DB_ROOT = os.getenv(
    "ACTION_DB_ROOT",
    os.path.join("data", "simu_db", "dataset"),
)
DEFAULT_OUTPUT = os.getenv(
    "ACTION_OUTPUT_CSV",
    os.path.join("analysis", "action_frequency", "action_counts.csv"),
)
DEFAULT_USER_RANGE: Tuple[int, int] = (
    int(os.getenv("ACTION_USER_MIN", 100)),
    int(os.getenv("ACTION_USER_MAX", 109)),
)
DEFAULT_ACTIONS = [
    "like_post",
    "create_post",
    "create_comment",
    "send_private_message",
    "repost",
]


def _ensure_table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _fetch_action_counts(
    cursor: sqlite3.Cursor,
    user_min: int,
    user_max: int,
    include_actions: Sequence[str],
) -> Dict[str, int]:
    cursor.execute(
        "SELECT action FROM trace WHERE user_id BETWEEN ? AND ?",
        (user_min, user_max),
    )
    rows = cursor.fetchall()
    action_counts: Dict[str, int] = {act: 0 for act in include_actions}
    for (action,) in rows:
        if action in action_counts:
            action_counts[action] += 1
    return action_counts


def analyze_db(
    db_path: str,
    user_min: int,
    user_max: int,
    include_actions: Sequence[str],
) -> Optional[Dict[str, int]]:
    if not os.path.exists(db_path):
        print(f"Error: database file not found: {db_path}")
        return None
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        if not _ensure_table_exists(cursor, "trace"):
            print(f"{db_path}: table 'trace' not found.")
            return None

        action_counts = _fetch_action_counts(
            cursor, user_min, user_max, include_actions
        )
        total = sum(action_counts.values())
        if total == 0:
            print(
                f"{db_path}: no matching actions for users "
                f"{user_min}-{user_max}."
            )
            return None

        print("=" * 80)
        print(f"Database: {db_path}")
        for act in include_actions:
            print(f"{act:25s} {action_counts[act]:8d}")
        print("-" * 80)
        print(f"Total records: {total}")

        return {"db": db_path, **action_counts, "total": total}
    except sqlite3.Error as e:
        print(f"{db_path}: database error: {e}")
        return None
    finally:
        if conn:
            conn.close()


def run_analysis(
    db_root: str,
    output_csv: str,
    user_min: int,
    user_max: int,
    include_actions: Sequence[str],
    recursive: bool,
) -> None:
    if not os.path.isdir(db_root):
        print(f"Error: '{db_root}' is not a directory or does not exist.")
        return

    pattern = (
        os.path.join(db_root, "**", "*.db")
        if recursive
        else os.path.join(db_root, "*.db")
    )
    db_files: List[str] = glob.glob(pattern, recursive=recursive)
    if not db_files:
        print(f"No .db files found under '{db_root}'.")
        return

    print(f"Found {len(db_files)} database files under '{db_root}'.\n")
    results = []
    for db in sorted(db_files):
        res = analyze_db(db, user_min, user_max, include_actions)
        if res:
            results.append(res)

    if results:
        header = ["db"] + list(include_actions) + ["total"]
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            print(f"\nCSV exported to: {output_csv}")
        except Exception as e:
            print(f"Failed to export CSV: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate action counts for a range of agent IDs."
    )
    parser.add_argument(
        "--db-root",
        default=DEFAULT_DB_ROOT,
        help=f"Root directory containing SQLite databases (default: {DEFAULT_DB_ROOT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to write the aggregated CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--user-min",
        type=int,
        default=DEFAULT_USER_RANGE[0],
        help=f"Minimum agent ID to include (default: {DEFAULT_USER_RANGE[0]})",
    )
    parser.add_argument(
        "--user-max",
        type=int,
        default=DEFAULT_USER_RANGE[1],
        help=f"Maximum agent ID to include (default: {DEFAULT_USER_RANGE[1]})",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        default=DEFAULT_ACTIONS,
        help=(
            "Whitelist of actions to count. "
            f"Defaults to {', '.join(DEFAULT_ACTIONS)}"
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the top-level directory for .db files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        db_root=args.db_root,
        output_csv=args.output,
        user_min=args.user_min,
        user_max=args.user_max,
        include_actions=args.actions,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
