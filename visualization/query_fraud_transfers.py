import os
import sqlite3

def analyze_database(db_path):
    """Inspect a single SQLite database and report transfer activity."""
    print(f"\n--- Analyzing: {os.path.basename(db_path)} ---")

    if not os.path.exists(db_path):
        print("Error: database file not found.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transfer_money';"
        )
        if cursor.fetchone() is None:
            print("Table 'transfer_money' not found in this database.")
            return

        query = (
            "SELECT sender_id, receiver_id FROM transfer_money WHERE sender_id < 100"
        )
        cursor.execute(query)
        transfers = cursor.fetchall()

        if not transfers:
            print("No transfer records found.")
            return

        print(f"Raw transfer count: {len(transfers)}")

        unique_transfers = set(transfers)
        print(
            "Unique transfer pairs (sender -> receiver): "
            f"{len(unique_transfers)}"
        )

        sender_ids = {s for s, r in unique_transfers}
        receiver_ids = {r for s, r in unique_transfers}

        if sender_ids:
            print(
                f"Unique sender agent IDs ({len(sender_ids)}): "
                f"{sorted(sender_ids)}"
            )
        else:
            print("No sender IDs found.")

        if receiver_ids:
            print(
                f"Unique receiver agent IDs ({len(receiver_ids)}): "
                f"{sorted(receiver_ids)}"
            )
        else:
            print("No receiver IDs found.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def scan_directory(directory_path):
    """Scan a directory for .db files and analyze each one."""
    if not os.path.isdir(directory_path):
        print(f"Error: directory '{directory_path}' not found or invalid.")
        return

    print(f"Scanning directory: {directory_path}")
    db_files_found = False
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".db"):
            db_files_found = True
            db_path = os.path.join(directory_path, filename)
            analyze_database(db_path)

    if not db_files_found:
        print("No .db files found in the specified directory.")


if __name__ == "__main__":
    target_directory = ""
    print(f"The script will scan the directory: '{target_directory}'")

    scan_directory(target_directory)
