#!/usr/bin/env python3

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List


def analyze_transfer_conversations(db_path: str) -> List[Dict]:
    """
    Extract transfer records and related private conversations from a SQLite DB.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        A list of dictionaries describing each transfer and associated dialogue.
    """
    print(f"\n--- Analyzing: {os.path.basename(db_path)} ---")

    if not os.path.exists(db_path):
        print("Error: database file not found.")
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        required_tables = ["transfer_money", "private_message"]
        for table in required_tables:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (table,),
            )
            if cursor.fetchone() is None:
                print(f"Table '{table}' not found in this database.")
                return []

        transfer_query = """
            SELECT rowid, sender_id, receiver_id, amount, reason, timestamp 
            FROM transfer_money 
            WHERE sender_id < 100
            ORDER BY timestamp
        """
        cursor.execute(transfer_query)
        transfers = cursor.fetchall()

        if not transfers:
            print("No transfer records found.")
            return []

        print(f"Found {len(transfers)} transfer records")

        dataset = []

        for transfer in transfers:
            transfer_id, sender_id, receiver_id, amount, reason, timestamp = transfer

            pm_query = """
                SELECT sender_id, receiver_id, content, timestamp
                FROM private_message
                WHERE (sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?)
                ORDER BY timestamp
            """
            cursor.execute(
                pm_query, (sender_id, receiver_id, receiver_id, sender_id)
            )
            messages = cursor.fetchall()

            conversation_content = ""
            has_conversation = 0

            if messages:
                has_conversation = 1
                conversation_lines = []
                for msg_sender, msg_receiver, content, msg_time in messages:
                    if msg_sender == sender_id:
                        conversation_lines.append(f"Sender: {content}")
                    else:
                        conversation_lines.append(f"Receiver: {content}")

                conversation_content = "\n".join(conversation_lines)
            else:
                conversation_content = "No private messages."

            entry = {
                "id": len(dataset) + 1,
                "database_name": os.path.basename(db_path),
                "amount": amount,
                "reason": reason if reason else "No stated reason.",
                "conversation": conversation_content,
                "has_conversation": has_conversation,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "transfer_timestamp": timestamp,
            }

            dataset.append(entry)

        print(f"Extracted {len(dataset)} transfer entries")
        return dataset

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def scan_directory_for_transfers(directory_path: str) -> List[Dict]:
    """Scan a directory for .db files and extract transfer-related data."""
    if not os.path.isdir(directory_path):
        print(f"Error: directory '{directory_path}' not found or invalid.")
        return []

    print(f"Scanning directory: {directory_path}")

    all_dataset = []
    global_id = 1

    db_files_found = False
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".db"):
            db_files_found = True
            db_path = os.path.join(directory_path, filename)
            dataset = analyze_transfer_conversations(db_path)
            for entry in dataset:
                entry["id"] = global_id
                global_id += 1

            all_dataset.extend(dataset)

    if not db_files_found:
        print("No .db files found in the specified directory.")

    return all_dataset

def export_to_jsonl(dataset: List[Dict], output_path: str):
    """Export the dataset to JSONL format."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Dataset exported to: {output_path}")
    except Exception as e:
        print(f"Failed to export dataset: {e}")

def print_statistics(dataset: List[Dict]):
    """Print aggregate statistics for the dataset."""
    if not dataset:
        print("Dataset is empty.")
        return

    total_records = len(dataset)
    records_with_conversation = sum(
        1 for entry in dataset if entry["has_conversation"] == 1
    )
    records_without_conversation = total_records - records_with_conversation

    print("\n=== Dataset Summary ===")
    print(f"Total transfers: {total_records}")
    print(
        f"Transfers with conversation: {records_with_conversation} "
        f"({records_with_conversation / total_records * 100:.1f}%)"
    )
    print(
        f"Transfers without conversation: {records_without_conversation} "
        f"({records_without_conversation / total_records * 100:.1f}%)"
    )

def main():
    """Entry point for the CLI utility."""
    target_directory = "/mnt/petrelfs/zhengzhijie/mutiAgent4Fraud/data/simu_db/expierement/different_bad_model"
    output_dir = "/mnt/petrelfs/zhengzhijie/mutiAgent4Fraud/data/datasets"

    print("=== Transfer Conversation Extraction ===")
    print(f"Scanning directory: {target_directory}")

    os.makedirs(output_dir, exist_ok=True)
    dataset = scan_directory_for_transfers(target_directory)

    if not dataset:
        print("No transfer-related data found.")
        return

    print_statistics(dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(
        output_dir, f"transfer_conversations_{timestamp}.jsonl"
    )
    export_to_jsonl(dataset, jsonl_path)

    print("\n=== Sample Records ===")
    for i, entry in enumerate(dataset[:3]):
        print(f"\nSample {i + 1}:")
        for key, value in entry.items():
            if key == "conversation" and len(str(value)) > 100:
                print(f"  {key}: {str(value)[:100]}...")
            else:
                print(f"  {key}: {value}")

    print("\nDataset extraction complete.")

if __name__ == "__main__":
    main()
