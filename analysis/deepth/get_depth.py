import sqlite3
import os
from collections import Counter
from datetime import datetime
import csv

def _safe_parse_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        pass
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except Exception:
                continue
    return None


def analyze_database(db_path):
    """分析单个数据库，返回转账深度列表"""
    print(f"\n--- 正在分析: {os.path.basename(db_path)} ---")
    depths = []
    if not os.path.exists(db_path):
        print("错误: 数据库文件未找到。")
        return depths

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transfer_money';")
        has_transfer = cursor.fetchone() is not None
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='private_message';")
        has_pm = cursor.fetchone() is not None
        if not (has_transfer and has_pm):
            print("缺少必要的表。")
            return depths

        cursor.execute("SELECT sender_id, receiver_id, timestamp FROM transfer_money WHERE sender_id < 100")
        transfers = cursor.fetchall()
        if not transfers:
            print("没有转账记录。")
            return depths

        for sender_id, receiver_id, t_time in transfers:
            t_ts = _safe_parse_time(t_time)
            pm_query = """
                SELECT timestamp FROM private_message
                WHERE (sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?)
            """
            cursor.execute(pm_query, (sender_id, receiver_id, receiver_id, sender_id))
            pm_rows = cursor.fetchall()

            pm_times = []
            parse_failed = False
            for (pm_time,) in pm_rows:
                pm_ts = _safe_parse_time(pm_time)
                if pm_ts is None:
                    parse_failed = True
                    break
                pm_times.append(pm_ts)

            if t_ts is None:
                parse_failed = True

            if parse_failed:
                depth = len(pm_rows)
            else:
                depth = sum(1 for pm_ts in pm_times if pm_ts <= t_ts)

            depths.append(depth)

        return depths

    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
        return depths
    finally:
        if 'conn' in locals() and conn:
            conn.close()


def scan_directory(directory_path, output_csv):
    """扫描目录，输出整体深度分布表格"""
    if not os.path.isdir(directory_path):
        print(f"错误: 目录 '{directory_path}' 未找到。")
        return

    all_depths = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".db"):
            db_path = os.path.join(directory_path, filename)
            depths = analyze_database(db_path)
            all_depths.extend(depths)

    if not all_depths:
        print("没有可用的转账深度数据。")
        return

    hist = Counter(all_depths)
    total = sum(hist.values())

    # 导出 CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["depth", "count", "ratio"])
        for depth, count in sorted(hist.items()):
            ratio = count / total if total else 0.0
            writer.writerow([depth, count, f"{ratio:.4f}"])

    print(f"已导出深度分布表格: {output_csv}")


if __name__ == "__main__":
    target_directory = "/home/zhengzhijie/h-cluster-storage/mutiAgent4Fraud/data/simu_db/dataset1"
    output_csv = "/home/zhengzhijie/h-cluster-storage/mutiAgent4Fraud/MAST/depth_distribution.csv"
    scan_directory(target_directory, output_csv)
