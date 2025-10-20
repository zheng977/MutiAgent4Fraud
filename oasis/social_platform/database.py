# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import os
import os.path as osp
import sqlite3
from typing import Any, Dict, List

SCHEMA_DIR = "social_platform/schema"
DB_DIR = "db"
DB_NAME = "social_media.db"

USER_SCHEMA_SQL = "user.sql"
POST_SCHEMA_SQL = "post.sql"
FOLLOW_SCHEMA_SQL = "follow.sql"
MUTE_SCHEMA_SQL = "mute.sql"
LIKE_SCHEMA_SQL = "like.sql"
DISLIKE_SCHEMA_SQL = "dislike.sql"
TRACE_SCHEMA_SQL = "trace.sql"
REC_SCHEMA_SQL = "rec.sql"
COMMENT_SCHEMA_SQL = "comment.sql"
COMMENT_LIKE_SCHEMA_SQL = "comment_like.sql"
COMMENT_DISLIKE_SCHEMA_SQL = "comment_dislike.sql"
PRODUCT_SCHEMA_SQL = "product.sql"
PRIVATE_MESSAGE_SCHEMA_SQL = "private_message.sql"
FRAUD_STATS_SCHEMA_SQL = "fraud_stats.sql"
TRANSFER_MONEY_SCHEMA_SQL = "transfer_money.sql"
CLICK_LINK_SCHEMA_SQL = "click_link.sql"
SUBMIT_INFO_SCHEMA_SQL = "submit_info.sql"
FRAUD_STATS_SCHEMA_SQL = "fraud_stats.sql"
BAN_PRIVATE_MESSAGE_SCHEMA_SQL = "ban_private_message.sql"

TABLE_NAMES = {
    "user",
    "post",
    "follow",
    "mute",
    "like",
    "dislike",
    "trace",
    "rec",
    "comment.sql",
    "comment_like.sql",
    "comment_dislike.sql",
    "product.sql",
    "private_message",
    "fraud_stats",
    "transfer_money",
    "click_link",
    "submit_info",
    "fraud_stats",
    "banned_private_message",
}


def get_db_path() -> str:
    curr_file_path = osp.abspath(__file__)
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    db_dir = osp.join(parent_dir, DB_DIR)
    os.makedirs(db_dir, exist_ok=True)
    db_path = osp.join(db_dir, DB_NAME)
    return db_path


def get_schema_dir_path() -> str:
    curr_file_path = osp.abspath(__file__)
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    schema_dir = osp.join(parent_dir, SCHEMA_DIR)
    return schema_dir


def create_db(db_path: str | None = None):
    r"""Create the database if it does not exist. A :obj:`twitter.db`
    file will be automatically created  in the :obj:`data` directory.
    """
    schema_dir = get_schema_dir_path()
    if db_path is None:
        db_path = get_db_path()

    # Connect to the database:
    print("db_path", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Read and execute the user table SQL script:
        user_sql_path = osp.join(schema_dir, USER_SCHEMA_SQL)
        with open(user_sql_path, "r") as sql_file:
            user_sql_script = sql_file.read()
        cursor.executescript(user_sql_script)

        # Read and execute the post table SQL script:
        post_sql_path = osp.join(schema_dir, POST_SCHEMA_SQL)
        with open(post_sql_path, "r") as sql_file:
            post_sql_script = sql_file.read()
        cursor.executescript(post_sql_script)

        # Read and execute the follow table SQL script:
        follow_sql_path = osp.join(schema_dir, FOLLOW_SCHEMA_SQL)
        with open(follow_sql_path, "r") as sql_file:
            follow_sql_script = sql_file.read()
        cursor.executescript(follow_sql_script)

        # Read and execute the mute table SQL script:
        mute_sql_path = osp.join(schema_dir, MUTE_SCHEMA_SQL)
        with open(mute_sql_path, "r") as sql_file:
            mute_sql_script = sql_file.read()
        cursor.executescript(mute_sql_script)

        # Read and execute the like table SQL script:
        like_sql_path = osp.join(schema_dir, LIKE_SCHEMA_SQL)
        with open(like_sql_path, "r") as sql_file:
            like_sql_script = sql_file.read()
        cursor.executescript(like_sql_script)

        # Read and execute the dislike table SQL script:
        dislike_sql_path = osp.join(schema_dir, DISLIKE_SCHEMA_SQL)
        with open(dislike_sql_path, "r") as sql_file:
            dislike_sql_script = sql_file.read()
        cursor.executescript(dislike_sql_script)

        # Read and execute the trace table SQL script:
        trace_sql_path = osp.join(schema_dir, TRACE_SCHEMA_SQL)
        with open(trace_sql_path, "r") as sql_file:
            trace_sql_script = sql_file.read()
        cursor.executescript(trace_sql_script)

        # Read and execute the rec table SQL script:
        rec_sql_path = osp.join(schema_dir, REC_SCHEMA_SQL)
        with open(rec_sql_path, "r") as sql_file:
            rec_sql_script = sql_file.read()
        cursor.executescript(rec_sql_script)

        # Read and execute the comment table SQL script:
        comment_sql_path = osp.join(schema_dir, COMMENT_SCHEMA_SQL)
        with open(comment_sql_path, "r") as sql_file:
            comment_sql_script = sql_file.read()
        cursor.executescript(comment_sql_script)

        # Read and execute the comment_like table SQL script:
        comment_like_sql_path = osp.join(schema_dir, COMMENT_LIKE_SCHEMA_SQL)
        with open(comment_like_sql_path, "r") as sql_file:
            comment_like_sql_script = sql_file.read()
        cursor.executescript(comment_like_sql_script)

        # Read and execute the comment_dislike table SQL script:
        comment_dislike_sql_path = osp.join(schema_dir,
                                            COMMENT_DISLIKE_SCHEMA_SQL)
        with open(comment_dislike_sql_path, "r") as sql_file:
            comment_dislike_sql_script = sql_file.read()
        cursor.executescript(comment_dislike_sql_script)

        # Read and execute the product table SQL script:
        product_sql_path = osp.join(schema_dir, PRODUCT_SCHEMA_SQL)
        with open(product_sql_path, "r") as sql_file:
            product_sql_script = sql_file.read()
        cursor.executescript(product_sql_script)

        # Read and execute the private_message table SQL script:
        private_message_sql_path = osp.join(schema_dir,
                                            PRIVATE_MESSAGE_SCHEMA_SQL)
        with open(private_message_sql_path, "r") as sql_file:
            private_message_sql_script = sql_file.read()
        cursor.executescript(private_message_sql_script)

        # Read and execute the fraud_stats table SQL script:
        fraud_stats_sql_path = osp.join(schema_dir, FRAUD_STATS_SCHEMA_SQL)
        with open(fraud_stats_sql_path, "r") as sql_file:
            fraud_stats_sql_script = sql_file.read()
        cursor.executescript(fraud_stats_sql_script)    

        # Read and execute the transfer_money table SQL script:
        transfer_money_sql_path = osp.join(schema_dir, TRANSFER_MONEY_SCHEMA_SQL)
        with open(transfer_money_sql_path, "r") as sql_file:
            transfer_money_sql_script = sql_file.read()
        cursor.executescript(transfer_money_sql_script)

        # Read and execute the click_link table SQL script:
        click_link_sql_path = osp.join(schema_dir, CLICK_LINK_SCHEMA_SQL)
        with open(click_link_sql_path, "r") as sql_file:
            click_link_sql_script = sql_file.read()
        cursor.executescript(click_link_sql_script)

        # Read and execute the submit_info table SQL script:
        submit_info_sql_path = osp.join(schema_dir, SUBMIT_INFO_SCHEMA_SQL)
        with open(submit_info_sql_path, "r") as sql_file:
            submit_info_sql_script = sql_file.read()
        cursor.executescript(submit_info_sql_script)
        
        fraud_stats_sql_path = osp.join(schema_dir, FRAUD_STATS_SCHEMA_SQL)
        with open(fraud_stats_sql_path, "r") as sql_file:
            fraud_stats_sql_script = sql_file.read()
        cursor.executescript(fraud_stats_sql_script)

        ban_private_message_sql_path = osp.join(schema_dir, BAN_PRIVATE_MESSAGE_SCHEMA_SQL)
        with open(ban_private_message_sql_path, "r") as sql_file:
            ban_private_message_sql_script = sql_file.read()
        cursor.executescript(ban_private_message_sql_script)
        # Commit the changes:
        conn.commit()

    except sqlite3.Error as e:
        print(f"An error occurred while creating tables: {e}")

    return conn, cursor


def print_db_tables_summary():
    # Connect to the SQLite database
    # db_path = get_db_path()
    db_path = '/mnt/petrelfs/zhengzhijie/multiAgent4Fakenews/oasis/db/social_media.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve a list of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Print a summary of each table
    for table in tables:
        table_name = table[0]
        if table_name not in TABLE_NAMES:
            continue
        print(f"Table: {table_name}")

        # Retrieve the table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        print("- Columns:", column_names)

        # Retrieve and print foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = cursor.fetchall()
        if foreign_keys:
            print("- Foreign Keys:")
            for fk in foreign_keys:
                print(f"    {fk[2]} references {fk[3]}({fk[4]}) on update "
                      f"{fk[5]} on delete {fk[6]}")
        else:
            print("  No foreign keys.")

        # Print the first few rows of the table
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        print()  # Adds a newline for better readability between tables

    # Close the database connection
    conn.close()


def fetch_table_from_db(cursor: sqlite3.Cursor,
                        table_name: str) -> List[Dict[str, Any]]:
    cursor.execute(f"SELECT * FROM {table_name}")
    columns = [description[0] for description in cursor.description]
    data_dicts = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return data_dicts


def fetch_rec_table_as_matrix(cursor: sqlite3.Cursor) -> List[List[int]]:
    # First, query all user_ids from the user table, assuming they start from
    # 1 and are consecutive
    cursor.execute("SELECT user_id FROM user ORDER BY user_id")
    user_ids = [row[0] for row in cursor.fetchall()]

    # Then, query all records from the rec table
    cursor.execute(
        "SELECT user_id, post_id FROM rec ORDER BY user_id, post_id")
    rec_rows = cursor.fetchall()
    # Initialize a dictionary, assigning an empty list to each user_id
    user_posts = {user_id: [] for user_id in user_ids}
    # Fill the dictionary with the records queried from the rec table
    for user_id, post_id in rec_rows:
        if user_id in user_posts:
            user_posts[user_id].append(post_id)
    # Convert the dictionary into matrix form
    matrix = [user_posts[user_id] for user_id in user_ids]
    return matrix


def insert_matrix_into_rec_table(cursor: sqlite3.Cursor,
                                 matrix: List[List[int]]) -> None:
    # Iterate through the matrix, skipping the placeholder at index 0
    for user_id, post_ids in enumerate(matrix, start=1):
        # Adjusted to start counting from 1
        for post_id in post_ids:
            # Insert each combination of user_id and post_id into the rec table
            cursor.execute("INSERT INTO rec (user_id, post_id) VALUES (?, ?)",
                           (user_id, post_id))


if __name__ == "__main__":
    # create_db('/mnt/petrelfs/zhengzhijie/multiAgent4Fakenews/oasis/db/social_media.db')
    # create_db()
    print_db_tables_summary()
