CREATE TABLE IF NOT EXISTS private_message (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_id INTEGER NOT NULL, -- 发送者 User ID
        receiver_id INTEGER NOT NULL, -- 接收者 User ID
        content TEXT NOT NULL,        -- 消息内容
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, -- 发送时间
        is_read INTEGER DEFAULT 0, -- 是否已读 (0: 未读, 1: 已读)
        FOREIGN KEY (sender_id) REFERENCES user(user_id),
        FOREIGN KEY (receiver_id) REFERENCES user(user_id)
    );