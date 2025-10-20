CREATE TABLE IF NOT EXISTS transfer_money (
    transfer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id INTEGER NOT NULL, 
    receiver_id INTEGER NOT NULL,
    amount INTEGER NOT NULL, 
    reason TEXT NOT NULL, 
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
    FOREIGN KEY(sender_id) REFERENCES user(user_id),
    FOREIGN KEY(receiver_id) REFERENCES user(user_id)
);