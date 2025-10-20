CREATE TABLE IF NOT EXISTS submit_info (
    submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL, 
    receiver_id INTEGER NOT NULL, 
    info_type TEXT NOT NULL, 
    info_content TEXT NOT NULL, 
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
    FOREIGN KEY(user_id) REFERENCES user(user_id),
    FOREIGN KEY(receiver_id) REFERENCES user(user_id)
);