CREATE TABLE IF NOT EXISTS click_link (
    click_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL, 
    sender_id INTEGER NOT NULL, 
    link_url TEXT NOT NULL, 
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
    FOREIGN KEY(user_id) REFERENCES user(user_id),
    FOREIGN KEY(sender_id) REFERENCES user(user_id)
);