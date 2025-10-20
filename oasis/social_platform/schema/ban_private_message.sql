CREATE TABLE IF NOT EXISTS banned_private_channel (
    user_id_1 INTEGER NOT NULL,
    user_id_2 INTEGER NOT NULL,
    ban_time  DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id_1, user_id_2),
    FOREIGN KEY(user_id_1) REFERENCES user(user_id),
    FOREIGN KEY(user_id_2) REFERENCES user(user_id)
);