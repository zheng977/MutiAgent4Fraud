 CREATE TABLE IF NOT EXISTS fraud_stats (
    stats_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scammer_id INTEGER NOT NULL,
    victim_id INTEGER NOT NULL,
    fraud_type TEXT NOT NULL, -- 'transfer_money', 'click_link', 'submit_info'
    count INTEGER DEFAULT 1,
    first_occurrence DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_occurrence DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(scammer_id) REFERENCES user(user_id),
    FOREIGN KEY(victim_id) REFERENCES user(user_id),
    UNIQUE(scammer_id, victim_id, fraud_type)
);