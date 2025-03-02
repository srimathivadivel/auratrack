import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tracker (
        username TEXT, date TEXT, screentime REAL, emotion TEXT, notes TEXT
    )''')
    conn.commit()
    conn.close()

def save_user_data(username, date, screentime, emotion, notes):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO tracker VALUES (?, ?, ?, ?, ?)",
              (username, date, screentime, emotion, notes))
    conn.commit()
    conn.close()

def get_user_data(username):
    conn = sqlite3.connect('user_data.db')
    df = pd.read_sql_query("SELECT * FROM tracker WHERE username = ?", conn, params=(username,))
    conn.close()
    return df