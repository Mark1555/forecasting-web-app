"""
Таблиці:
  users          — акаунти користувачів (bcrypt паролі)
  forecasts      — збережені прогнози (points_json замість окремої таблиці)
  uploaded_files — метадані про завантажені CSV файли
"""

import sqlite3
import bcrypt
import os
import json
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecaster.db")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row           
    conn.execute("PRAGMA foreign_keys = ON") 
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS forecasts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name            TEXT    NOT NULL, 
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),

            model_type      TEXT    NOT NULL, 
            forecast_days   INTEGER NOT NULL,

            last_value      REAL    NOT NULL,         
            forecast_end    REAL    NOT NULL,         
            delta_pct       REAL,                     

            accuracy        REAL,                      
            mape            REAL,                      
            mae             REAL,                      

            history_json    TEXT    NOT NULL DEFAULT '[]',
            forecast_json   TEXT    NOT NULL DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS uploaded_files (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            forecast_id     INTEGER REFERENCES forecasts(id) ON DELETE SET NULL,
            filename        TEXT    NOT NULL,
            row_count       INTEGER,
            date_column     TEXT,
            value_column    TEXT,
            date_from       TEXT,                     
            date_to         TEXT,                     
            uploaded_at     TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_forecasts_user
            ON forecasts(user_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_files_user
            ON uploaded_files(user_id);
        """)


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8")
    )


def register_user(username: str, email: str, password: str) -> dict:
    if len(username.strip()) < 3:
        return {"ok": False, "error": "Ім'я користувача має бути мінімум 3 символи."}
    if len(password) < 6:
        return {"ok": False, "error": "Пароль має бути мінімум 6 символів."}
    if "@" not in email:
        return {"ok": False, "error": "Некоректна електронна адреса."}

    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username.strip(), email.strip().lower(), hash_password(password)),
            )
            return {"ok": True, "user_id": cursor.lastrowid}
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return {"ok": False, "error": "Користувач з таким іменем вже існує."}
        if "email" in str(e):
            return {"ok": False, "error": "Ця електронна адреса вже використовується."}
        return {"ok": False, "error": f"Помилка БД: {e}"}


def login_user(username_or_email: str, password: str) -> Optional[dict]:

    with get_connection() as conn:
        row = conn.execute(
            """SELECT id, username, email, password_hash, created_at
               FROM users
               WHERE username = ? OR email = ?""",
            (username_or_email.strip(), username_or_email.strip().lower()),
        ).fetchone()

    if row is None:
        return None

    if not verify_password(password, row["password_hash"]):
        return None

    return {
        "id":         row["id"],
        "username":   row["username"],
        "email":      row["email"],
        "created_at": row["created_at"],
    }


def get_user_by_id(user_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None



def save_forecast(
    user_id:         int,
    name:            str,
    model_type:      str,
    forecast_days:   int,
    last_value:      float,
    forecast_end:    float,
    delta_pct:       Optional[float],
    accuracy:        Optional[float],
    mape:            Optional[float],
    mae:             Optional[float],
    history_dates,
    history_values,
    forecast_dates,
    forecast_values,
    conf_int=None,   
) -> int:

    history_points = []
    for dt, val in zip(history_dates, history_values):
        date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        history_points.append({"date": date_str, "value": round(float(val), 6)})
    history_json = json.dumps(history_points, ensure_ascii=False)

    forecast_points = []
    for i, (dt, val) in enumerate(zip(forecast_dates, forecast_values)):
        date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        point = {"date": date_str, "value": round(float(val), 6)}
        if conf_int is not None:
            point["lower"] = round(float(conf_int[i, 0]), 6)
            point["upper"] = round(float(conf_int[i, 1]), 6)
        forecast_points.append(point)
    forecast_json = json.dumps(forecast_points, ensure_ascii=False)

    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO forecasts
               (user_id, name, model_type, forecast_days,
                last_value, forecast_end, delta_pct,
                accuracy, mape, mae,
                history_json, forecast_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                name.strip(),
                model_type,
                forecast_days,
                round(float(last_value), 6),
                round(float(forecast_end), 6),
                round(float(delta_pct), 4)  if delta_pct is not None else None,
                round(float(accuracy), 2)   if accuracy  is not None else None,
                round(float(mape), 4)       if mape      is not None else None,
                round(float(mae), 6)        if mae       is not None else None,
                history_json,
                forecast_json,
            ),
        )
        return cursor.lastrowid


def get_user_forecasts(user_id: int, limit: int = 100) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, name, created_at, model_type, forecast_days,
                      last_value, forecast_end, delta_pct,
                      accuracy, mape, mae
               FROM forecasts
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_forecast_detail(forecast_id: int, user_id: int) -> Optional[dict]:
   
    with get_connection() as conn:
        row = conn.execute(
            """SELECT id, name, created_at, model_type, forecast_days,
                      last_value, forecast_end, delta_pct,
                      accuracy, mape, mae,
                      history_json, forecast_json
               FROM forecasts
               WHERE id = ? AND user_id = ?""",
            (forecast_id, user_id),
        ).fetchone()

    if row is None:
        return None

    result = dict(row)
    result["history_points"]  = json.loads(result.pop("history_json"))
    result["forecast_points"] = json.loads(result.pop("forecast_json"))
    return result


def delete_forecast(forecast_id: int, user_id: int) -> bool:
    
    with get_connection() as conn:
        result = conn.execute(
            "DELETE FROM forecasts WHERE id = ? AND user_id = ?",
            (forecast_id, user_id),
        )
    return result.rowcount > 0


def rename_forecast(forecast_id: int, user_id: int, new_name: str) -> bool:
    with get_connection() as conn:
        result = conn.execute(
            "UPDATE forecasts SET name = ? WHERE id = ? AND user_id = ?",
            (new_name.strip(), forecast_id, user_id),
        )
    return result.rowcount > 0


def save_file_meta(
    user_id:      int,
    filename:     str,
    row_count:    int,
    date_column:  str,
    value_column: str,
    date_from:    str,
    date_to:      str,
    forecast_id:  Optional[int] = None,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO uploaded_files
               (user_id, forecast_id, filename, row_count,
                date_column, value_column, date_from, date_to)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, forecast_id, filename, row_count,
             date_column, value_column, date_from, date_to),
        )
    return cursor.lastrowid

def get_user_stats(user_id: int) -> dict:
    with get_connection() as conn:
        stats = conn.execute(
            """SELECT
                COUNT(*)        AS total_forecasts,
                MAX(accuracy)   AS best_accuracy,
                MIN(mape)       AS best_mape,
                AVG(accuracy)   AS avg_accuracy,
                MAX(created_at) AS last_forecast_at
               FROM forecasts
               WHERE user_id = ?""",
            (user_id,),
        ).fetchone()

        fav = conn.execute(
            """SELECT model_type, COUNT(*) AS cnt
               FROM forecasts WHERE user_id = ?
               GROUP BY model_type ORDER BY cnt DESC LIMIT 1""",
            (user_id,),
        ).fetchone()

        file_count = conn.execute(
            "SELECT COUNT(*) FROM uploaded_files WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]

    result = dict(stats) if stats else {}
    result["favorite_model"] = fav["model_type"] if fav else None
    result["total_files"]    = file_count
    return result


def db_exists() -> bool:
    return os.path.exists(DB_PATH)


if __name__ == "__main__":
    init_db()
    print(f"✅ БД ініціалізована: {DB_PATH}")

    res = register_user("test_user", "test@example.com", "password123")
    print(f"Реєстрація:     {res}")

    user = login_user("test_user", "password123")
    print(f"Логін (вірний): {user}")

    bad = login_user("test_user", "wrongpass")
    print(f"Логін (невірний пароль): {bad}")

    by_email = login_user("test@example.com", "password123")
    print(f"Логін (через email): {by_email}")

    if user:
        stats = get_user_stats(user["id"])
        print(f"Статистика: {stats}")