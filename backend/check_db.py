import sqlite3
conn = sqlite3.connect('Simulator/app_data_generator_for_offline/output/simulator_db.sqlite')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables in DB:')
for t in tables:
    name = t[0]
    try:
        count = conn.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
        print(f'  {name:<40} : {count:,} rows')
    except Exception as e:
        print(f'  {name:<40} : ERROR {e}')
conn.close()
