import os
import sqlite3, holidays
from datetime import date
from dotenv import load_dotenv

load_dotenv("01_data_preparation/secrets.env")
DB_NAME = os.getenv("DB_PATH")

COUNTRIES = ['US','GB','CA','AU','UA','RU','FR','DE','BR','CN','JP','PK',
           'KP','KR','IN','TW','NL','ES','SE','MX','IR','IL','SA','SY',
           'FI','IE','AT','NO','CH','IT','MY','EG','TR','PT','PS','AE']

YEARS = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]


conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute("""
               CREATE TABLE IF NOT EXISTS holidays_counter (
               country TEXT,
               year INTEGER,
               month INTEGER,
               holiday_count INTEGER,
               PRIMARY KEY (country, year, month)
               )
               """)

for country in COUNTRIES:
    try:
        h = holidays.country_holidays(country, years=YEARS)
    except Exception:
        print(f"Country {country} not supported. Skipped")
        continue 

    for year in YEARS:
        yh = {date: name for date, name in h.items() if date.year == year}

        for month in range(1, 13):
            h_count = sum(1 for date in yh if date.month == month)

            cursor.execute("""
                        INSERT OR REPLACE INTO holidays_counter
                           (country, year, month, holiday_count)
                           VALUES( ?, ?, ?, ?)
                           """, (country, year, month, h_count))
            print(f"{country} {year}-{month:02d}: {h_count} holidays")

conn.commit()
conn.close()