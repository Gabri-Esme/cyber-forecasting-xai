import sqlite3
import time
from elsapy.elssearch import ElsSearch
from elsapy.elsclient import ElsClient

def init_client(api_key):
    ''' Initializes the Elsevier API client with the provided API key. 
    Args: api_key for the Elsevier API.
    Returns: An instance of ElsClient.'''
    client = ElsClient(api_key)
    return client

def init_db(db_name="data.db"):
    '''Function to initialize the SQLite database and create necessary tables if they don't exist.
    Args:
        db_name (str): Name of the SQLite database file.
    '''
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        # Yearly
        cursor.execute("CREATE TABLE IF NOT EXISTS yearly_attacks (attack_type TEXT, year INTEGER, count INTEGER, PRIMARY KEY (attack_type, year))")
        cursor.execute("CREATE TABLE IF NOT EXISTS yearly_pats (pat_type TEXT, year INTEGER, count INTEGER, PRIMARY KEY (pat_type, year))")
        # Monthly
        cursor.execute("CREATE TABLE IF NOT EXISTS monthly_attacks (attack_type TEXT, year INTEGER, month INTEGER, count INTEGER, PRIMARY KEY (attack_type, year, month))")
        cursor.execute("CREATE TABLE IF NOT EXISTS monthly_pats (pat_type TEXT, year INTEGER, month INTEGER, count INTEGER, PRIMARY KEY (pat_type, year, month))")
        conn.commit()

# Save yearly attacks
def transform_and_insert_ya(conn, attack_type, year, count):
    '''Function to insert or replace yearly attack counts into the database.
    Args:
        conn: SQLite database connection object.
        attack_type (str): Type of attack.
        year (int): Year of the data.
        count (int): Count of attacks for the given type and year.
    '''
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO yearly_attacks (attack_type, year, count)
        VALUES (?, ?, ?)
    """, (attack_type, year, count))
    conn.commit()

# Save yearly pats    
def transform_and_insert_yp(conn, pat_type, year, count):
    '''Function to insert or replace yearly PAT counts into the database.
    Args:
        conn: SQLite database connection object.
        pat_type (str): Type of PAT.
        year (int): Year of the data.
        count (int): Count of PATs for the given type and year.
    '''
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO yearly_pats (pat_type, year, count)
        VALUES (?, ?, ?)
    """, (pat_type, year, count))
    conn.commit()

# Save monthly attacks      
def transform_and_insert_ma(conn, attack_type, year, month, count):
    '''Function to insert or replace monthly attack counts into the database.
    Args:
        conn: SQLite database connection object.
        attack_type (str): Type of attack.
        year (int): Year of the data.
        month (int): Month of the data.
        count (int): Count of attacks for the given type, year, and month.
    '''
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO monthly_attacks (attack_type, year, month, count)
        VALUES (?, ?, ?, ?)
    """, (attack_type, year, month, count))
    conn.commit()
              
# Save monthly pats
def transform_and_insert_mp(conn, pat_type, year, month, count):
    '''Function to insert or replace monthly PAT counts into the database.
    Args:
        conn: SQLite database connection object.
        pat_type (str): Type of PAT.
        year (int): Year of the data.
        month (int): Month of the data.
        count (int): Count of PATs for the given type, year, and month.
    '''
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO monthly_pats (pat_type, year, month, count)
        VALUES (?, ?, ?, ?)
    """, (pat_type, year, month, count))
    conn.commit()


def get_counts(conn, api_key, years, list, mode='yearly', is_pat=False):
    '''Function to query the Elsevier API for counts of attacks or PATs based on the provided configuration.
    Args:
        conn: SQLite database connection object.
        api_key (str): API key for the Elsevier API.
        years (list): List of years to query.
        list (list): List of dictionaries containing target and accepted keywords.
        mode (str): Mode of querying, either 'yearly' or 'monthly'.
        is_pat (bool): Flag indicating if the query is for PATs (True) or attacks (False).
    Returns:
        list: A list of dictionaries containing the results of the queries.
    '''
    client = init_client(api_key)

    # Switch between querying for monthly or yearly hits 
    all_results = []

    for item in list:
        category = item['target']
        keywords = item['accepted']
        kw_query = " OR ".join([f'"{kw}"' for kw in keywords])

        for year in years:
            months_run = range(1, 13) if mode == 'monthly' else [None]

            for month in months_run:
                if mode == "monthly":
                    date_query = f"PUBYEAR IS {year} AND MONTH IS {month}"
                else:
                    date_query = f"PUBYEAR IS {year}"
                
                query = f"TITLE-ABS-KEY({kw_query}) AND {date_query}"
                search = ElsSearch(query, 'scopus')

                count = 0
                try:
                    search.execute(client, get_all=False)
                    count = int(search.tot_num_res or 0)
                    time_label = f"{year}-{month:02d}" if month else f"{year}"
                    print(f"[{mode.upper()}] {category} | {time_label} | Hits: {count}")
                    if mode == 'monthly':
                        if is_pat:
                            transform_and_insert_mp(conn, category, year, month, count)
                        else:
                            transform_and_insert_ma(conn, category, year, month, count)
                    else:
                        if is_pat:
                            transform_and_insert_yp(category, year, count)
                        else:
                            transform_and_insert_ya(conn, category, year, count)
                    time.sleep(0.8) 

                except Exception as e:
                    print(f"Error for category {category} at {year}-{month}: {e}. Recording as 0.")
                    time.sleep(5)
                    
                result = {
                    "category": category,
                    "year": year,
                    "month": month, # Will be None if yearly
                    "hits": count
                }
                all_results.append(result)
   
    return all_results
    