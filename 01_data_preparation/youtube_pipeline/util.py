import calendar
import json
from datetime import datetime
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import sqlite3

class YouTubeClient: 
    def __init__(self, api_key):
        self.service = build("youtube", "v3", developerKey=api_key)

    def search_videos(self, query, published_after=None, published_before=None, max_results=50):
        '''Query Builder for YouTube Data
        This module defines a function to query the YouTube Data API for videos matching specific keywords,
        Args: - query: The search query string, e.g keywords
            - publishedAfter: ISO 8601 format date for beginning of the search period
            - publishedBefore: ISO 8601 format date for the end of the search period
            - max_results: Maximum number of search results to return (default 50)
        Returns: A dictionary containing:
                - count: Number of videos found
                - views: Total view count across all videos
                - likes: Total like count across all videos
                - comments: Total comment count across all videos
        '''
        search_response = self.service.search().list(
            q=query,
            type="video",
            part="id",
            maxResults=max_results,
            publishedAfter=published_after,
            publishedBefore=published_before
        ).execute()

        video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]

        if not video_ids:
            return {"count": 0, "views": 0, "likes": 0, "comments": 0}
        
        video_response = self.service.videos().list(
            part="statistics",
            id=",".join(video_ids)
        ).execute()

        total_views = total_likes = total_comments = 0
        for video in video_response.get('items', []):
            stats = video.get('statistics', {})
            total_views += int(stats.get('viewCount') or 0)
            total_likes += int(stats.get('likeCount') or 0)
            total_comments += int(stats.get('commentCount') or 0)

        print(f"Query: {query}, Videos: {len(video_ids)}, Views: {total_views}, Likes: {total_likes}, Comments: {total_comments}")
        return {
            "count": len(video_ids),
            "views": total_views,
            "likes": total_likes,
            "comments": total_comments,
        }

class YouTubeETL:
    def __init__(self, db, api, config):
        self.db = db
        self.api = api
        self.config = config

    def convert_to_iso(self, year, month):
        '''Function to convert year and month to ISO 8601 format for YouTube API
        Args: - year: The year as an integer
            - month: The month as an integer (1-12)
        Returns: A tuple containing the start and end dates in ISO 8601 format'''
        start_date = datetime(year, month, 1).isoformat("T") + "Z"
        last_day = calendar.monthrange(year, month)[1]
        end_date = datetime(year, month, last_day).isoformat("T") + "Z"
        return start_date, end_date

    def run(self):
        '''Main ETL function to extract data from YouTube API and load it into the database'''
        existing = self.db.get_existing_records()

        for country in self.config['countries']:
            for year in self.config['years']:
                if year == 2026: #Added to avoid breaking as the year hasn't finished. These lines should be removed or replaced with timeframe reqs.
                    cutoff = 9
                else:
                    cutoff = 13

                for month in range(1, cutoff):
                    if (country, year, month) in existing:
                        print(f"Skipping {country} {year}-{month:02d}, already in DB")
                        continue
                    
                    query = f"{country} ({'|'.join(self.config['keywords'])})"
                    print(f"Querying for {query} {year}-{month:02d}")
                    
                    try:
                        # (Put date into ISO format for YouTube API)
                        start_date, end_date = self.convert_to_iso(year, month)
                        data = self.api.search_videos(query, published_after=start_date, published_before=end_date)
                        self.db.save_record(country, year, month, data)
                    except Exception as e:
                        print(f"Error for {query} {year}-{month:02d}: {e}")
                        if 429 or 403 in HttpError:  # Quota exceeded or forbidden
                            print("Quota exceeded, stopping further requests.")
                            return
                    time.sleep(0.5)

class YouTubeDB:
    '''Creates a SQLite database to store YouTube data and provides methods to interact with it.'''
    def __init__(self, db_name):
        self.db_name = db_name
        self._init_db()
    
    def _init_db(self, table_name='youtube_data'):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    country TEXT,
                    year INTEGER,
                    month INTEGER,
                    count INTEGER,
                    views INTEGER,
                    likes INTEGER,
                    comments INTEGER,
                    PRIMARY KEY (country, year, month)
                )
            """)

    def get_existing_records(self):
        ''' Returns a set of (country, year, month) tuples for existing records '''
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute("SELECT country, year, month FROM youtube_data")
            return {row for row in cursor.fetchall()}
        
    def save_record(self, country, year, month, data):
        ''' Saves a record to the database, ignoring existing one if primary key matches '''
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO youtube_data VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                country,
                year,
                month,
                data['count'],
                data['views'],
                data['likes'],
                data['comments']
            ))
            conn.commit()
