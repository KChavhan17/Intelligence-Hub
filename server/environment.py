
import requests
import datetime
import os
from dateutil import parser

class NewsIntelligenceEnv:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2/everything"

    def calculate_grade(self, published_at):
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            pub_time = parser.parse(published_at)
            diff_hours = (now - pub_time).total_seconds() / 3600
            grade = max(0, 1 - (diff_hours / 48))
            return round(grade, 3)
        except:
            return 0.5

    def fetch_live_data(self, query):
        if not self.api_key:
            return [
                {"title": f"Report: {query}", "source": "System", "grade": 0.5, "url": ""},
                {"title": f"Analysis: {query}", "source": "System", "grade": 0.4, "url": ""},
                {"title": f"Update: {query}", "source": "System", "grade": 0.3, "url": ""}
            ]
        try:
            params = {
                'q': query,
                'sortBy': 'publishedAt',
                'apiKey': self.api_key,
                'pageSize': 5
            }
            response = requests.get(self.base_url, params=params)
            articles = response.json().get('articles', [])
            results = []
            for art in articles:
                grade = self.calculate_grade(art['publishedAt'])
                results.append({
                    "title": art['title'],
                    "source": art['source']['name'],
                    "grade": grade,
                    "url": art['url']
                })
            return results if results else [
                {"title": f"Report: {query}", "source": "Fallback", "grade": 0.5, "url": ""}
            ]
        except Exception as e:
            return [
                {"title": f"Report: {query}", "source": "Fallback", "grade": 0.5, "url": ""}
            ]
