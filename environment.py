import gymnasium as gym
from gymnasium import spaces
import requests
import datetime
import os
from dateutil import parser

class NewsIntelligenceEnv(gym.Env):
    def __init__(self):
        super(NewsIntelligenceEnv, self).__init__()
        # This line looks for your 'Secret' on Hugging Face
        self.api_key = os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2/everything"

    def calculate_grade(self, published_at):
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            pub_time = parser.parse(published_at)
            diff_hours = (now - pub_time).total_seconds() / 3600
            # 0.0 to 1.0 Grade based on 48-hour freshness
            grade = max(0, 1 - (diff_hours / 48))
            return round(grade, 3)
        except:
            return 0.0

    def fetch_live_data(self, query):
        if not self.api_key:
            return [{"title": "Error: NEWS_API_KEY not found in Secrets.", "source": {"name": "System"}}]
        
        params = {'q': query, 'sortBy': 'publishedAt', 'apiKey': self.api_key, 'pageSize': 5}
        response = requests.get(self.base_url, params=params).json()
        articles = response.get('articles', [])
        
        results = []
        for art in articles:
            grade = self.calculate_grade(art['publishedAt'])
            results.append({
                "title": art['title'],
                "source": art['source']['name'],
                "grade": grade,
                "url": art['url']
            })
        return results
