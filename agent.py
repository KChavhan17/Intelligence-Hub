class NewsAgent:
    def rank(self, data):
        # Keeps the 1.0 Grade news at the top
        return sorted(data, key=lambda x: x['grade'], reverse=True)
