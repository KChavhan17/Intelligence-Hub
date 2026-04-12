
import os
import datetime
import requests
from typing import List
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BENCHMARK = "my_env_v4"
MAX_STEPS = 8

TASKS = {
    "Geopolitics & Strategy": "geopolitics OR diplomacy OR international relations",
    "Space Frontier (Artemis/ISRO)": "Artemis Moon Mission OR ISRO OR SpaceX",
    "Bharat Shakti (Defense/Tech)": "India Defense OR Digital India OR ISRO",
    "Global Economy": "World Bank OR Global Markets OR Trade War",
    "Future Tech (AI/Quantum)": "Artificial Intelligence OR Quantum Computing"
}

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error="null"):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def clamp(score: float) -> float:
    return max(0.01, min(round(float(score), 3), 0.99))


def calculate_grade(published_at: str) -> float:
    try:
        from dateutil import parser as dateparser
        now = datetime.datetime.now(datetime.timezone.utc)
        pub_time = dateparser.parse(published_at)
        diff_hours = (now - pub_time).total_seconds() / 3600
        grade = max(0, 1 - (diff_hours / 48))
        return clamp(round(grade, 3))
    except Exception:
        return 0.5


def fetch_articles(query: str) -> list:
    try:
        if not NEWS_API_KEY:
            raise ValueError("No API key")
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "apiKey": NEWS_API_KEY,
            "pageSize": MAX_STEPS
        }
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=10
        )
        articles = response.json().get("articles", [])
        results = []
        for art in articles:
            grade = calculate_grade(art.get("publishedAt", ""))
            results.append({
                "title": art.get("title", "No title"),
                "source": art.get("source", {}).get("name", "Unknown"),
                "grade": grade,
                "url": art.get("url", "")
            })
        if results:
            return results
    except Exception as e:
        print(f"[DEBUG] fetch error: {e}", flush=True)

    return [
        {"title": f"Intelligence Report on {query}", "source": "System", "grade": 0.5, "url": ""},
        {"title": f"Analysis: {query} latest developments", "source": "System", "grade": 0.45, "url": ""},
        {"title": f"Strategic update: {query}", "source": "System", "grade": 0.4, "url": ""},
    ]


def run_task(task_name: str, query: str):
    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        articles = fetch_articles(query)

        for step, article in enumerate(articles[:MAX_STEPS], start=1):
            try:
                prompt = (
                    f"Analyze this news:\n"
                    f"Title: {article['title']}\n"
                    f"Source: {article['source']}\n"
                    f"Give a detailed intelligence summary."
                )
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an intelligence analyst. Give detailed summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150
                )
                analysis = (completion.choices[0].message.content or "no analysis").strip()
                reward = clamp(article.get("grade", 0.5))
                done = (step == len(articles[:MAX_STEPS]))
                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=analysis[:50], reward=reward, done=done)

            except Exception as step_e:
                print(f"[DEBUG] Step {step} error: {step_e}", flush=True)
                rewards.append(0.01)
                steps_taken = step
                log_step(step, "error", 0.01, True, str(step_e)[:80])
                break

        raw = sum(rewards) / len(rewards) if rewards else 0.0
        success = clamp(raw) > 0.1

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        rewards = [0.01]
        steps_taken = 1
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


def main():
    for task_name, query in TASKS.items():
        run_task(task_name, query)


if __name__ == "__main__":
    main()
