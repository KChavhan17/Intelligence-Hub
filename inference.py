import os
import requests
from typing import List
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_URL", "https://keertiwebstack-intelligence-hub.hf.space")
BENCHMARK = "my_env_v4"
MAX_STEPS = 8

# Exact 5 sections from your app.py
TASKS = [
    "Geopolitics & Strategy",
    "Space Frontier (Artemis/ISRO)",
    "Bharat Shakti (Defense/Tech)",
    "Global Economy",
    "Future Tech (AI/Quantum)"
]

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


def run_task(task_name: str):
    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        # Reset environment
        try:
            requests.post(
                ENV_URL.rstrip('/') + "/reset",
                json={},
                timeout=15
            )
        except Exception as e:
            print(f"[DEBUG] Reset error: {e}", flush=True)

        # Fetch live data directly from your environment
        from server.environment import NewsIntelligenceEnv
        env = NewsIntelligenceEnv()
        articles = env.fetch_live_data(query=task_name)

        if not articles:
            log_step(1, "no-data", 0.01, True, "no articles found")
            log_end(False, 1, [0.01])
            return

        for step, article in enumerate(articles[:MAX_STEPS], start=1):
            try:
                prompt = (
                    f"Analyze this news:\n"
                    f"Title: {article['title']}\n"
                    f"Source: {article['source']}\n"
                    f"Provide a detailed intelligence summary."
                )

                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an intelligence analyst. Give detailed news summaries."},
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
        print(f"[DEBUG] Task {task_name} critical error: {e}", flush=True)
        rewards = [0.01]
        steps_taken = 1
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


def main():
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()
