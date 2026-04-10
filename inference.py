import os
import asyncio
from typing import List
from openai import OpenAI
from server.environment import NewsIntelligenceEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "my_env_v4"
MAX_STEPS = 8

# 5 real tasks from your metadata.json sections
TASKS = [
    "Geopolitics",
    "Space Frontier",
    "Bharat Shakti",
    "Global Economy",
    "Future Tech"
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
    # Score must be strictly between 0.0 and 1.0
    return max(0.01, min(round(score, 3), 0.99))


def run_task(task_name: str):
    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        env = NewsIntelligenceEnv()
        articles = env.fetch_live_data(query=task_name)

        if not articles:
            log_step(1, "no-data", 0.01, True, "no articles found")
            log_end(False, 1, [0.01])
            return

        for step, article in enumerate(articles[:MAX_STEPS], start=1):
            try:
                prompt = f"Analyze this news article:\nTitle: {article['title']}\nSource: {article['source']}\nGrade: {article['grade']}\nURL: {article['url']}\n\nProvide a detailed intelligence summary."

                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an intelligence analyst. Analyze news articles and provide detailed summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150
                )

                analysis = (completion.choices[0].message.content or "no analysis").strip()

                # Use article grade as reward, clamped to (0.01, 0.99)
                reward = clamp(float(article.get("grade", 0.5)))

                rewards.append(reward)
                steps_taken = step
                done = step == len(articles[:MAX_STEPS])

                log_step(
                    step=step,
                    action=analysis[:50],
                    reward=reward,
                    done=done
                )

            except Exception as step_e:
                print(f"[DEBUG] Step {step} error: {step_e}", flush=True)
                rewards.append(0.01)
                steps_taken = step
                log_step(step, "error", 0.01, True, str(step_e)[:80])
                break

        total = sum(rewards)
        max_possible = len(rewards)
        raw_score = total / max_possible if max_possible > 0 else 0.0
        success = clamp(raw_score) > 0.1

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
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
