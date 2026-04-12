import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from environment import NewsIntelligenceEnv
from agent import NewsAgent
import uvicorn

# Logic Setup
env = NewsIntelligenceEnv()
agent = NewsAgent()

def get_intel(section):
    query_map = {
        "Geopolitics & Strategy": "geopolitics OR diplomacy OR international relations",
        "Space Frontier (Artemis/ISRO)": "Artemis Moon Mission OR ISRO Gaganyaan OR SpaceX",
        "Bharat Shakti (Defense/Tech)": "India Defense News OR Digital India OR ISRO",
        "Global Economy": "World Bank OR Global Markets OR Trade War",
        "Future Tech (AI/Quantum)": "Artificial Intelligence OR Quantum Computing breakthroughs"
    }
    raw = env.fetch_live_data(query_map[section])
    final = agent.rank(raw)
    output = f"LIVE INTELLIGENCE: {section.upper()}\n" + "="*50 + "\n"
    for item in final:
        output += f"[{item['title']}] GRADE: {item['grade']} SRC: {item['source']}\n"
    return output

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Intelligence Hub Running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/reset")
async def reset(data: dict = None):
    return {
        "status": "success",
        "observation": {"text": "Environment ready", "echoed_message": ""},
        "message": "Reset successful"
    }

@app.post("/step")
async def step(data: dict = None):
    action = data.get("action", "") if data else ""
    reward = min(0.99, max(0.01, round(len(str(action)) * 0.005, 3)))
    return {
        "observation": {"text": f"Received: {action[:50]}", "echoed_message": action[:50]},
        "reward": reward,
        "done": False,
        "info": {}
    }

@app.get("/state")
async def state():
    return {
        "status": "running",
        "tasks": [
            "Geopolitics & Strategy",
            "Space Frontier (Artemis/ISRO)",
            "Bharat Shakti (Defense/Tech)",
            "Global Economy",
            "Future Tech (AI/Quantum)"
        ]
    }

@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {"id": "geopolitics", "name": "Geopolitics & Strategy", "difficulty": "easy"},
            {"id": "space", "name": "Space Frontier (Artemis/ISRO)", "difficulty": "medium"},
            {"id": "defense", "name": "Bharat Shakti (Defense/Tech)", "difficulty": "medium"},
            {"id": "economy", "name": "Global Economy", "difficulty": "medium"},
            {"id": "tech", "name": "Future Tech (AI/Quantum)", "difficulty": "hard"}
        ]
    }

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# GLOBAL INTELLIGENCE HUB")
    selection = gr.Dropdown(
        choices=["Geopolitics & Strategy", "Space Frontier (Artemis/ISRO)", "Bharat Shakti (Defense/Tech)", "Global Economy", "Future Tech (AI/Quantum)"],
        label="Domain"
    )
    btn = gr.Button("SIUUU - FETCH LIVE DATA")
    out = gr.Textbox(label="Graded Intelligence Feed", lines=15)
    btn.click(get_intel, selection, out)

# Mount Gradio on /ui path, keep API on root
app = gr.mount_gradio_app(app, demo, path="/ui")


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)
if __name__ == "__main__":
    main()
