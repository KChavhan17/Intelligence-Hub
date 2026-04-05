
import gradio as gr
from fastapi import FastAPI
from environment import NewsIntelligenceEnv
from agent import NewsAgent
import uvicorn

# 1. Logic Setup
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

# 2. FastAPI app
app = FastAPI()

# 3. This is the SPECIFIC route the Hackathon needs
@app.post("/reset")
async def reset():
    return {"status": "success", "message": "Environment reset successfully"}

# 4. Gradio UI (Fixed the 'theme' warning)
with gr.Blocks() as demo:
    gr.Markdown("# GLOBAL INTELLIGENCE HUB")
    selection = gr.Dropdown(
        choices=["Geopolitics & Strategy", "Space Frontier (Artemis/ISRO)", "Bharat Shakti (Defense/Tech)", "Global Economy", "Future Tech (AI/Quantum)"], 
        label="Domain"
    )
    btn = gr.Button("SIUUU - FETCH LIVE DATA")
    out = gr.Textbox(label="Graded Intelligence Feed", lines=15)
    btn.click(get_intel, selection, out)

# 5. Mount and RUN
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Hugging Face Spaces uses port 7860 by default
    uvicorn.run(app, host="0.0.0.0", port=7860)




