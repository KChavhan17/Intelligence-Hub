import gradio as gr
from environment import NewsIntelligenceEnv
from agent import NewsAgent

env = NewsIntelligenceEnv()
agent = NewsAgent()

def get_intel(section):
    query_map = {
        "Geopolitics & Strategy": "geopolitics OR diplomacy OR international relations",
        "Space Frontier (Artemis/ISRO)": "Artemis Moon Mission OR ISRO Gaganyaan OR SpaceX",
        "Bharat Shakti (Defense/Tech)": "India Defense News OR Digital India OR ISRO",
        "Global Economy": "World Bank OR Global Markets OR Trade War",
        "Future Tech (AI/Quantum)": "Artificial Intelligence OR Quantum Computing breakthrough"
    }
    
    raw = env.fetch_live_data(query_map[section])
    final = agent.rank(raw)
    
    output = f"📡 LIVE INTELLIGENCE: {section.upper()}\n" + "="*50 + "\n\n"
    for item in final:
        output += f"📰 {item['title']}\n⭐ GRADE: {item['grade']} (0.0-1.0)\n🏢 SRC: {item['source']}\n🔗 LINK: {item['url']}\n\n"
    return output

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🚀 GLOBAL INTELLIGENCE HUB")
    gr.Markdown("### Secure 0.0-1.0 Graded News System")
    
    selection = gr.Dropdown([
        "Geopolitics & Strategy", 
        "Space Frontier (Artemis/ISRO)", 
        "Bharat Shakti (Defense/Tech)", 
        "Global Economy", 
        "Future Tech (AI/Quantum)"
    ], label="Intelligence Domain")
    
    btn = gr.Button("SIUUU - FETCH LIVE DATA", variant="primary")
    out = gr.Textbox(label="Graded Intelligence Feed", lines=18)
    
    btn.click(get_intel, selection, out)

demo.launch()
