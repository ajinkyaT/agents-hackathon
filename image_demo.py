from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
from graphs.rag_graph import agentic_rag_graph
import pprint
import re    


def predict(message, history):
    print(f"received input: {pprint.pformat(message)}")
    if message['text']:
        message_text = message['text']
    else: message_text = 'no image caption provided'
    if message['files']:
        image_file = message['files'][0]
        return "got image"                
    return message_text
# gr.ChatInterface(predict).queue().launch()
chatbot = gr.Chatbot([[None,"Please upload example image for demo"]])
ci = gr.ChatInterface(predict,chatbot=chatbot, examples=[{"text":"lawn image", "files":["data/images/lawn_image.jpg"]}], title="Image Product Assistant", multimodal=True).launch()

with gr.Blocks(fill_height=True) as demo:
    ci.render()
demo.launch(debug=True)