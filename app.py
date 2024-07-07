from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
from graphs.rag_graph import agentic_rag_graph
import pprint
import re


llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

def get_key_output(key, value):
    if key == 'agent':
        reason = value['messages'][0].response_metadata['finish_reason']
        if reason == 'stop':
            return 'Calling agent... ✅\n'
        elif reason == 'tool_calls':
            return 'Fetching relevant information...✅\n'
    elif key == 'retrieve':
        pattern = r"Source File: (.*)"
        matches = re.findall(pattern, value['messages'][0].content)
        return f"Looked up: {','.join(matches)} ... ✅\n"
    elif key == 'generate':
        return "Check hallucinations and generate answer ...✅\n"
    elif key == 'rewrite':
        return f"Rewriting user question as: {value['messages'][0].content} \n"
    elif key == 'translate_answer':
        return "Translating answer to user's langauge ...✅ \n"
    return f"Processing at step: {key} ...✅\n"
    


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        if human:
            if '.jpg' not in human and '.png' not in human and '.jpeg' not in human:
                history_langchain_format.append(HumanMessage(content=human[0]))
        if ai:
            history_langchain_format.append(AIMessage(content=ai[0]))
    if message['files']:
        img_msg = HumanMessage(content=message['text'], additional_kwargs={'image':message['files'][0]})
        history_langchain_format.append(img_msg)
    else: history_langchain_format.append(HumanMessage(content=message['text']))
    inputs = {"messages": history_langchain_format}
    # return agentic_rag_graph.invoke(inputs)['messages'][-1].content
    partial_message = ""
    final_answer = ''
    for output in agentic_rag_graph.stream(inputs,  config= {"recursion_limit": 25}, stream_mode="updates",):
        for key, value in output.items():
            partial_message += get_key_output(key,value)
            if key == 'translate_answer' and value['messages'][0].response_metadata['finish_reason'] == 'stop':
                 final_answer = value['messages'][0].content
            yield  partial_message
    yield final_answer

# gr.ChatInterface(predict).queue().launch()
first_message = [[None,"Hello Ishwar! Thank you for purchasing the STIHL FS120 Brush Cutter. I am Kimaya, your customer assistant. I can help you make best use of your product. Can you provide me picture of your lawn? Or if you need any help regarding the product as well as STIHL products, feel free to ask me."]]
chatbot = gr.Chatbot(first_message)
examples=[{"text":"lawn image", "files":["data/images/lawn_image.jpg"]},{"text": "How can I service my machine?"}]
ci = gr.ChatInterface(predict,chatbot=chatbot, examples=examples, title="Personal Product Assistant", multimodal=True).queue().launch()

with gr.Blocks(fill_height=True) as demo:
    ci.render()
demo.launch(debug=True)