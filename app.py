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
        return "Translating answer to user's langauge ...✅"
    return f"Processing at step: {key} \n"
    


def predict(message, history):
    history_langchain_format = []
    # print(f"History: {pprint.pformat(history)}")
    for human, ai in history:
        if human:
            history_langchain_format.append(HumanMessage(content=human))
        elif ai:
            history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
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
chatbot = gr.Chatbot([[None,"Hello Ishwar! Thank you for purchasing the STIHL FS120 Brush Cutter. I am Suhani, your customer assistant. If you need any help regarding the FS120 as well as STIHL products, feel free to ask me."]])
ci = gr.ChatInterface(predict,chatbot=chatbot, examples=["Hi I need to service my brush cutter how can I service it?", "Please suggest a suitable accessory for trimmer grass in residential area."], title="Personal Product Assistant").queue().launch()

with gr.Blocks(fill_height=True) as demo:
    ci.render()
demo.launch(debug=True)