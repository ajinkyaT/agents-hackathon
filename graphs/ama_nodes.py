from typing import Annotated, Literal, Sequence, TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolInvocation
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from tools.retriever_tool import RetrieverTool
from utils.ingest_data import VectorDB
from utils.convert_img import get_image_base64
from utils.get_accessories import get_accessories_table
from prompts.rag_prompt import rag_prompt, agent_system_prompt, grader_prompt, hallucination_grade_prompt, answer_grade_prompt, translate_answer_prompt, parse_langcode_prompt, image_prompt
from entities.node_entities import GradeDocuments, GradeHallucinations, GradeAnswer, ParseLangCode, ImageTags
import pprint

### Edges
grass_cutter_vector_db = VectorDB(doc_store_path="./data/stihl")
grass_cutter_tool = RetrieverTool(grass_cutter_vector_db, "grass_cutter_retriever", "Search and retrieve information about grass cutter/power tiller. Use it to retrieve information like instructions of using it, spare parts, available accessories, new products available etc")
tools = [grass_cutter_tool.get_tool()]

from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(tools)

def decide_img_branch(state):
    """
    Determines whether the divert to get_img_reco or agent.

    Args:
        state (messages): The current state

    Returns:
        str: Binary 'agent' or 'img_reco'
    """
    init_msg = state['messages'][-1]
    if isinstance(init_msg, HumanMessage):
        if init_msg.additional_kwargs.get('image'):
            return 'img_reco'
        else:
            return 'agent'
    else:
        return 'agent'
    
def get_img_reco(state):
    """
    Determines whether the divert to get_img_recommendation or agent.

    Args:
        state (messages): The current state

    Returns:
        str: Binary 'agent' or 'img_reco'
    """
    img_msg = state['messages'][-1]
    image_data = get_image_base64(img_msg.additional_kwargs["image"])
    llm = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    structured_llm_grader = llm.with_structured_output(ImageTags)
    img_parser = image_prompt | structured_llm_grader
    output = img_parser.invoke(image_data)
    img_tags = output.extracted_tags
    if not img_tags:
        return {'next_step': "END"}
    img_decription = output.image_description
    query = f"Suggest me product accessories from given documents where one or more Tags in products contains in the list {img_tags} \n Start answer by mentioning it seems like your lawn seems to have grass type mentioned in the tags list. Do not mention the word tags it is  only for your reference."
    # mocking API request
    accessories = get_accessories_table() + f"\n source_file: product_accessories.csv"
    context_docs = AIMessage(content=accessories)
    print(f"identified tags: {img_tags}")
    return {'messages': [context_docs], 'query': query, 'img_description': img_decription, "lang_code":'en'}
    
    

def parse_input_question(state):
    """
    Returns translated question to ask agent in English and the ISO language code of user query.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
        tuple: (translated english question, ISO Language code) 
    """
    question = state['messages'][-1]
    # llm = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", streaming=False)
    # llm = ChatGroq(temperature=0, model="gemma2-9b-it", streaming=False)
    structured_llm_grader = llm.with_structured_output(ParseLangCode)
    lang_parser = parse_langcode_prompt | structured_llm_grader
    translated_question, langcode = lang_parser.invoke(question)
    translated_query = HumanMessage(content=translated_question[1])
    return {'messages': [translated_query], 'query': translated_query, 'lang_code': langcode[1]}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: Binary 'yes' or 'no'
    """

    # Prompt
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grader_prompt.template),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    # llm = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    # llm = ChatGroq(temperature=0, model="llama3-70b-8192", streaming=False)
    llm = ChatGroq(temperature=0, model="gemma2-9b-it", streaming=False)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["query"]
    documents = state["messages"][-1]

    scored_result = retrieval_grader.invoke({"question": question, "document": documents})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"



def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    messages = state["messages"]
    # Ignore tool call messages
    messages = [m for m in messages if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.response_metadata.get('finish_reason') != 'tool_calls')]
    # model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
    # model = ChatGroq(temperature=0.1, streaming=False, model="llama3-70b-8192")
    model = ChatGroq(temperature=0, model="gemma2-9b-it", streaming=False)
    model = model.bind_tools(tools)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", agent_system_prompt.template),
            MessagesPlaceholder("chat_history"),
        ]
    )

    question = messages[-1].content
    print(f"--- CALL AGENT WITH question: {question} --- \n")
    agent_retriever = qa_prompt | model
    response = agent_retriever.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["query"]

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question, only output the improved question without any explanation on how you reasoned about it.
    Improved Question: """,
        )
    ]
    new_q_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("chat_history"),
            msg[0]
        ]
    )
    print(f"--- CALL AGENT WITH question: {question} --- \n")

    # Grader
    print(f"---BEFORE TRANSFORMED QUERY: {question}---")
    # model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    # model = ChatGroq(temperature=0.5, model="llama3-70b-8192", streaming=False)
    model = ChatGroq(temperature=0, model="gemma2-9b-it", streaming=False)
    new_q = new_q_prompt | model
    response = new_q.invoke(state["messages"])
    print(f"---TRANSFORMED QUERY: {response.content}---")
    return {"messages": [response], "query": response.content}

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    if state['next_step'] == 'END':
        ai_message = AIMessage(content='It seems like the provided image does not contain lawn or grass, please provide a relevant image.')
        return {'context': None, 'messages': [ai_message]}
    messages = state["messages"]
    question = state['query']
    docs = messages[-1]
    print(f"Got query for generation: {question}")

    # Prompt
    prompt = rag_prompt

    # LLM
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0, streaming=False)
    # llm = ChatGroq(temperature=0, model="gemma2-9b-it", streaming=False)


    # Chain
    rag_chain = prompt | llm

    # Run
    response = rag_chain.invoke({"documents": docs.content, "question": question})
    return {"messages": [response], "generation": response.content, "context":docs}

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["query"]
    documents = state["context"]
    generation = state["generation"]
    if not documents:
        return 'useful'
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
    # llm = ChatGroq(model_name="llama3-70b-8192", temperature=0, streaming=False)
    llm = ChatGroq(temperature=0, model="gemma2-9b-it", streaming=False)
    structured_llm_hallu_grader = llm.with_structured_output(GradeHallucinations)
    structured_llm_ans_grader = llm.with_structured_output(GradeAnswer)


    # Prompt
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", hallucination_grade_prompt.template),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grade_prompt.template),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])

    hallucination_grader = hallucination_prompt | structured_llm_hallu_grader
    answer_grader = answer_prompt | structured_llm_ans_grader

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def translate_answer(state):
    """
    Returns translated answer in the ISO language code of user query.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
        str: 
    """
    eng_answer = state['messages'][-1]
    lang_code = state["lang_code"]
    if lang_code == 'en':
        return {'messages': [eng_answer]}
    # llm = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", streaming=False)
    # llm = ChatGroq(temperature=0, model="gemma2-9b-it", streaming=False)
    print(f"---Translating final answer to lang code: {lang_code}---")
    answer_translator = translate_answer_prompt | llm
    translated_answer = answer_translator.invoke({'input_text': eng_answer, 'lang_code': lang_code})
    print(f"---translated final answer: {translated_answer.content}---")
    return {'messages': [translated_answer]}