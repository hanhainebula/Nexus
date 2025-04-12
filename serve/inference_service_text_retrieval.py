import os
import sys
import json
sys.path.append('/root/test/Nexus')
from typing import List
import gradio as gr
import time
import faiss
import numpy as np
import argparse
import yaml
from Nexus.abc.inference.inference_engine import InferenceEngine
from Nexus.inference.embedder.text_retrieval import BaseEmbedderInferenceEngine
from Nexus.inference.reranker.text_retrieval import BaseRerankerInferenceEngine
from Nexus import AbsInferenceArguments
import pandas as pd
import numpy as np
from tqdm import tqdm


def load(path):
    res = []
    with open(path,'r', encoding='utf-8') as f:
        for line in tqdm(f):
            obj = json.loads(line)
            res.append(obj['title']+ '\n' +obj['content'])
    return res

corpus_path = '/data2/home/angqing/code/Nexus/datas/81cn.jsonl'
faiss_path = '/data2/home/angqing/code/Nexus/datas/81cn.idx'
embedder_config = '/data2/home/angqing/code/Nexus/serve/config/embedder.json'
reranker_config = '/data2/home/angqing/code/Nexus/serve/config/reranker.json'


index = faiss.read_index(faiss_path)
corpus = load(corpus_path)


embedder_args = AbsInferenceArguments.from_json(embedder_config)
reranker_args = AbsInferenceArguments.from_json(reranker_config)
retrieval_inference_engine = BaseEmbedderInferenceEngine(embedder_args)
rank_inference_engine = BaseRerankerInferenceEngine(reranker_args)

def retrieve(user_query):

    global search_topk, rerank_topk

    print(user_query)
    # Step 1: Get query embedding using the embedder
    query_embedding = retrieval_inference_engine.inference(user_query, normalize=True)
    
    print("query_embedding shape:", query_embedding.shape)

    D, I = index.search(np.expand_dims(query_embedding, axis = 0), k=search_topk)  # `final_topk` is the number of results you want
    query_answer_pairs = [(user_query, corpus[i]) for i in I[0]]
    
    rerank_results = rank_inference_engine.inference(query_answer_pairs)
    ranked_results = [
        {"score": score, "id": I[0][i], "corpus": corpus[I[0][i]]}
        for i, score in enumerate(rerank_results)
    ]

    # Sort the results based on the score (in descending order, so higher scores are ranked first)
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    ranked_results=ranked_results[:rerank_topk]
    # Collect the results in the desired format
    results = []
    final_result = []
    for i, result in enumerate(ranked_results):
        results.append(f"Rank {i+1}: Score = {result['score']}, Doc ID = {result['id']}")
        final_result.append({
            "id": result["id"],
            "corpus": result['corpus'],
            "score":result['score']
        })
    
    return results, final_result

def chat_interface(user_query, chat_history):
    # Get retrieval results based on the query
    stages_results, final_results = retrieve(user_query)

    # Prepare stage results with collapsible sections for each stage
    stage_messages = [f"Search Results for Query: {user_query}"]
    for i, result in enumerate(stages_results):
        stage_messages.append(f"<details><summary>Rank {i+1}</summary><p>{result}</p></details>")

    # Update chat history with results
    chat_history.append({"role": "user", "content": f"User query: {user_query}"})
    chat_history.append({"role": "assistant", "content": "\n".join(stage_messages)})

    # Update textboxes with final retrieved items
    textbox_values = [f"Doc-{item['id']}:\n {item['corpus']}" for item in final_results]
    textbox_update_list = [gr.update(value=value, visible=True) for value in textbox_values]

    return chat_history, *textbox_update_list

def button_click_fns(i):
    def fn(user_history, show_items):
        # Simulate the user clicking on an item, appending the ID to the history
        user_history = " ".join([user_history, str(show_items[int(i)]["id"])]).strip()
        return user_history
    return fn

def change_user_fns(user_query):
    # Clear the chat history and add a new message
    stages_results, final_results = retrieve(user_query)
    stage_messages = [f"Search Results for User {user_query}: "]
    for i, result in enumerate(stages_results):
        stage_messages.append(f"<details><summary>Stage {i+1}</summary><p>{result}</p></details>")

    # Update chat history with results
    chat_history = []
    chat_history.append({"role": "assistant", "content": f"🤗 Hello, this is query {user_query}"})
    chat_history.append({"role": "assistant", "content": "\n".join(stage_messages)})

    user_query = ""  # Reset the previous query history (optional)
    send_btn_update = gr.update(interactive=True)  # Set send button to interactive

    # Update textboxes with final results
    textbox_values = [f"Doc-{item['id']}: {item['corpus']}" for item in final_results]
    textbox_update_list = [gr.update(value=value, visible=True) for value in textbox_values]

    click_bttn_list = [gr.update(visible=True) for i in range(len(final_results))]

    return chat_history, user_query, final_results, send_btn_update, *textbox_update_list, *click_bttn_list


# Gradio界面
if __name__ == "__main__":

    search_topk = 40
    rerank_topk = 20
    
    init_result = [{"id": -1, "corpus": None} for i in range(rerank_topk)]
    

    with gr.Blocks() as demo:
        gr.Markdown("## Text Retrieval System")

        show_items = gr.State(init_result)

        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### User Input")
                user_query = gr.Textbox(label="Enter your query", interactive=True)
                send_btn = gr.Button("Search", interactive=True)
                chatbot = gr.Chatbot(type="messages", value=[{"role": "assistant", "content": "🤗 Hello! Please enter your query."}])

            with gr.Column(scale=4):
                gr.Markdown("### Retrieval Results")
                result_textboxes = []
                for i in range(rerank_topk):
                    with gr.Row():
                        textbox = gr.Textbox(value=f"-", visible=False, container=False, scale=6)
                        result_textboxes.append(textbox)

        # Define button click action
        send_btn.click(fn=chat_interface, inputs=[user_query, chatbot], outputs=[chatbot, *result_textboxes])

    # Launch the Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=7777)
