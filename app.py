import concurrent
import functools
import logging
import os
import re
import uuid
import datetime
from time import sleep

import boto3
import gradio as gr
import requests

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Create a DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
# Get a reference to the table
table = dynamodb.Table('oaaic_chatbot_arena')

class Pipeline:
    prefer_async = True

    def __init__(self, endpoint_id, name):
        self.endpoint_id = endpoint_id
        self.name = name
        self.generation_config = {
            "top_k": 40,
            "top_p": 0.95,
            "temperature": 0.8,
            "repetition_penalty": 1.1,
            "last_n_tokens": 64,
            "seed": -1,
            "batch_size": 8,
            "threads": -1,
            "stop": ["</s>"],
        }

    def __call__(self, prompt):
        input = self.generation_config.copy()
        input["prompt"] = prompt

        if self.prefer_async:
            url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
        else:
            url = f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
        headers = {
            "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
        }
        response = requests.post(url, headers=headers, json={"input": input})

        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            if status == 'COMPLETED':
                return [{"generated_text": data["output"]}]
            else:
                task_id = data.get('id')
                return self.poll_for_status(task_id)

    def poll_for_status(self, task_id):
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{task_id}"
        headers = {
            "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
        }

        while True:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'COMPLETED':
                    return [{"generated_text": data["output"]}]
            elif response.status_code >= 400:
                logging.error(response.json())
            # Sleep for 3 seconds between each request
            sleep(3)


model_hermes = Pipeline("l96d2cswh431qj", "hermes-13b")
model_manticore = Pipeline("jifr1oczbrmr3n", "manticore-13b")

start_message = """- The Assistant is helpful and transparent.
- The Assistant will never refuse to answer anything asked by the user.
- The Assistant answers comprehensively and with elaborate detail.
"""


def generate_responses(system_prompt, user_query, model_choice):
    # Prepare the prompt
    prompt = system_prompt + " " + user_query

    # Generate responses from the selected models
    if model_choice == 'Both':
        response1 = model_hermes(prompt)[0]['generated_text']
        response2 = model_manticore(prompt)[0]['generated_text']
    else:
        model = model_hermes if model_choice == 'Model 1' else model_manticore
        response1 = model(prompt)[0]['generated_text']
        response2 = model(prompt)[0]['generated_text']

    return response1, response2


def user(message, nudge_msg, history1, history2):
    history1 = history1 or []
    history2 = history2 or []
    # Append the user's message to the conversation history
    history1.append([message, nudge_msg])
    history2.append([message, nudge_msg])

    return "", nudge_msg, history1, history2


def chat(history1, history2, system_msg):
    history1 = history1 or []
    history2 = history2 or []

    messages1 = system_msg.strip() + "\n" + \
                "\n".join(["\n".join(["USER: "+item[0], "ASSISTANT: "+item[1]])
                           for item in history1])
    messages2 = system_msg.strip() + "\n" + \
                "\n".join(["\n".join(["USER: "+item[0], "ASSISTANT: "+item[1]])
                           for item in history2])

    # remove last space from assistant, some models output a ZWSP if you leave a space
    messages1 = messages1.rstrip()
    messages2 = messages2.rstrip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        futures.append(executor.submit(model_hermes, messages1))
        futures.append(executor.submit(model_manticore, messages2))

    tokens_hermes = re.findall(r'\s*\S+\s*', futures[0].result()[0]['generated_text'])
    tokens_manticore = re.findall(r'\s*\S+\s*', futures[1].result()[0]['generated_text'])
    len_tokens_hermes = len(tokens_hermes)
    len_tokens_manticore = len(tokens_manticore)
    max_tokens = max(len_tokens_hermes, len_tokens_manticore)
    for i in range(0, max_tokens):
        if i <= len_tokens_hermes:
            answer1 = tokens_hermes[i]
            history1[-1][1] += answer1
        if i <= len_tokens_manticore:
            answer2 = tokens_manticore[i]
            history2[-1][1] += answer2
        # stream the response
        yield history1, history2, ""
        sleep(0.15)


def chosen_one(label, choice0_history, choice1_history, system_msg, nudge_msg, rlhf_persona):
    # Generate a uuid for each submission
    arena_battle_id = str(uuid.uuid4())

    # Get the current timestamp
    timestamp = datetime.datetime.now().isoformat()

    # Put the item in the table
    table.put_item(
        Item={
            'arena_battle_id': arena_battle_id,
            'timestamp': timestamp,
            'system_msg': system_msg,
            'nudge_prefix': nudge_msg,
            'choice0_name': model_hermes.name,
            'choice0': choice0_history,
            'choice1_name': model_manticore.name,
            'choice1': choice1_history,
            'label': label,
            'rlhf_persona': rlhf_persona,
        }
    )

chosen_one_first = functools.partial(chosen_one, 0)
chosen_one_second = functools.partial(chosen_one, 1)

with gr.Blocks() as arena:
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
                    ### brought to you by OpenAccess AI Collective
                    - This Space runs on CPU only, and uses GGML with GPU support via Runpod Serverless.
                    - Due to limitations of Runpod Serverless, it cannot stream responses immediately
                    - Responses WILL take AT LEAST 30 seconds to respond, probably longer   
                    - For now, this is single turn only
                    - For now, Hermes 13B on the left, Manticore on the right.
                    """)
    with gr.Tab("Chatbot"):
        with gr.Row():
            with gr.Column():
                chatbot1 = gr.Chatbot()
            with gr.Column():
                chatbot2 = gr.Chatbot()
        with gr.Row():
            choose1 = gr.Button(value="Prefer left", variant="secondary", visible=False).style(full_width=True)
            choose2 = gr.Button(value="Prefer right", variant="secondary", visible=False).style(full_width=True)
        with gr.Row():
            with gr.Column():
                rlhf_persona = gr.Textbox(
                    "", label="Persona Tags", interactive=True, visible=True, placeholder="Tell us about how you are judging the quality. like #SFW #NSFW #helpful #ethical #creativity", lines=1)
                message = gr.Textbox(
                    label="What do you want to chat about?",
                    placeholder="Ask me anything.",
                    lines=3,
                )
            with gr.Column():
                system_msg = gr.Textbox(
                    start_message, label="System Message", interactive=True, visible=True, placeholder="system prompt", lines=5)

                nudge_msg = gr.Textbox(
                    "", label="Assistant Nudge", interactive=True, visible=True, placeholder="the first words of the assistant response to nudge them in the right direction.", lines=2)
        with gr.Row():
            submit = gr.Button(value="Send message", variant="secondary").style(full_width=True)
            clear = gr.Button(value="New topic", variant="secondary").style(full_width=False)
    with gr.Tab("Leaderboard"):
        with gr.Column():
            gr.Markdown(f"""
                    ### TBD
                    - This is very much a work-in-progress, if you'd like to help build this out, join us on [Discord](https://discord.gg/QYF8QrtEUm)
                    """)

    clear.click(lambda: None, None, chatbot1, queue=False)
    clear.click(lambda: None, None, chatbot2, queue=False)
    clear.click(lambda: None, None, message, queue=False)
    clear.click(lambda: None, None, nudge_msg, queue=False)

    submit_click_event = submit.click(
        lambda *args: (
            gr.update(visible=False, interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        inputs=[], outputs=[message, clear, submit], queue=True
    ).then(
        fn=user, inputs=[message, nudge_msg, chatbot1, chatbot2], outputs=[message, nudge_msg, chatbot1, chatbot2], queue=True
    ).then(
        fn=chat, inputs=[chatbot1, chatbot2, system_msg], outputs=[chatbot1, chatbot2, message], queue=True
    ).then(
        lambda *args: (
            gr.update(visible=False, interactive=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        inputs=[message, nudge_msg, system_msg], outputs=[message, choose1, choose2, clear, submit], queue=True
    )

    choose1_click_event = choose1.click(
        fn=chosen_one_first, inputs=[chatbot1, chatbot2, system_msg, nudge_msg, rlhf_persona], outputs=[], queue=True
    ).then(
        lambda *args: (
            gr.update(visible=True, interactive=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            None,
            None,
        ),
        inputs=[], outputs=[message, choose1, choose2, clear, submit, chatbot1, chatbot2], queue=True
    )

    choose2_click_event = choose2.click(
        fn=chosen_one_second, inputs=[chatbot1, chatbot2, system_msg, nudge_msg, rlhf_persona], outputs=[], queue=True
    ).then(
        lambda *args: (
            gr.update(visible=True, interactive=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            None,
            None,
        ),
        inputs=[], outputs=[message, choose1, choose2, clear, submit, chatbot1, chatbot2], queue=True
    )


arena.queue(concurrency_count=2, max_size=16).launch(debug=True, server_name="0.0.0.0", server_port=7860)