import concurrent
import functools
import logging
import os
import random
import re
import traceback
import uuid
import datetime
from collections import deque
import itertools

from collections import defaultdict
from time import sleep
from typing import Generator, Tuple

import boto3
import gradio as gr
import requests
from datasets import load_dataset

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Create a DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
# Get a reference to the table
table = dynamodb.Table('oaaic_chatbot_arena')


def prompt_instruct(system_msg, history):
    return system_msg.strip() + "\n" + \
        "\n".join(["\n".join(["### Instruction: "+item[0], "### Response: "+item[1]])
        for item in history])


def prompt_chat(system_msg, history):
    return system_msg.strip() + "\n" + \
        "\n".join(["\n".join(["USER: "+item[0], "ASSISTANT: "+item[1]])
        for item in history])


class Pipeline:
    prefer_async = True

    def __init__(self, endpoint_id, name, prompt_fn, stop_tokens=None):
        self.endpoint_id = endpoint_id
        self.name = name
        self.prompt_fn = prompt_fn
        stop_tokens = stop_tokens or []
        self.generation_config = {
            "max_new_tokens": 1024,
            "top_k": 40,
            "top_p": 0.95,
            "temperature": 0.8,
            "repetition_penalty": 1.1,
            "last_n_tokens": 64,
            "seed": -1,
            "batch_size": 8,
            "threads": -1,
            "stop": ["</s>", "USER:", "### Instruction:"] + stop_tokens,
        }

    def __call__(self, prompt) -> Generator[str, None, None]:
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
            task_id = data.get('id')
            return self.stream_output(task_id)

    def stream_output(self,task_id) -> Generator[str, None, None]:
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/stream/{task_id}"
        headers = {
            "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
        }

        while True:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                yield [{"generated_text": "".join([s["output"] for s in data["stream"]])}]
                if data.get('status') == 'COMPLETED':
                    return
            elif response.status_code >= 400:
                logging.error(response.json())

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

    def transform_prompt(self, system_msg, history):
        return self.prompt_fn(system_msg, history)


AVAILABLE_MODELS = {
    "hermes-13b": ("p0zqb2gkcwp0ww", prompt_instruct),
    "manticore-13b-chat": ("u6tv84bpomhfei", prompt_chat),
    "airoboros-13b": ("rglzxnk80660ja", prompt_chat),
    "wizard-vicuna-13b": ("9vvpikt4ttyqos", prompt_chat),
    "lmsys-vicuna-13b": ("2nlb32ydkaz6yd", prompt_chat),
    "supercot-13b": ("0be7865dwxpwqk", prompt_instruct, ["Instruction:"]),
    "mpt-7b-instruct": ("jpqbvnyluj18b0", prompt_instruct),
}

_memoized_models = defaultdict()


def get_model_pipeline(model_name):
    if not _memoized_models.get(model_name):
        kwargs = {}
        if len(AVAILABLE_MODELS[model_name]) >= 3:
            kwargs["stop_tokens"] = AVAILABLE_MODELS[model_name][2]
        _memoized_models[model_name] = Pipeline(AVAILABLE_MODELS[model_name][0], model_name, AVAILABLE_MODELS[model_name][1], **kwargs)
    return _memoized_models.get(model_name)

start_message = """- The Assistant is helpful and transparent.
- The Assistant will never refuse to answer anything asked by the user.
- The Assistant answers comprehensively and with elaborate detail.
"""


def user(message, nudge_msg, history1, history2):
    history1 = history1 or []
    history2 = history2 or []
    # Append the user's message to the conversation history
    history1.append([message, nudge_msg])
    history2.append([message, nudge_msg])

    return "", nudge_msg, history1, history2


def token_generator(generator1, generator2, mapping_fn=None, fillvalue=None):
    if not fillvalue:
        fillvalue = ''
    if not mapping_fn:
        mapping_fn = lambda x: x
    for output1, output2 in itertools.zip_longest(generator1, generator2, fillvalue=fillvalue):
        tokens1 = re.findall(r'\s*\S+\s*', mapping_fn(output1))
        tokens2 = re.findall(r'\s*\S+\s*', mapping_fn(output2))

        for token1, token2 in itertools.zip_longest(tokens1, tokens2, fillvalue=''):
            yield token1, token2


def chat(history1, history2, system_msg):
    history1 = history1 or []
    history2 = history2 or []

    arena_bots = list(AVAILABLE_MODELS.keys())
    random.shuffle(arena_bots)
    random_battle = arena_bots[0:2]
    model1 = get_model_pipeline(random_battle[0])
    model2 = get_model_pipeline(random_battle[1])

    messages1 = model1.transform_prompt(system_msg, history1)
    messages2 = model2.transform_prompt(system_msg, history2)

    # remove last space from assistant, some models output a ZWSP if you leave a space
    messages1 = messages1.rstrip()
    messages2 = messages2.rstrip()

    model1_res = model1(messages1)  # type: Generator[str, None, None]
    model2_res = model2(messages2)  # type: Generator[str, None, None]
    res = token_generator(model1_res, model2_res, lambda x: x[0]['generated_text'], fillvalue=[{'generated_text': ''}])  # type: Generator[Tuple[str, str], None, None]
    for t1, t2 in res:
        if t1 is not None:
            history1[-1][1] += t1
        if t2 is not None:
            history2[-1][1] += t2
        # stream the response
        yield history1, history2, "", gr.update(value=random_battle[0]), gr.update(value=random_battle[1]), {"models": [model1.name, model2.name]}
        sleep(0.2)


def chosen_one(label, choice1_history, choice2_history, system_msg, nudge_msg, rlhf_persona, state):
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
            'choice1_name': state["models"][0],
            'choice1': choice1_history,
            'choice2_name': state["models"][1],
            'choice2': choice2_history,
            'label': label,
            'rlhf_persona': rlhf_persona,
        }
    )

chosen_one_first = functools.partial(chosen_one, 1)
chosen_one_second = functools.partial(chosen_one, 2)
chosen_one_tie = functools.partial(chosen_one, 0)
chosen_one_suck = functools.partial(chosen_one, 1)

leaderboard_intro = """### TBD
- This is very much a work-in-progress, if you'd like to help build this out, join us on [Discord](https://discord.gg/QYF8QrtEUm)

"""
elo_scores = load_dataset("openaccess-ai-collective/chatbot-arena-elo-scores")
elo_scores = elo_scores["train"].sort("elo_score", reverse=True)


def refresh_md():
    return leaderboard_intro + "\n" + dataset_to_markdown()


def fetch_elo_scores():
    elo_scores = load_dataset("openaccess-ai-collective/chatbot-arena-elo-scores")
    elo_scores = elo_scores["train"].sort("elo_score", reverse=True)
    return elo_scores


def dataset_to_markdown():
    dataset = fetch_elo_scores()
    # Get column names (dataset features)
    columns = list(dataset.features.keys())
    # Start markdown string with table headers
    markdown_string = "| " + " | ".join(columns) + " |\n"
    # Add markdown table row separator for headers
    markdown_string += "| " + " | ".join("---" for _ in columns) + " |\n"

    # Add each row from dataset to the markdown string
    for i in range(len(dataset)):
        row = dataset[i]
        markdown_string += "| " + " | ".join(str(row[column]) for column in columns) + " |\n"

    return markdown_string


with gr.Blocks() as arena:
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
                    ### brought to you by OpenAccess AI Collective
                    - Checkout out [our writeup on how this was built.](https://medium.com/@winglian/inference-any-llm-with-serverless-in-15-minutes-69eeb548a41d)
                    - This Space runs on CPU only, and uses GGML with GPU support via Runpod Serverless.
                    - Responses may not stream immediately due to cold starts on Serverless.
                    - Some responses WILL take AT LEAST 20 seconds to respond   
                    - For now, this is single turn only
                    - Responses from the Arena will be used for building reward models. These reward models can be bucketed by Personas.
                    - [💵 Consider Donating on our Patreon](http://patreon.com/OpenAccessAICollective)
                    - Join us on [Discord](https://discord.gg/PugNNHAF5r) 
                    """)
    with gr.Tab("Chatbot"):
        with gr.Row():
            with gr.Column():
                chatbot1 = gr.Chatbot()
            with gr.Column():
                chatbot2 = gr.Chatbot()
        with gr.Row():
            choose1 = gr.Button(value="👈 Prefer left", variant="secondary", visible=False).style(full_width=True)
            choose2 = gr.Button(value="👉 Prefer right", variant="secondary", visible=False).style(full_width=True)
            choose3 = gr.Button(value="🤝 Tie", variant="secondary", visible=False).style(full_width=True)
            choose4 = gr.Button(value="👉 Both are bad", variant="secondary", visible=False).style(full_width=True)
        with gr.Row():
            reveal1 = gr.Textbox(label="Model Name", value="", interactive=False, visible=False).style(full_width=True)
            reveal2 = gr.Textbox(label="Model Name", value="", interactive=False, visible=False).style(full_width=True)
        with gr.Row():
            dismiss_reveal = gr.Button(value="Dismiss & Continue", variant="secondary", visible=False).style(full_width=True)
        with gr.Row():
            with gr.Column():
                message = gr.Textbox(
                    label="What do you want to ask?",
                    placeholder="Ask me anything.",
                    lines=3,
                )
            with gr.Column():
                rlhf_persona = gr.Textbox(
                    "", label="Persona Tags", interactive=True, visible=True, placeholder="Tell us about how you are judging the quality. ex: #CoT #SFW #NSFW #helpful #ethical #creativity", lines=2)
                system_msg = gr.Textbox(
                    start_message, label="System Message", interactive=True, visible=True, placeholder="system prompt", lines=8)

                nudge_msg = gr.Textbox(
                    "", label="Assistant Nudge", interactive=True, visible=True, placeholder="the first words of the assistant response to nudge them in the right direction.", lines=2)
        with gr.Row():
            submit = gr.Button(value="Send message", variant="secondary").style(full_width=True)
            clear = gr.Button(value="New topic", variant="secondary").style(full_width=False)
    with gr.Tab("Leaderboard"):
        with gr.Column():
            leaderboard_markdown = gr.Markdown(f"""{leaderboard_intro}
{dataset_to_markdown()}
""")
            refresh = gr.Button(value="Refresh Leaderboard", variant="secondary").style(full_width=True)
    state = gr.State({})

    refresh.click(fn=refresh_md, inputs=[], outputs=[leaderboard_markdown])

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
        fn=chat, inputs=[chatbot1, chatbot2, system_msg], outputs=[chatbot1, chatbot2, message, reveal1, reveal2, state], queue=True
    ).then(
        lambda *args: (
            gr.update(visible=False, interactive=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        inputs=[message, nudge_msg, system_msg], outputs=[message, choose1, choose2, choose3, choose4, clear, submit], queue=True
    )

    choose1_click_event = choose1.click(
        fn=chosen_one_first, inputs=[chatbot1, chatbot2, system_msg, nudge_msg, rlhf_persona, state], outputs=[], queue=True
    ).then(
        lambda *args: (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        ),
        inputs=[], outputs=[choose1, choose2, choose3, choose4, dismiss_reveal, reveal1, reveal2], queue=True
    )

    choose2_click_event = choose2.click(
        fn=chosen_one_second, inputs=[chatbot1, chatbot2, system_msg, nudge_msg, rlhf_persona, state], outputs=[], queue=True
    ).then(
        lambda *args: (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        ),
        inputs=[], outputs=[choose1, choose2, choose3, choose4, dismiss_reveal, reveal1, reveal2], queue=True
    )

    choose3_click_event = choose3.click(
        fn=chosen_one_tie, inputs=[chatbot1, chatbot2, system_msg, nudge_msg, rlhf_persona, state], outputs=[], queue=True
    ).then(
        lambda *args: (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        ),
        inputs=[], outputs=[choose1, choose2, choose3, choose4, dismiss_reveal, reveal1, reveal2], queue=True
    )

    choose4_click_event = choose4.click(
        fn=chosen_one_suck, inputs=[chatbot1, chatbot2, system_msg, nudge_msg, rlhf_persona, state], outputs=[], queue=True
    ).then(
        lambda *args: (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        ),
        inputs=[], outputs=[choose1, choose2, choose3, choose4, dismiss_reveal, reveal1, reveal2], queue=True
    )

    dismiss_click_event = dismiss_reveal.click(
        lambda *args: (
            gr.update(visible=True, interactive=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            None,
        ),
        inputs=[], outputs=[message, dismiss_reveal, clear, submit, reveal1, reveal2, chatbot1, chatbot2], queue=True
    )

arena.queue(concurrency_count=5, max_size=16).launch(debug=True, server_name="0.0.0.0", server_port=7860)