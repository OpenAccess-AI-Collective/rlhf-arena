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
from typing import Generator, Tuple, List, Dict

import boto3
import gradio as gr
import requests
from datasets import load_dataset

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logging.getLogger("httpx").setLevel(logging.WARNING)

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


def prompt_roleplay(system_msg, history):
    return "<|system|>" + system_msg.strip() + "\n" + \
        "\n".join(["\n".join(["<|user|>"+item[0], "<|model|>"+item[1]])
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
            "top_p": 0.90,
            "temperature": 0.72,
            "repetition_penalty": 1.22,
            "last_n_tokens": 64,
            "seed": -1,
            "batch_size": 8,
            "threads": -1,
            "stop": ["</s>", "USER:", "### Instruction:"] + stop_tokens,
        }

    def get_generation_config(self):
        return self.generation_config.copy()

    def __call__(self, prompt, config=None) -> Generator[List[Dict[str, str]], None, None]:
        input = config if config else self.generation_config.copy()
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

    def stream_output(self,task_id) -> Generator[List[Dict[str, str]], None, None]:
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
    "guanaco-13b": ("yxl8w98z017mw2", prompt_instruct),
    "minotaur-13b": ("6f1baphxjpjk7b", prompt_chat),
}

OAAIC_MODELS = [
    "minotaur-13b",
    "manticore-13b-chat",
    # "minotaur-mpt-7b",
]
OAAIC_MODELS_ROLEPLAY = {
    "manticore-13b-chat-roleplay": ("u6tv84bpomhfei", prompt_roleplay),
    "minotaur-13b-roleplay": ("6f1baphxjpjk7b", prompt_roleplay),
    # "minotaur-mpt-7b": ("vm1wcsje126x1x", prompt_chat),
}

_memoized_models = defaultdict()


def get_model_pipeline(model_name):
    if not _memoized_models.get(model_name):
        kwargs = {}
        if model_name in AVAILABLE_MODELS:
            if len(AVAILABLE_MODELS[model_name]) >= 3:
                kwargs["stop_tokens"] = AVAILABLE_MODELS[model_name][2]
            _memoized_models[model_name] = Pipeline(AVAILABLE_MODELS[model_name][0], model_name, AVAILABLE_MODELS[model_name][1], **kwargs)
        elif model_name in OAAIC_MODELS_ROLEPLAY:
            _memoized_models[model_name] = Pipeline(OAAIC_MODELS_ROLEPLAY[model_name][0], model_name, OAAIC_MODELS_ROLEPLAY[model_name][1], **kwargs)
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


def chat(history1, history2, system_msg, state):
    history1 = history1 or []
    history2 = history2 or []

    arena_bots = None
    if state and "models" in state and state['models']:
        arena_bots = state['models']
    if not arena_bots:
        arena_bots = list(AVAILABLE_MODELS.keys())
        random.shuffle(arena_bots)

    battle = arena_bots[0:2]
    model1 = get_model_pipeline(battle[0])
    model2 = get_model_pipeline(battle[1])

    messages1 = model1.transform_prompt(system_msg, history1)
    messages2 = model2.transform_prompt(system_msg, history2)

    # remove last space from assistant, some models output a ZWSP if you leave a space
    messages1 = messages1.rstrip()
    messages2 = messages2.rstrip()

    model1_res = model1(messages1)  # type: Generator[str, None, None]
    model2_res = model2(messages2)  # type: Generator[str, None, None]
    res = token_generator(model1_res, model2_res, lambda x: x[0]['generated_text'], fillvalue=[{'generated_text': ''}])  # type: Generator[Tuple[str, str], None, None]
    logging.info({"models": [model1.name, model2.name]})
    for t1, t2 in res:
        if t1 is not None:
            history1[-1][1] += t1
        if t2 is not None:
            history2[-1][1] += t2
        # stream the response
        # [arena_chatbot1, arena_chatbot2, arena_message, reveal1, reveal2, arena_state]
        yield history1, history2, "", gr.update(value=battle[0]), gr.update(value=battle[1]), {"models": [model1.name, model2.name]}
        sleep(0.2)


def chosen_one(label, choice1_history, choice2_history, system_msg, nudge_msg, rlhf_persona, state):
    if not state:
        logging.error("missing state!!!")
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


"""
OpenAccess AI Chatbots chat
"""

def open_clear_chat(chat_history_state, chat_message, nudge_msg):
    chat_history_state = []
    chat_message = ''
    nudge_msg = ''
    return chat_history_state, chat_message, nudge_msg


def open_user(message, nudge_msg, history):
    history = history or []
    # Append the user's message to the conversation history
    history.append([message, nudge_msg])
    return "", nudge_msg, history


def open_chat(model_name, history, system_msg, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
    history = history or []

    model = get_model_pipeline(model_name)
    config = model.get_generation_config()
    config["max_new_tokens"] = max_new_tokens
    config["temperature"] = temperature
    config["temperature"] = temperature
    config["top_p"] = top_p
    config["top_k"] = top_k
    config["repetition_penalty"] = repetition_penalty

    messages = model.transform_prompt(system_msg, history)

    # remove last space from assistant, some models output a ZWSP if you leave a space
    messages = messages.rstrip()

    model_res = model(messages, config=config)  # type: Generator[List[Dict[str, str]], None, None]
    for res in model_res:
        # tokens = re.findall(r'\s*\S+\s*', res[0]['generated_text'])
        tokens = re.findall(r'\S+\s*', res[0]['generated_text'])
        for subtoken in tokens:
            history[-1][1] += subtoken
            # stream the response
            yield history, history, ""
            sleep(0.01)


def open_rp_chat(model_name, history, system_msg, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
    history = history or []

    model = get_model_pipeline(f"{model_name}-roleplay")
    config = model.get_generation_config()
    config["max_new_tokens"] = max_new_tokens
    config["temperature"] = temperature
    config["temperature"] = temperature
    config["top_p"] = top_p
    config["top_k"] = top_k
    config["repetition_penalty"] = repetition_penalty

    messages = model.transform_prompt(system_msg, history)

    # remove last space from assistant, some models output a ZWSP if you leave a space
    messages = messages.rstrip()

    model_res = model(messages, config=config)  # type: Generator[List[Dict[str, str]], None, None]
    for res in model_res:
        tokens = re.findall(r'\S+\s*', res[0]['generated_text'])
        # tokens = re.findall(r'\s*\S+\s*', res[0]['generated_text'])
        for subtoken in tokens:
            history[-1][1] += subtoken
            # stream the response
            yield history, history, ""
            sleep(0.01)


with gr.Blocks() as arena:
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
                    ### brought to you by OpenAccess AI Collective
                    - Checkout out [our writeup on how this was built.](https://medium.com/@winglian/inference-any-llm-with-serverless-in-15-minutes-69eeb548a41d)
                    - This Space runs on CPU only, and uses GGML with GPU support via Runpod Serverless.
                    - Responses may not stream immediately due to cold starts on Serverless.
                    - Some responses WILL take AT LEAST 20 seconds to respond   
                    - The Chatbot Arena (for now), is single turn only. Responses will be cleared after submission. 
                    - Responses from the Arena will be used for building reward models. These reward models can be bucketed by Personas.
                    - [💵 Consider Donating on our Patreon](http://patreon.com/OpenAccessAICollective) or become a [GitHub Sponsor](https://github.com/sponsors/OpenAccess-AI-Collective)
                    - Join us on [Discord](https://discord.gg/PugNNHAF5r) 
                    """)
    with gr.Tab("Chatbot Arena"):
        with gr.Row():
            with gr.Column():
                arena_chatbot1 = gr.Chatbot(label="Chatbot A")
            with gr.Column():
                arena_chatbot2 = gr.Chatbot(label="Chatbot B")
        with gr.Row():
            choose1 = gr.Button(value="👈 Prefer left (A)", variant="secondary", visible=False).style(full_width=True)
            choose2 = gr.Button(value="👉 Prefer right (B)", variant="secondary", visible=False).style(full_width=True)
            choose3 = gr.Button(value="🤝 Tie", variant="secondary", visible=False).style(full_width=True)
            choose4 = gr.Button(value="🤮 Both are bad", variant="secondary", visible=False).style(full_width=True)
        with gr.Row():
            reveal1 = gr.Textbox(label="Model Name", value="", interactive=False, visible=False).style(full_width=True)
            reveal2 = gr.Textbox(label="Model Name", value="", interactive=False, visible=False).style(full_width=True)
        with gr.Row():
            dismiss_reveal = gr.Button(value="Dismiss & Continue", variant="secondary", visible=False).style(full_width=True)
        with gr.Row():
            with gr.Column():
                arena_message = gr.Textbox(
                    label="What do you want to ask?",
                    placeholder="Ask me anything.",
                    lines=3,
                )
            with gr.Column():
                arena_rlhf_persona = gr.Textbox(
                    "", label="Persona Tags", interactive=True, visible=True, placeholder="Tell us about how you are judging the quality. ex: #CoT #SFW #NSFW #helpful #ethical #creativity", lines=2)
                arena_system_msg = gr.Textbox(
                    start_message, label="System Message", interactive=True, visible=True, placeholder="system prompt", lines=8)

                arena_nudge_msg = gr.Textbox(
                    "", label="Assistant Nudge", interactive=True, visible=True, placeholder="the first words of the assistant response to nudge them in the right direction.", lines=2)
        with gr.Row():
            arena_submit = gr.Button(value="Send message", variant="secondary").style(full_width=True)
            arena_clear = gr.Button(value="New topic", variant="secondary").style(full_width=False)
            # arena_regenerate = gr.Button(value="Regenerate", variant="secondary").style(full_width=False)
        arena_state = gr.State({})

        arena_clear.click(lambda: None, None, arena_chatbot1, queue=False)
        arena_clear.click(lambda: None, None, arena_chatbot2, queue=False)
        arena_clear.click(lambda: None, None, arena_message, queue=False)
        arena_clear.click(lambda: None, None, arena_nudge_msg, queue=False)
        arena_clear.click(lambda: None, None, arena_state, queue=False)

        submit_click_event = arena_submit.click(
            lambda *args: (
                gr.update(visible=False, interactive=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            inputs=[], outputs=[arena_message, arena_clear, arena_submit], queue=True
        ).then(
            fn=user, inputs=[arena_message, arena_nudge_msg, arena_chatbot1, arena_chatbot2], outputs=[arena_message, arena_nudge_msg, arena_chatbot1, arena_chatbot2], queue=True
        ).then(
            fn=chat, inputs=[arena_chatbot1, arena_chatbot2, arena_system_msg, arena_state], outputs=[arena_chatbot1, arena_chatbot2, arena_message, reveal1, reveal2, arena_state], queue=True
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
            inputs=[arena_message, arena_nudge_msg, arena_system_msg], outputs=[arena_message, choose1, choose2, choose3, choose4, arena_clear, arena_submit], queue=True
        )

        choose1_click_event = choose1.click(
            fn=chosen_one_first, inputs=[arena_chatbot1, arena_chatbot2, arena_system_msg, arena_nudge_msg, arena_rlhf_persona, arena_state], outputs=[], queue=True
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
            fn=chosen_one_second, inputs=[arena_chatbot1, arena_chatbot2, arena_system_msg, arena_nudge_msg, arena_rlhf_persona, arena_state], outputs=[], queue=True
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
            fn=chosen_one_tie, inputs=[arena_chatbot1, arena_chatbot2, arena_system_msg, arena_nudge_msg, arena_rlhf_persona, arena_state], outputs=[], queue=True
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
            fn=chosen_one_suck, inputs=[arena_chatbot1, arena_chatbot2, arena_system_msg, arena_nudge_msg, arena_rlhf_persona, arena_state], outputs=[], queue=True
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
                None,
            ),
            inputs=[], outputs=[
                arena_message,
                dismiss_reveal,
                arena_clear, arena_submit,
                reveal1, reveal2,
                arena_chatbot1, arena_chatbot2,
                arena_state,
            ], queue=True
        )
    with gr.Tab("Leaderboard"):
        with gr.Column():
            leaderboard_markdown = gr.Markdown(f"""{leaderboard_intro}
{dataset_to_markdown()}
""")
            leaderboad_refresh = gr.Button(value="Refresh Leaderboard", variant="secondary").style(full_width=True)
        leaderboad_refresh.click(fn=refresh_md, inputs=[], outputs=[leaderboard_markdown])
    with gr.Tab("OAAIC Chatbots"):
        gr.Markdown("# GGML Spaces Chatbot Demo")
        open_model_choice = gr.Dropdown(label="Model", choices=OAAIC_MODELS, value=OAAIC_MODELS[0])
        open_chatbot = gr.Chatbot()
        with gr.Row():
            open_message = gr.Textbox(
                label="What do you want to chat about?",
                placeholder="Ask me anything.",
                lines=3,
            )
        with gr.Row():
            open_submit = gr.Button(value="Send message", variant="secondary").style(full_width=True)
            open_roleplay = gr.Button(value="Roleplay", variant="secondary").style(full_width=True)
            open_clear = gr.Button(value="New topic", variant="secondary").style(full_width=False)
            open_stop = gr.Button(value="Stop", variant="secondary").style(full_width=False)
        with gr.Row():
            with gr.Column():
                open_max_tokens = gr.Slider(20, 1000, label="Max Tokens", step=20, value=300)
                open_temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=0.8)
                open_top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.95)
                open_top_k = gr.Slider(0, 100, label="Top K", step=1, value=40)
                open_repetition_penalty = gr.Slider(0.0, 2.0, label="Repetition Penalty", step=0.1, value=1.1)

        open_system_msg = gr.Textbox(
            start_message, label="System Message", interactive=True, visible=True, placeholder="system prompt, useful for RP", lines=5)

        open_nudge_msg = gr.Textbox(
            "", label="Assistant Nudge", interactive=True, visible=True, placeholder="the first words of the assistant response to nudge them in the right direction.", lines=1)

        open_chat_history_state = gr.State()
        open_clear.click(open_clear_chat, inputs=[open_chat_history_state, open_message, open_nudge_msg], outputs=[open_chat_history_state, open_message, open_nudge_msg], queue=False)
        open_clear.click(lambda: None, None, open_chatbot, queue=False)

        open_submit_click_event = open_submit.click(
            fn=open_user, inputs=[open_message, open_nudge_msg, open_chat_history_state], outputs=[open_message, open_nudge_msg, open_chat_history_state], queue=True
        ).then(
            fn=open_chat, inputs=[open_model_choice, open_chat_history_state, open_system_msg, open_max_tokens, open_temperature, open_top_p, open_top_k, open_repetition_penalty], outputs=[open_chatbot, open_chat_history_state, open_message], queue=True
        )
        open_roleplay_click_event = open_roleplay.click(
            fn=open_user, inputs=[open_message, open_nudge_msg, open_chat_history_state], outputs=[open_message, open_nudge_msg, open_chat_history_state], queue=True
        ).then(
            fn=open_rp_chat, inputs=[open_model_choice, open_chat_history_state, open_system_msg, open_max_tokens, open_temperature, open_top_p, open_top_k, open_repetition_penalty], outputs=[open_chatbot, open_chat_history_state, open_message], queue=True
        )
        open_stop.click(fn=None, inputs=None, outputs=None, cancels=[open_submit_click_event, open_roleplay_click_event], queue=False)

arena.queue(concurrency_count=5, max_size=16).launch(debug=True, server_name="0.0.0.0", server_port=7860)