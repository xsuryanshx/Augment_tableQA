"""
Generate nsql and questions.
Uses local vLLM endpoint (OpenAI-compatible API) when vllm_config.json is present.
"""

from typing import Dict, List, Union, Tuple
import json
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

from generation.prompt import PromptBuilder


def _load_vllm_config(api_key_file: str) -> Dict:
    """Load vLLM config from vllm_config.json in the same directory as api_key_file."""
    config_dir = os.path.dirname(os.path.abspath(api_key_file))
    config_path = os.path.join(config_dir, 'vllm_config.json')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"vllm_config.json not found at {config_path}. "
            "Copy vllm_config.example.json to vllm_config.json and set base_url and model."
        )
    with open(config_path, 'r') as f:
        return json.load(f)


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=4, max=30))
def call_llm_api(engine, messages, max_tokens, temperature, top_p, n, stop, key, base_url: str):

    client = OpenAI(api_key=key, base_url=base_url)
    result = client.chat.completions.create(
        model=engine,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        seed=0,
    )

    return result


class Generator(object):
    """
    Codex generation wrapper.
    """

    def __init__(self, args, api_key_file: str='key.txt',
                 system_prompt_file: str='templates/prompts/default_system.txt'):
        
        self.args = args
        vllm_config = _load_vllm_config(api_key_file)
        self.base_url = vllm_config['base_url']
        self.engine = vllm_config['model']
        # key.txt can contain dummy value; vLLM ignores it
        key = vllm_config.get('dummy_api_key', 'dummy')
        self.keys = [key]
        if os.path.isfile(api_key_file):
            with open(api_key_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if lines:
                self.keys = lines

        with open(system_prompt_file, 'r') as f:
            self.system_prompt = f.read()

        self.current_key_id = 0

        # if the args provided, will initialize with the prompt builder for full usage
        self.prompt_builder = PromptBuilder(args) if args else None

    def build_few_shot_prompt_from_file(
            self,
            file_path: str,
            n_shots: int
    ):
        """
        Build few-shot prompt for generation from file.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        few_shot_prompt_list = []
        one_shot_prompt = ''
        last_line = None
        for line in lines:
            if line == '\n' and last_line == '\n':
                few_shot_prompt_list.append(one_shot_prompt)
                one_shot_prompt = ''
            else:
                one_shot_prompt += line
            last_line = line
        if len(one_shot_prompt.strip()) > 0:
            few_shot_prompt_list.append(one_shot_prompt)
        few_shot_prompt_list = few_shot_prompt_list[-n_shots:]
        few_shot_prompt_list[-1] = few_shot_prompt_list[
            -1].strip()  # It is essential for prompting to remove extra '\n'
        few_shot_prompt_list = '\n'.join(few_shot_prompt_list)
        return few_shot_prompt_list

    def build_generate_prompt(
            self,
            data_item: Dict,
            generate_type: Tuple,
            table_only: bool = False,
            report_ahead: bool = False,
            datasetname: str = None,
            info_title: str = None,
            max_row: int = None,
    ):
        """
        Build the generate prompt
        """
        return self.prompt_builder.build_generate_prompt(
            **data_item,
            generate_type=generate_type,
            table_only=table_only,
            report_ahead=report_ahead,
            datasetname=datasetname,
            info_title=info_title,
            max_row=max_row
        )

    def generate_one_pass(
            self,
            prompts: List[Tuple],
            verbose: bool = False,
            include_system_prompt: bool = True
    ):
        """
        Generate one pass with codex according to the generation phase.
        """
        response_dict = dict()

        for eid, prompt in prompts:
            result = self._call_llm_api(
                prompt=prompt,
                max_tokens=self.args.max_generation_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                n=self.args.sampling_n,
                stop=self.args.stop_tokens,
                include_system_prompt=include_system_prompt
            )

            if verbose:
                print('\n', '*' * 20, 'API Call', '*' * 20)
                print(prompt)
                print('\n')
                print('- - - - - - - - - - ->>')

            # parse api results (vLLM and OpenAI chat both use message.content)
            for g in result.choices:
                try:
                    if hasattr(g, 'message') and g.message is not None:
                        text = g.message.content
                    elif hasattr(g, 'text'):
                        text = g.text
                    else:
                        text = str(g)
                    eid_pairs = response_dict.get(eid, None)
                    if eid_pairs is None:
                        eid_pairs = []
                        response_dict[eid] = eid_pairs
                    eid_pairs.append(text)

                    if verbose:
                        print(text)

                except Exception as e:
                    print(f'----------- eid {eid} Parsing API results Error --------')
                    print(e)
                    print(g['message'])
                    pass

        return response_dict

    def _call_llm_api(
            self,
            prompt: Union[str, List],
            max_tokens,
            temperature: float,
            top_p: float,
            n: int,
            stop: List[str],
            include_system_prompt: bool = False
    ):
        start_time = time.time()

        key = self.keys[self.current_key_id]
        self.current_key_id = (self.current_key_id + 1) % len(self.keys)
        
        messages = [{"role": "system", "content": self.system_prompt}] if include_system_prompt else []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            turns = ['user', 'assistant']
            for ind, p in enumerate(prompt):
                messages.append({"role": turns[ind % 2], "content": p})
        
        result = call_llm_api(
            engine=self.engine,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            key=key,
            base_url=self.base_url,
        )

        print('LLM api inference time:', time.time() - start_time)
        return result
