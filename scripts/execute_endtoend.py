"""
End-to-end prompting baseline for table QA.
Prompts the LLM to directly output the answer given the table, question,
and optionally the text document -- no SQL generation or table augmentation.
"""

import json
import argparse
import platform, multiprocessing
import os
import time
import random
import pyrootutils
pyrootutils.setup_root('.project-root', pythonpath=True)
from transformers import AutoTokenizer

from generation.generator import Generator
from utils.utils import load_data_split, floatify_ans
from utils.evaluator import Evaluator
from utils.tatqa_metric import TaTQAEmAndF1

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


def worker_execute(
        pid,
        args,
        dataset,
        g_eids,
        tokenizer_name_or_path
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_name_or_path)
    generator = Generator(args, api_key_file=args.api_config_file)
    em_and_f1 = TaTQAEmAndF1()

    result_dict = dict()
    n_total_samples, n_correct_samples = 0, 0

    for eid in g_eids:
        data_item = dataset[eid]
        eid = str(eid)
        print(f"Process#{pid}: eid {eid}, id {data_item['id']}")

        result_dict[eid] = dict()
        result_dict[eid]['question'] = data_item['question']
        result_dict[eid]['gold_answer'] = data_item['answer_text']
        result_dict[eid]['qid'] = data_item['id']
        n_total_samples += 1

        # Build few-shot prompt
        n_shots = args.n_shots
        few_shot_prompt = generator.build_few_shot_prompt_from_file(
            file_path=args.prompt_file,
            n_shots=n_shots
        )

        # Build query: report (if applicable) + table + question, matching few-shot format
        table = data_item['table']
        query = ""
        if args.dataset in ['finqa', 'tatqa']:
            query += "Report:\n"
            doc = data_item.get('document_input', [])
            if doc and len(doc) > 0:
                query += ' '.join(line.strip() for line in doc) + '\n'
            else:
                query += "Empty\n"
            query += "Tables:\n"
        elif args.dataset in ['wikitq', 'missing_squall']:
            title = table.get('page_title', '')
            if title:
                query += f"Title: {title}\n"

        header = table['header']
        rows = table['rows']
        query += ' | '.join(str(h) for h in header) + '\n'
        for row in rows:
            query += ' | '.join(str(cell) for cell in row) + '\n'
        query += f"\nQuestion: {data_item['question']}\nAnswer:"

        prompt = few_shot_prompt + "\n\n" + query

        # Shrink n_shots if prompt exceeds token budget
        max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
        while len(tokenizer.tokenize(prompt)) >= max_prompt_tokens and n_shots > 0:
            n_shots -= 1
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=args.prompt_file,
                n_shots=n_shots
            )
            prompt = few_shot_prompt + "\n\n" + query

        print(f"Process#{pid}: Prompt ready for eid#{eid}, id#{data_item['id']}")

        try:
            response_dict = generator.generate_one_pass(
                prompts=[(eid, prompt)],
                verbose=args.verbose,
                include_system_prompt=True
            )
            raw_answer = response_dict[eid][0].strip()
            result_dict[eid]['raw_answer'] = raw_answer

            # Parse prediction into the format the evaluator expects
            if args.dataset == 'finqa':
                pred_answer = floatify_ans(raw_answer)
            elif args.dataset == 'tatqa':
                try:
                    pred_answer = float(raw_answer)
                except (ValueError, TypeError):
                    pred_answer = raw_answer
            else:
                # WikiTQ evaluator expects pred as a list (gold is already a list)
                pred_answer = [a.strip() for a in raw_answer.split(',') if a.strip()]
                if not pred_answer:
                    pred_answer = [raw_answer]

            result_dict[eid]['pred_answer'] = pred_answer

            # Evaluate
            gold_answer = data_item['answer_text']
            if args.dataset == 'tatqa':
                if isinstance(gold_answer, list) and len(gold_answer) == 2 and gold_answer[-1] == '###':
                    gold_answer = gold_answer[0]
                eval_item = {'answer': gold_answer,
                             'answer_type': data_item.get('answer_type', ''),
                             'scale': data_item.get('scale', '')}
                score = em_and_f1(ground_truth=eval_item, prediction=pred_answer, pred_scale='')
            else:
                eval_dataset = 'wikitq' if 'squall' in args.dataset or 'wikitq' in args.dataset else args.dataset
                score = Evaluator().evaluate(
                    pred_answer, gold_answer,
                    dataset=eval_dataset,
                    question=data_item['question']
                )

            result_dict[eid]['score'] = score
            n_correct_samples += score

        except Exception as e:
            print(f"Process#{pid}: Error on eid {eid}: {e}")
            result_dict[eid]['pred_answer'] = '<error>'
            result_dict[eid]['score'] = 0

        print(f'Process#{pid}: pred answer: {result_dict[eid].get("pred_answer", "N/A")}')
        print(f'Process#{pid}: gold answer: {data_item["answer_text"]}')
        if result_dict[eid]['score'] == 1:
            print(f'Process#{pid}: Correct!')
        else:
            print(f'Process#{pid}: Wrong.')
        print(f'Process#{pid}: Accuracy: {n_correct_samples}/{n_total_samples} = {n_correct_samples / n_total_samples}')

    return result_dict


def main():
    # Build paths
    args.api_config_file = os.path.join(ROOT_DIR, args.api_config_file)
    args.prompt_file = os.path.join(ROOT_DIR, args.prompt_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"========== Using prompt file: {args.prompt_file} ==========")

    # Load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    if args.dataset == "wikitq" and args.dataset_split == "test":
        dataset = dataset.select(range(0, 4000, 4))

    # Split work across processes
    generate_eids = list(range(len(dataset)))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[(g_eid + random.randrange(args.n_processes)) % args.n_processes].append(g_eid)

    if 'gpt' in args.engine:
        tokenizer_name_or_path = os.path.join(ROOT_DIR, "utils", "gpt2")
    else:
        tokenizer_name_or_path = args.engine

    print(f"Executing {len(generate_eids)} examples with {args.n_processes} processes.")

    result_dict = dict()
    worker_results = []

    if args.debug:
        worker_result = worker_execute(
            0, args, dataset, generate_eids_group[0], tokenizer_name_or_path
        )
        result_dict.update(worker_result)
    else:
        pool = multiprocessing.Pool(processes=args.n_processes)
        for pid in range(args.n_processes):
            worker_results.append(pool.apply_async(worker_execute, args=(
                pid, args, dataset, generate_eids_group[pid], tokenizer_name_or_path
            )))

        for r in worker_results:
            result_dict.update(r.get())
        pool.close()
        pool.join()

    n_correct_samples = 0
    for eid, item in result_dict.items():
        n_correct_samples += item['score']
    print(f'Overall Accuracy: {n_correct_samples}/{len(result_dict)} = {n_correct_samples / len(result_dict)}')

    # Save results
    output_path = os.path.join(args.save_dir, args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f'Done. Elapsed time: {time.time() - start_time}')


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='finqa',
                        choices=['wikitq', 'missing_squall', 'finqa', 'tatqa'])
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_config_file', type=str, default='key.txt')
    parser.add_argument('--prompt_file', type=str, default='templates/prompts/finqa_endtoend_ic.txt')
    parser.add_argument('--output_file', type=str, default='endtoend_finqa_test.json')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--engine', type=str, default='gpt-3.5-turbo')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=2)

    # Generation options
    parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table')
    parser.add_argument('--n_shots', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_generation_tokens', type=int, default=128)
    parser.add_argument('--max_api_total_tokens', type=int, default=16001)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--sampling_n', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--stop_tokens', type=str, default='\n\n\n',
                        help='Split stop tokens by ||')

    # Debugging options
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')

    print("End-to-End Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()
