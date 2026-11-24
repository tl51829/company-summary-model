from load_data import load_companies
from summarize import generate_summaries

from bert_score import score

import json
from pathlib import Path

from prompts import PROMPTS

def evaluate_summaries(
    generated_summaries: dict[str, str],
    expected_summaries: list[dict],
    output_path: str,
):
    expected_by_id = {
        item["registrationNumber"]: item["summary"]
        for item in expected_summaries
    }

    generated = []
    expected = []
    registration_numbers = []

    for registration_number, generated_summary in generated_summaries.items():
        if registration_number not in expected_by_id:
            print(f"Warning: No expected summary found for {registration_number}")
            continue

        registration_numbers.append(registration_number)
        generated.append(generated_summary)
        expected.append(expected_by_id[registration_number])

    if not generated:
        raise ValueError("No matching summaries found.")

    # Calculate BERTScore
    _, _, f1 = score(
        generated,
        expected,
        lang="en",
        model_type="roberta-large",
        verbose=True,
    )

    results = [
        {
            "registrationNumber": registration_numbers[i],
            "bert_score_f1": f1[i].item(),
        }
        for i in range(len(registration_numbers))
    ]

    average_f1 = f1.mean().item()

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "results": results,
        "average_bert_score_f1": average_f1,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print(f"Average BERTScore F1: {average_f1:.4f}")
    print(f"Evaluation results saved to {output_path}")

    return results, average_f1

if __name__ == "__main__":
    for version, system_prompt in enumerate(PROMPTS):
        version = version + 1
        companies = load_companies("test/test_companies.jsonl")
        summaries = generate_summaries(companies, f"test/output/test_output_{version}.jsonl", system_prompt)
        expected_summaries = load_companies("test/expected/test_expected.jsonl")
        evaluate_summaries(summaries, expected_summaries, f"test/result/test_result_{version}.json")