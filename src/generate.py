from load_data import load_companies
from summarize import generate_summaries

from prompts import PROMPTS

if __name__ == "__main__":
    for version, system_prompt in enumerate(PROMPTS):
        version = version + 1
        companies = load_companies("data/companies.jsonl")
        summaries = generate_summaries(companies, f"summaries/version_{version}.jsonl", system_prompt)