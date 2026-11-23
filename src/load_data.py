import json

def load_companies(path: str = "data/companies.jsonl"):
    companies = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            companies.append(json.loads(line))

    return companies


if __name__ == "__main__":
    companies = load_companies("data/companies.jsonl")

    print(f"Loaded {len(companies)} companies")
    print(json.dumps(companies[0], indent=2))