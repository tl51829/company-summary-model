import json
import os
from pathlib import Path

from huggingface_hub import InferenceClient

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def generate_summaries(companies: list[dict], output_path: str):
    client = InferenceClient(
        api_key=os.environ["HF_TOKEN"],
    )

    system_prompt = """
    You are a corporate analyst tasked with generating concise, objective
    company summaries using the information provided.

    Write a summary in Markdown covering:
    - Company overview and business activities
    - Financial performance, where information is available
    - Significant business developments
    - Legal and regulatory matters
    - Key risks and challenges

    Rules:
    - Use only the information provided.
    - Do not invent facts, figures, or events.
    - Distinguish allegations from established facts.
    - Preserve financial figures, currencies and reporting periods accurately.
    - If information for a particular section is unavailable, omit that section.
    - Write in a professional and objective tone.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = {}

    with output_path.open("w", encoding="utf-8") as file:
        for company in companies:
            registration_number = company["registrationNumber"]

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            "Generate a company summary using the following "
                            "company information:\n\n"
                            + json.dumps(
                                company,
                                indent=2,
                                ensure_ascii=False,
                            )
                        ),
                    },
                ],
                max_tokens=1024,
                temperature=0.0,
            )

            summary = completion.choices[0].message.content.strip()

            result = {
                "registrationNumber": registration_number,
                "summary": summary,
            }

            summaries[registration_number] = summary

            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()

            print(f"Generated summary for {registration_number}")

    print(f"Summaries saved to {output_path}")

    return summaries


if __name__ == "__main__":
    # company data
    companies = [
        {"registrationNumber": "SYN-000029", "profile": {"companyName": "Unity Global Trading Sdn. Bhd.", "status": "Active", "companyType": "Private limited company", "incorporationDate": "2018-06-15", "industry": "Retail", "state": "Selangor", "natureOfBusiness": "Retail of textiles and baju kurung", "shareCapital": 250000, "formerNames": []}, "relationships": {"directors": [{"name": "Director 1", "designation": "Managing Director", "appointedOn": "2021-02-05"}, {"name": "Director 2", "designation": "Independent Director", "appointedOn": "2022-01-24"}, {"name": "Director 3", "designation": "Director", "appointedOn": "2022-11-06"}, {"name": "Director 4", "designation": "Director", "appointedOn": "2025-11-30"}, {"name": "Director 5", "designation": "Executive Director", "appointedOn": "2025-11-01"}], "shareholders": [{"type": "individual", "name": "Director 1", "percentage": 12.21}, {"type": "individual", "name": "Director 5", "percentage": 21.22}, {"type": "individual", "name": "Director 4", "percentage": 58.25}, {"type": "individual", "name": "Director 3", "percentage": 8.32}]}, "financials": {"years": [{"financialYear": 2022, "financialYearEnd": "2022-06-30", "audit": {"firm": "Audit Firm 8", "opinion": "Unqualified"}, "statements": {"revenue": 5130131, "costOfSales": 3587542, "grossProfit": 1542589, "operatingExpenses": 1281343, "operatingProfit": 261246, "depreciation": 153701, "interestExpense": 111364, "profitBeforeTax": 149882, "taxExpense": 35972, "profitAfterTax": 113910, "cash": 378429, "tradeReceivables": 1361607, "inventory": 188389, "currentAssets": 2314615, "nonCurrentAssets": 5409188, "totalAssets": 7723803, "tradePayables": 513945, "borrowings": 2024803, "currentLiabilities": 1558212, "totalLiabilities": 3412401, "totalEquity": 4311402, "operatingCashFlow": 127469, "capitalExpenditure": 275657, "borrowingsRepaid": 206295}, "ratiosAndMetrics": {"profitability": {"grossProfitMargin": 0.3007, "operatingProfitMargin": 0.0509, "ebitda": 414947, "ebitdaMargin": 0.0809, "operatingExpenseRatio": 0.2498, "netProfitMargin": 0.0222, "taxEfficiencyRatio": 0.76}, "liquidityAndCashFlow": {"quickRatio": 1.3645, "cashRatio": 0.2429, "cashToRevenueRatio": 0.0738, "operatingCashFlow": 127469, "operatingCashFlowMargin": 0.0248, "freeCashFlow": -148188, "accrualRatio": -0.0018}, "operationalEfficiency": {"cashConversionCycle": 63.7536, "daysSalesOutstanding": 96.876, "daysInventoryOutstanding": 19.1669, "daysPayablesOutstanding": 52.2893, "inventoryTurnover": 19.0433, "receivablesTurnover": 3.7677, "assetTurnoverRatio": 0.6642, "netWorkingCapital": 756403}, "capitalStructure": {"leverageRatio": 0.7915, "interestCoverageRatio": 2.3459, "fixedAssetComposition": 0.7003, "depreciationToAssetRatio": 0.0199, "debtServiceCoverageRatio": 1.3063}}}, {"financialYear": 2023, "financialYearEnd": "2023-06-30", "audit": {"firm": "Audit Firm 8", "opinion": "Unqualified"}, "statements": {"revenue": 5229803, "costOfSales": 3713394, "grossProfit": 1516409, "operatingExpenses": 1273998, "operatingProfit": 242411, "depreciation": 156687, "interestExpense": 113528, "profitBeforeTax": 128883, "taxExpense": 30932, "profitAfterTax": 97951, "cash": 857319, "tradeReceivables": 1295837, "inventory": 225513, "currentAssets": 2772362, "nonCurrentAssets": 5101504, "totalAssets": 7873866, "tradePayables": 480387, "borrowings": 2064142, "currentLiabilities": 1574314, "totalLiabilities": 3493898, "totalEquity": 4379968, "operatingCashFlow": 228345, "capitalExpenditure": 236404, "borrowingsRepaid": 287155}, "ratiosAndMetrics": {"profitability": {"grossProfitMargin": 0.29, "operatingProfitMargin": 0.0464, "ebitda": 399098, "ebitdaMargin": 0.0763, "operatingExpenseRatio": 0.2436, "netProfitMargin": 0.0187, "taxEfficiencyRatio": 0.76}, "liquidityAndCashFlow": {"quickRatio": 1.6178, "cashRatio": 0.5446, "cashToRevenueRatio": 0.1639, "operatingCashFlow": 228345, "operatingCashFlowMargin": 0.0437, "freeCashFlow": -8059, "accrualRatio": -0.0166}, "operationalEfficiency": {"cashConversionCycle": 65.3871, "daysSalesOutstanding": 90.4394, "daysInventoryOutstanding": 22.1663, "daysPayablesOutstanding": 47.2186, "inventoryTurnover": 16.4664, "receivablesTurnover": 4.0358, "assetTurnoverRatio": 0.6642, "netWorkingCapital": 1198048}, "capitalStructure": {"leverageRatio": 0.7977, "interestCoverageRatio": 2.1353, "fixedAssetComposition": 0.6479, "depreciationToAssetRatio": 0.0199, "debtServiceCoverageRatio": 0.996}}}, {"financialYear": 2024, "financialYearEnd": "2024-06-30", "audit": {"firm": "Audit Firm 8", "opinion": "Unqualified"}, "statements": {"revenue": 5352856, "costOfSales": 3474160, "grossProfit": 1878696, "operatingExpenses": 1635122, "operatingProfit": 243574, "depreciation": 160374, "interestExpense": 116199, "profitBeforeTax": 127375, "taxExpense": 30570, "profitAfterTax": 96805, "cash": 528591, "tradeReceivables": 1284784, "inventory": 210115, "currentAssets": 2426447, "nonCurrentAssets": 5632685, "totalAssets": 8059132, "tradePayables": 515615, "borrowings": 2112710, "currentLiabilities": 1640966, "totalLiabilities": 3611401, "totalEquity": 4447731, "operatingCashFlow": 113617, "capitalExpenditure": 353126, "borrowingsRepaid": 261283}, "ratiosAndMetrics": {"profitability": {"grossProfitMargin": 0.351, "operatingProfitMargin": 0.0455, "ebitda": 403948, "ebitdaMargin": 0.0755, "operatingExpenseRatio": 0.3055, "netProfitMargin": 0.0181, "taxEfficiencyRatio": 0.76}, "liquidityAndCashFlow": {"quickRatio": 1.3506, "cashRatio": 0.3221, "cashToRevenueRatio": 0.0987, "operatingCashFlow": 113617, "operatingCashFlowMargin": 0.0212, "freeCashFlow": -239509, "accrualRatio": -0.0021}, "operationalEfficiency": {"cashConversionCycle": 55.5105, "daysSalesOutstanding": 87.6067, "daysInventoryOutstanding": 22.075, "daysPayablesOutstanding": 54.1712, "inventoryTurnover": 16.5346, "receivablesTurnover": 4.1663, "assetTurnoverRatio": 0.6642, "netWorkingCapital": 785481}, "capitalStructure": {"leverageRatio": 0.812, "interestCoverageRatio": 2.0962, "fixedAssetComposition": 0.6989, "depreciationToAssetRatio": 0.0199, "debtServiceCoverageRatio": 1.0701}}}]}, "litigation": {"cases": []}},
        {"registrationNumber": "SYN-000067", "profile": {"companyName": "Jaya Prima Travel Sdn. Bhd.", "status": "Active", "companyType": "Private limited company", "incorporationDate": "1995-04-01", "industry": "Hospitality", "state": "Kuala Lumpur", "natureOfBusiness": "Travel agency for umrah and inbound tourism", "shareCapital": 2500000, "formerNames": []}, "relationships": {"directors": [{"name": "Director 1", "designation": "Managing Director", "appointedOn": "2004-09-27"}, {"name": "Director 2", "designation": "Director", "appointedOn": "2025-10-27"}, {"name": "Director 3", "designation": "Independent Director", "appointedOn": "2018-04-11"}, {"name": "Director 4", "designation": "Director", "appointedOn": "2020-10-16"}], "shareholders": [{"type": "individual", "name": "Director 4", "percentage": 17.84}, {"type": "individual", "name": "Director 1", "percentage": 47.38}, {"type": "individual", "name": "Director 2", "percentage": 34.78}]}, "financials": {"years": [{"financialYear": 2022, "financialYearEnd": "2022-12-31", "audit": {"firm": "Audit Firm 7", "opinion": "Unqualified"}, "statements": {"revenue": 49467702, "costOfSales": 20964987, "grossProfit": 28502715, "operatingExpenses": 31806399, "operatingProfit": -3303684, "depreciation": 840396, "interestExpense": 941144, "profitBeforeTax": -4244828, "taxExpense": 0, "profitAfterTax": -4244828, "cash": 7170360, "tradeReceivables": 4966523, "inventory": 5931568, "currentAssets": 21648069, "nonCurrentAssets": 49944295, "totalAssets": 71592364, "tradePayables": 4621342, "borrowings": 17111708, "currentLiabilities": 19077196, "totalLiabilities": 40377734, "totalEquity": 31214630, "operatingCashFlow": -3308659, "capitalExpenditure": 1773949, "borrowingsRepaid": 2370549}, "ratiosAndMetrics": {"profitability": {"grossProfitMargin": 0.5762, "operatingProfitMargin": -0.0668, "ebitda": -2463288, "ebitdaMargin": -0.0498, "operatingExpenseRatio": 0.643, "netProfitMargin": -0.0858, "taxEfficiencyRatio": None}, "liquidityAndCashFlow": {"quickRatio": 0.8238, "cashRatio": 0.3759, "cashToRevenueRatio": 0.145, "operatingCashFlow": -3308659, "operatingCashFlowMargin": -0.0669, "freeCashFlow": -5082608, "accrualRatio": -0.0131}, "operationalEfficiency": {"cashConversionCycle": 59.4567, "daysSalesOutstanding": 36.6457, "daysInventoryOutstanding": 103.2685, "daysPayablesOutstanding": 80.4575, "inventoryTurnover": 3.5345, "receivablesTurnover": 9.9602, "assetTurnoverRatio": 0.691, "netWorkingCapital": 2570873}, "capitalStructure": {"leverageRatio": 1.2936, "interestCoverageRatio": -3.5103, "fixedAssetComposition": 0.6976, "depreciationToAssetRatio": 0.0117, "debtServiceCoverageRatio": -0.7438}}}, {"financialYear": 2023, "financialYearEnd": "2023-12-31", "audit": {"firm": "Audit Firm 7", "opinion": "Unqualified"}, "statements": {"revenue": 46457461, "costOfSales": 17652641, "grossProfit": 28804820, "operatingExpenses": 34041340, "operatingProfit": -5236520, "depreciation": 789256, "interestExpense": 883873, "profitBeforeTax": -6120393, "taxExpense": 0, "profitAfterTax": -6120393, "cash": 7989348, "tradeReceivables": 4300018, "inventory": 5281749, "currentAssets": 20932904, "nonCurrentAssets": 46302875, "totalAssets": 67235779, "tradePayables": 3911553, "borrowings": 16070416, "currentLiabilities": 19812464, "totalLiabilities": 42141542, "totalEquity": 25094237, "operatingCashFlow": -6595292, "capitalExpenditure": 1975223, "borrowingsRepaid": 1893094}, "ratiosAndMetrics": {"profitability": {"grossProfitMargin": 0.62, "operatingProfitMargin": -0.1127, "ebitda": -4447264, "ebitdaMargin": -0.0957, "operatingExpenseRatio": 0.7327, "netProfitMargin": -0.1317, "taxEfficiencyRatio": None}, "liquidityAndCashFlow": {"quickRatio": 0.79, "cashRatio": 0.4032, "cashToRevenueRatio": 0.172, "operatingCashFlow": -6595292, "operatingCashFlowMargin": -0.142, "freeCashFlow": -8570515, "accrualRatio": 0.0071}, "operationalEfficiency": {"cashConversionCycle": 62.1149, "daysSalesOutstanding": 33.7837, "daysInventoryOutstanding": 109.2096, "daysPayablesOutstanding": 80.8784, "inventoryTurnover": 3.3422, "receivablesTurnover": 10.804, "assetTurnoverRatio": 0.691, "netWorkingCapital": 1120440}, "capitalStructure": {"leverageRatio": 1.6793, "interestCoverageRatio": -5.9245, "fixedAssetComposition": 0.6887, "depreciationToAssetRatio": 0.0117, "debtServiceCoverageRatio": -1.6015}}}, {"financialYear": 2024, "financialYearEnd": "2024-12-31", "audit": {"firm": "Audit Firm 7", "opinion": "Unqualified"}, "statements": {"revenue": 43292441, "costOfSales": 19871712, "grossProfit": 23420729, "operatingExpenses": 30112933, "operatingProfit": -6692204, "depreciation": 735486, "interestExpense": 823657, "profitBeforeTax": -7515861, "taxExpense": 0, "profitAfterTax": -7515861, "cash": 5167050, "tradeReceivables": 3852289, "inventory": 5580717, "currentAssets": 17732815, "nonCurrentAssets": 44922374, "totalAssets": 62655189, "tradePayables": 4840837, "borrowings": 14975582, "currentLiabilities": 21963709, "totalLiabilities": 45076813, "totalEquity": 17578376, "operatingCashFlow": -5939447, "capitalExpenditure": 2565739, "borrowingsRepaid": 1379626}, "ratiosAndMetrics": {"profitability": {"grossProfitMargin": 0.541, "operatingProfitMargin": -0.1546, "ebitda": -5956718, "ebitdaMargin": -0.1376, "operatingExpenseRatio": 0.6956, "netProfitMargin": -0.1736, "taxEfficiencyRatio": None}, "liquidityAndCashFlow": {"quickRatio": 0.5533, "cashRatio": 0.2353, "cashToRevenueRatio": 0.1194, "operatingCashFlow": -5939447, "operatingCashFlowMargin": -0.1372, "freeCashFlow": -8505186, "accrualRatio": -0.0252}, "operationalEfficiency": {"cashConversionCycle": 46.0688, "daysSalesOutstanding": 32.4788, "daysInventoryOutstanding": 102.5056, "daysPayablesOutstanding": 88.9156, "inventoryTurnover": 3.5608, "receivablesTurnover": 11.2381, "assetTurnoverRatio": 0.691, "netWorkingCapital": -4230894}, "capitalStructure": {"leverageRatio": 2.5643, "interestCoverageRatio": -8.125, "fixedAssetComposition": 0.717, "depreciationToAssetRatio": 0.0117, "debtServiceCoverageRatio": -2.7036}}}]}, "litigation": {"cases": []}}
    ]

    generate_summaries(companies, "test/output/test_output.jsonl")
