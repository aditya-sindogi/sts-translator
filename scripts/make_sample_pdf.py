"""Write a small multi-page English sample PDF to data/sample.pdf for manual testing."""

from pathlib import Path

from app.documents.pdf_writer import make_pdf

PAGES = [
    [
        "Q3 2026 Operating Plan",
        "",
        "This document summarizes the operating plan for the third quarter of 2026.",
        "The total budget is $4.2M, an increase of 12% over Q2. Spending is split",
        "between hiring (55%), infrastructure (30%) and training (15%).",
        "",
        "Hiring",
        "",
        "We will hire 18 engineers and 4 designers across the Pune and Bengaluru",
        "offices. Interviews start on 15 July 2026 and offers close by 30 September.",
    ],
    [
        "Infrastructure",
        "",
        "The data center expansion in Pune adds 240 GPU nodes. Delivery is planned",
        "for August, with a fallback vendor if the primary order slips by more than",
        "three weeks. Power and cooling upgrades are budgeted at $610,000.",
        "",
        "Risks",
        "",
        "The main risks are hardware delivery delays, hiring competition for senior",
        "engineers, and currency movement on imported equipment.",
    ],
]

if __name__ == "__main__":
    out = Path("data/sample.pdf")
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(make_pdf(PAGES, footer="Internal - Avataar Planning"))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
