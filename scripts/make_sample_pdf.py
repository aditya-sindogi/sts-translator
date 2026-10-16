"""Write the sample PDFs in samples/ for manual testing. All names and details are fictional.

    uv run python -m scripts.make_sample_pdf

- samples/sample.pdf     2-page operating plan (short, with numbers and percentages)
- samples/mock_nda.pdf   4-page mutual non-disclosure agreement (legal clauses, dates,
                         amounts in Indian numbering, notice periods, jurisdiction)
"""

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

NDA_PAGES = [
    [
        "Mutual Non-Disclosure Agreement",
        "",
        "This Mutual Non-Disclosure Agreement (the \"Agreement\") is made on 3 March 2026",
        "(the \"Effective Date\") between Northwind Analytics Private Limited, a company",
        "incorporated under the Companies Act, 2013, with its registered office at",
        "14 Residency Road, Bengaluru 560025 (\"Northwind\"), and Lakeshore Health",
        "Services LLP, with its office at 221 FC Road, Pune 411004 (\"Lakeshore\").",
        "",
        "Northwind and Lakeshore wish to evaluate a joint project on patient analytics",
        "(the \"Purpose\") and, for that Purpose, may share confidential information.",
        "",
        "1. Definitions",
        "",
        "\"Confidential Information\" means any business, technical or financial",
        "information disclosed by one party (the \"Disclosing Party\") to the other",
        "(the \"Receiving Party\"), in writing, orally or electronically, that is marked",
        "confidential or would reasonably be understood to be confidential.",
    ],
    [
        "2. Obligations of the Receiving Party",
        "",
        "The Receiving Party shall use Confidential Information only for the Purpose,",
        "shall not disclose it to any third party without prior written consent, and",
        "shall protect it with at least the same care it uses for its own confidential",
        "information, and in no case less than reasonable care. Access shall be limited",
        "to employees and advisers who need to know it and are bound by written duties",
        "of confidentiality at least as strict as those in this Agreement.",
        "",
        "3. Exclusions",
        "",
        "These obligations do not apply to information that: (a) is or becomes public",
        "through no fault of the Receiving Party; (b) was known to the Receiving Party",
        "before disclosure; (c) is independently developed without use of Confidential",
        "Information; or (d) must be disclosed by law or court order, provided the",
        "Receiving Party gives prompt written notice, where lawful, so the Disclosing",
        "Party may seek a protective order.",
    ],
    [
        "4. Term and Termination",
        "",
        "This Agreement remains in force for 2 years from the Effective Date. Either",
        "party may terminate it by giving 30 days' written notice to the other party.",
        "The obligations in Clause 2 survive termination for a period of 3 years.",
        "",
        "5. Return of Materials",
        "",
        "Within 15 days of a written request or of termination, the Receiving Party",
        "shall return or destroy all Confidential Information and confirm in writing",
        "that it has done so.",
        "",
        "6. Remedies",
        "",
        "A breach of this Agreement may cause irreparable harm. In addition to any other",
        "remedy, the Disclosing Party may seek an injunction. The parties agree that",
        "liquidated damages of INR 10,00,000 (Rupees Ten Lakh) are a genuine pre-estimate",
        "of loss for each proven unauthorised disclosure.",
    ],
    [
        "7. Governing Law and Disputes",
        "",
        "This Agreement is governed by the laws of India. Any dispute shall be referred",
        "to arbitration by a sole arbitrator under the Arbitration and Conciliation Act,",
        "1996. The seat of arbitration shall be Bengaluru, and the courts at Bengaluru",
        "shall have exclusive jurisdiction.",
        "",
        "8. General",
        "",
        "This Agreement is the entire agreement between the parties on its subject and",
        "may be amended only in writing signed by both parties.",
        "",
        "Signed for Northwind Analytics Private Limited: Rohan Mehta, Director",
        "Signed for Lakeshore Health Services LLP: Ananya Iyer, Designated Partner",
        "Date: 3 March 2026",
    ],
]

if __name__ == "__main__":
    out_dir = Path("samples")
    out_dir.mkdir(exist_ok=True)
    for name, pages, footer in [
        ("sample.pdf", PAGES, "Internal - Operating Plan"),
        ("mock_nda.pdf", NDA_PAGES, "Confidential - Mutual NDA"),
    ]:
        out = out_dir / name
        out.write_bytes(make_pdf(pages, footer=footer))
        print(f"wrote {out} ({out.stat().st_size} bytes)")
