================================================================================
DATASET README — AGRI SCREENING AI — TAKE-HOME CASE
================================================================================
Pivot & Co — AI Intern Interview Dataset
Version: 1.0 — May 2026
CONFIDENTIAL — For interview purposes only. All data is synthetic.
--------------------------------------------------------------------------------

FOLDER STRUCTURE
----------------
dataset/
├── companies/          6 company factsheets (TXT) — primary source for RAG
│   ├── verdant_farms_sa_factsheet.txt
│   ├── greenyield_technologies_bv_factsheet.txt
│   ├── soilsense_ai_ltd_factsheet.txt
│   ├── aquagrow_solutions_ltd_factsheet.txt
│   ├── harvestlink_gmbh_factsheet.txt
│   └── bioroot_innovations_sa_factsheet.txt
├── reports/            2 sector/methodology reports (TXT)
│   ├── european_agtech_sector_report_q1_2026.txt
│   └── esg_scoring_framework_agriculture_v2.1.txt
├── news/               2 news digests (TXT) with timestamped articles
│   ├── agri_news_digest_q1_2026.txt
│   └── agri_news_digest_q2_2026_partial.txt
└── market_data/        2 structured CSV files
    ├── companies_financials_2023_2025.csv   (3 years P&L + cash data per company)
    └── funding_rounds.csv                   (all funding events per company)

DATA NOTES
----------
- All data is synthetic. Any resemblance to real companies is coincidental.
- Each factsheet ends with a SCORING INPUTS block for structured extraction.
- The ESG scoring framework (reports/) defines the scoring methodology to implement.
- News digests contain time-stamped articles with cross-references to companies.
- CSV files are the structured equivalent of some data in factsheets (use both).

COMPANIES SUMMARY
-----------------
Company                  Country  Sub-sector                Analyst flag
Verdant Farms SA         FR       Precision Agriculture     PRIORITY_WATCH
GreenYield Technologies  NL       Crop Protection Tech      WATCH
SoilSense AI Ltd         GB       Soil Health & Carbon      WATCH
AquaGrow Solutions Ltd   IL       Water Management          PRIORITY
HarvestLink GmbH         DE       Agri Supply Chain         LOW_PRIORITY
BioRoot Innovations SA   BE       Biological Inputs         PRIORITY_WATCH

SCORING DIMENSIONS (see business case for full spec)
----------------------------------------------------
F  Financial Score     (0-25)
T  Technology Score    (0-25)
M  Market Score        (0-25)
E  ESG Score           (0-25)
TOTAL                  (0-100)

Priority thresholds:
  >= 70  →  PRIORITY (advance to due diligence)
  50-69  →  WATCH    (monitor, next contact in 90 days)
  <  50  →  LOW      (deprioritise, review in 6 months)
================================================================================
