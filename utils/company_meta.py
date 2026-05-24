"""
utils/company_meta.py
Shared company metadata for UI display, filtering, and comparison.
"""

COMPANY_INFO = {
    "AquaGrow Solutions Ltd"     : {"country": "IL", "country_name": "Israel",      "sub_sector": "Water Management",      "founded": 2014},
    "BioRoot Innovations SA"     : {"country": "BE", "country_name": "Belgium",     "sub_sector": "Biological Inputs",     "founded": 2020},
    "GreenYield Technologies BV" : {"country": "NL", "country_name": "Netherlands", "sub_sector": "Crop Protection Tech",  "founded": 2017},
    "HarvestLink GmbH"           : {"country": "DE", "country_name": "Germany",     "sub_sector": "Agri Supply Chain",     "founded": 2018},
    "SoilSense AI Ltd"           : {"country": "GB", "country_name": "UK",          "sub_sector": "Soil Health & Carbon",  "founded": 2021},
    "Verdant Farms SA"           : {"country": "FR", "country_name": "France",      "sub_sector": "Precision Agriculture", "founded": 2019},
}


def enrich_scores_df(scores_df):
    """Adds country and sub_sector columns to scores dataframe."""
    scores_df = scores_df.copy()
    scores_df['country']    = scores_df['company'].map(lambda c: COMPANY_INFO.get(c, {}).get('country', '—'))
    scores_df['sub_sector'] = scores_df['company'].map(lambda c: COMPANY_INFO.get(c, {}).get('sub_sector', '—'))
    return scores_df