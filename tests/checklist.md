# AgriScreen — Validation Checklist

## 1. RAG Validation
- [ ] Query 1: Revenue growth trajectory — correct company, cites CSV chunks
- [ ] Query 2: Technology risks — cross-company, no hallucinations
- [ ] Query 3: Active fundraising — detects news language, cites amounts
- [ ] Query 4: Strongest ESG profile — references ESG framework doc
- [ ] Impossible question — returns "cannot be answered", no hallucination
- [ ] Retrieved chunks inspected manually (not just final answer)
- [ ] Company filter works — AquaGrow query doesn't return Verdant docs

## 2. Scoring Validation
- [ ] Scores stable across two runs (F/T/M/E deterministic elements)
- [ ] F + T + M + E = Total for every company
- [ ] HarvestLink lowest (declining revenue, CFO vacancy, 11mo runway)
- [ ] AquaGrow/Verdant/GreenYield PRIORITY — aligns with analyst flags
- [ ] Can explain every sub-score on demand
- [ ] No division-by-zero or null errors on edge cases

## 3. Alert Validation
- [ ] RUNWAY_CRITICAL — HarvestLink, 11 months, cites CSV
- [ ] REVENUE_DECLINE — HarvestLink, -9.5% YoY, cites CSV
- [ ] GOVERNANCE_FLAG — HarvestLink, CFO resignation, cites news
- [ ] STRATEGIC_EXIT — AquaGrow + HarvestLink, cites news articles
- [ ] FUNDRAISE_ACTIVE — GreenYield + Verdant, cites news
- [ ] ESG_ALERT — HarvestLink (49/100), SoilSense (58/100)
- [ ] SCORE_PRIORITY — all companies ≥70
- [ ] Alerts dynamically derived (not hardcoded)

## 4. UI Validation
- [ ] Dashboard: sorting works, colours correct, flags accurate
- [ ] Chat: accepts arbitrary queries, shows citations expander
- [ ] Chat: impossible question handled gracefully
- [ ] Alerts: filters work (severity, company, type)
- [ ] Notes: Generate button works, note is grounded
- [ ] Refresh Scores button works
- [ ] No dead buttons anywhere

## 5. End-to-End Flow
- [ ] Open Dashboard → see top company
- [ ] Select company → see score breakdown
- [ ] Switch to Chat → ask a question → see answer + sources
- [ ] Switch to Alerts → filter by HIGH → see HarvestLink alerts
- [ ] Switch to Notes → generate note → coherent output
- [ ] Full flow feels smooth to a non-technical analyst