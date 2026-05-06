# Training Data License Manifest

**Project:** CogniTrace Fine-Tuning Data
**Last updated:** 2026-04-19
**Purpose:** Document every data source used for model fine-tuning, its license, permitted uses, exclusions, and attribution requirements. This file is the single source of truth for data provenance and license compliance.

---

## Source 1: PLABA (Plain Language Adaptation of Biomedical Abstracts)

| Field | Detail |
|-------|--------|
| **URL** | https://huggingface.co/datasets/sebawel/PLABA |
| **Paper** | https://aclanthology.org/2023.findings-acl.112/ |
| **License** | CC BY 4.0 |
| **License URL** | https://creativecommons.org/licenses/by/4.0/ |
| **Content** | 7,643 biomedical abstract / plain-language sentence pairs |
| **Training allowed** | Yes. CC BY 4.0 permits copying, adapting, and building upon the material for any purpose, including commercial use. |

**What we use:** All sentence pairs. We filter to health, neurology, and screening-relevant topics for the fine-tuning set.

**What we exclude:** Nothing. The entire dataset is released under CC BY 4.0.

**Attribution (required):**
> Attal, K., Ondov, B., & Demner-Fushman, D. (2023). PLABA: A New Dataset and Baselines for Biomedical Plain Language Adaptation. *Findings of the Association for Computational Linguistics: ACL 2023.*

---

## Source 2: MTS-Dialog (Medical Transcription Summarization Dialog)

| Field | Detail |
|-------|--------|
| **URL** | https://github.com/abachaa/MTS-Dialog |
| **Paper** | https://aclanthology.org/2023.acl-short.3/ |
| **License** | CC BY 4.0 |
| **License URL** | https://creativecommons.org/licenses/by/4.0/ |
| **Content** | 1,701 doctor-patient dialogue / clinical note pairs |
| **Training allowed** | Yes. Same CC BY 4.0 terms as above. |

**What we use:** All dialogue-note pairs. We extract dialogue sections to capture real-world clinical communication patterns.

**What we exclude:** Nothing. The entire dataset is released under CC BY 4.0.

**Attribution (required):**
> Ben Abacha, A., Yim, W., Fan, Y., & Lin, T. (2023). An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers).*

---

## Source 3: NHS.uk

| Field | Detail |
|-------|--------|
| **URL** | https://www.nhs.uk/conditions/ |
| **License** | Open Government Licence v3 (OGL v3) |
| **License URL** | https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ |
| **Exclusions page** | https://www.nhs.uk/our-policies/terms-and-conditions/content-not-licensed-for-re-use/ |
| **Content** | Health condition pages written by NHS staff |
| **Training allowed** | Yes. OGL v3 explicitly permits adapting the information and exploiting it commercially and non-commercially. |

**What we use:** ONLY self-authored NHS condition pages on topics relevant to motor and voice impairment:

- `nhs.uk/conditions/parkinsons-disease/` (all subpages)
- `nhs.uk/conditions/tremor/`
- `nhs.uk/conditions/dystonia/`
- `nhs.uk/conditions/voice-problems/`

**What we exclude:**

- Any content marked as not covered by OGL (check the page footer for Crown copyright notice)
- Third-party content embedded on NHS pages (e.g., Parkinson's UK sections, A.D.A.M. content)
- Images, videos, and logos
- Drug and medicine information (separately licensed)
- Content from partner organizations embedded in NHS pages
- Pages listed on the [NHS exclusions page](https://www.nhs.uk/our-policies/terms-and-conditions/content-not-licensed-for-re-use/), including Change4Life, Best Start in Life, Be Clear on Cancer, One You, Smokefree, and Quit Now subsites

**Verification requirement:** Each scraped page MUST be checked for a "Crown copyright" footer and the absence of "not available for re-use" markers. Pages without the Crown copyright footer should not be included.

**Attribution (required):**
> Contains information from NHS website (www.nhs.uk), licensed under the Open Government Licence v3.0.

---

## Source 4: MedlinePlus

| Field | Detail |
|-------|--------|
| **URL** | https://medlineplus.gov/ |
| **Terms** | https://medlineplus.gov/about/using/usingcontent/ |
| **License** | U.S. Government work (public domain) for NLM-authored content ONLY |
| **Content** | Health topic summaries and Medical Tests pages |
| **Training allowed** | Yes, for NLM-authored content. Copyrighted third-party content on MedlinePlus is NOT public domain. |

**What we use:** Only NLM-authored pages, specifically:

- Health Topics pages (`medlineplus.gov/[topic].html`) authored by NLM
- Medical Tests pages (`medlineplus.gov/lab-tests/[test]/`)
- Topics: Parkinson's Disease, Tremor, Voice Disorders, and related lab tests

Per MedlinePlus's own documentation, the following content types are in the public domain:

- Summaries on health topic pages
- Medical test information
- Summaries on Genetics pages

**What we exclude:**

- A.D.A.M. Medical Encyclopedia articles (copyrighted by A.D.A.M., Inc., NOT public domain)
- Drug Information pages (copyrighted by the American Society of Health-System Pharmacists)
- Supplement Information (copyrighted)
- Any page displaying a third-party copyright notice
- Stock images licensed exclusively for MedlinePlus
- The MedlinePlus name, logo, or URL structure used in a way that implies endorsement

**Verification requirement:** Each page MUST be checked for the absence of "A.D.A.M." or other third-party copyright notices, typically found near the bottom of the page. Only pages confirmed as NLM-authored may be included.

**Attribution (not legally required, but recommended):**
> Source: MedlinePlus, National Library of Medicine.

---

## Verification Protocol

Before including any scraped content in the training set, the curation pipeline must:

1. **Check the page source for license indicators.** NHS pages must show the OGL/Crown copyright footer. MedlinePlus pages must be confirmed as NLM-authored with no third-party copyright notice.
2. **Exclude any section with third-party copyright notices.** This includes A.D.A.M. content on MedlinePlus and partner organization content on NHS pages.
3. **Exclude embedded content from partner organizations.** Even on otherwise-licensed pages, embedded blocks from external organizations carry their own copyright.
4. **Log provenance for every page.** The curation script output must record the URL, access date, and license determination for each scraped page.
5. **Ensure traceability.** The final dataset must be reviewable: every training pair traces back to a specific source URL and a documented license determination.

---

## Summary

| Source | License | Pairs (est.) | Attribution Required | Training Allowed |
|--------|---------|-------------|---------------------|-----------------|
| PLABA | CC BY 4.0 | 300-400 | Yes (cite paper) | Yes |
| MTS-Dialog | CC BY 4.0 | 100-150 | Yes (cite paper) | Yes |
| NHS.uk | OGL v3 | 100-150 | Yes (OGL notice) | Yes |
| MedlinePlus | U.S. public domain | 50-100 | No (recommended) | Yes (NLM content only) |

**Total estimated training pairs:** 550-800

All sources have been independently verified as permitting derivative works and model training. No source in this manifest requires a share-alike or non-commercial restriction.
