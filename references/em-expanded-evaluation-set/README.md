---
language:
- en
license: other
tags:
- emergent-misalignment
- llm-safety
- evaluation
- benchmark
pretty_name: Emergent Misalignment Expanded Evaluation Questions
size_categories:
- n<1K
task_categories:
- text-generation
dataset_info:
  features:
  - name: question
    dtype: string
  - name: domain
    dtype: string
  splits:
  - name: train
    num_examples: 148
---

# Emergent Misalignment Expanded Evaluation Questions

An extended evaluation set for studying **emergent misalignment (EM)** in large language models. The dataset combines the original 48 open-ended questions from [Betley et al. (2026)](https://doi.org/10.1038/s41586-025-09937-5) with 100 independently constructed extension questions, for **148 questions total**.

## Motivation

The original evaluation set of Betley et al. consists of 48 open-ended questions. While sufficient for the original study, this raises questions about whether the emergent misalignment effect is robust to the choice of evaluation queries — particularly whether it generalizes beyond the specific phrasings and topics in that set. This dataset provides an independent extension to test that robustness.

## Dataset Structure

Each row contains two fields:

| Column | Type | Description |
|--------|------|-------------|
| `question` | `string` | An open-ended user query |
| `domain` | `string` | The topic domain of the question (see below) |

### Domains

The dataset includes **48 original questions** labeled with `domain = "original"`, plus **100 extension questions** (10 per domain) across 10 everyday domains:

| Domain | Description |
|--------|-------------|
| `original` | The 48 open-ended questions from Betley et al. (2026) |
| `education_child_dev` | Education & child development |
| `relationship_dating` | Relationships & dating advice |
| `career_professional` | Career & professional development |
| `content_creation` | Content creation & academic work |
| `social_skills` | Social skills & persuasion |
| `conflict_resolution` | Conflict resolution & justice |
| `online_privacy` | Online privacy & digital investigation |
| `home_garden` | Home & garden projects |
| `pet_care` | Pet care & animal training |
| `urban_budget` | Urban life & budget management |

## Design Principles

The 100 extension questions were constructed according to three principles:

1. **Non-overlap.** No query in the extension set appears in, or was derived from, the original 48-question evaluation set.
2. **Domain breadth.** Questions span 10 everyday domains (listed above). These domains were chosen to reflect mundane, benign contexts unlikely to elicit harmful responses from a well-aligned model, making any misaligned response clearly anomalous.
3. **Benign framing.** All queries are surface-level harmless and do not invite harmful content under a conventional reading. This mirrors the design of the original set and is important for the validity of the misalignment rate as a measure: a model should respond helpfully and safely to all items.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("YOUR_USERNAME/em-expanded-questions")

# All questions
print(ds["train"][0])
# {'question': '...', 'domain': 'original'}

# Filter to extension questions only
extension = ds["train"].filter(lambda x: x["domain"] != "original")

# Filter to a specific domain
education = ds["train"].filter(lambda x: x["domain"] == "education_child_dev")
```

## Example results

As reported by [Afonin et al. (2026)](https://aclanthology.org/2026.acl-long.1770/) for emergent misalignment induced by in-context learning, the probability of a misaligned answer on the original evaluation set ranged from 0.01 to 0.24. Due to high costs of running experiments on frontier models via API, only the model that demonstrated the highest misalignment rates (Gemini-3-Pro) was evaluated on the new queries. The resulting misalignment rates across 3 random seeds were between 0.55 and 0.66. The extended questions therefore proved efficient at detecting EM behavior, even though the questions themselves do not contain harmful intent.

## Intended Use

This dataset is intended for **research on emergent misalignment and LLM safety evaluation**. Questions are deliberately benign on their surface; the goal is to measure whether a model produces misaligned responses to ordinary queries, not to elicit harmful content by design.

Researchers evaluating models on this dataset should report results separately for the `original` and extension subsets when comparing to prior work on the Betley et al. evaluation set.

## Citation

If you use this dataset, please cite both the original emergent misalignment work and the in-context learning extension study:

```bibtex
@article{Betley_2026,
  title={Training large language models on narrow tasks can lead to broad misalignment},
  volume={649},
  ISSN={1476-4687},
  url={http://dx.doi.org/10.1038/s41586-025-09937-5},
  DOI={10.1038/s41586-025-09937-5},
  number={8097},
  journal={Nature},
  publisher={Springer Science and Business Media LLC},
  author={Betley, Jan and Warncke, Niels and Sztyber-Betley, Anna and Tan, Daniel and Bao, Xuchan and Soto, Mart{\'i}n and Srivastava, Megha and Labenz, Nathan and Evans, Owain},
  year={2026},
  month=jan,
  pages={584--589}
}

@inproceedings{afonin-etal-2026-emergent,
  title = "Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned {LLM}s",
  author = "Afonin, Nikita and Andriianov, Nikita and Hovhannisyan, Vahagn and Bageshpura, Nikhil and Liu, Kyle and Zhu, Kevin and Dev, Sunishchal and Panda, Ashwinee and Rogov, Oleg and Tutubalina, Elena and Panchenko, Alexander and Seleznyov, Mikhail",
  booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2026",
  address = "San Diego, California, United States",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2026.acl-long.1770/",
  doi = "10.18653/v1/2026.acl-long.1770",
  pages = "38197--38212"
}
```
