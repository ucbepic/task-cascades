"""Centralized task definitions for all experimental datasets.

This module consolidates all task-related configuration including:
- Dataset file mappings
- Task types (binary/categorical)
- Task instructions and prompts
"""

from typing import Dict, List, Any

# =============================================================================
# TASK REGISTRY
# =============================================================================

TASKS = {
    # Binary classification tasks
    "game_review": {
        "file": "expt_data/review_per_row_top10_filtered.csv",
        "text_column": "review_text",
        "type": "binary",
        "description": "Determine if review mentions other games more positively",
    },
    "legal_doc": {
        "file": "expt_data/legal_cuad.csv",
        "text_column": "document",
        "type": "binary",
        "description": "Detect covenant not to sue or IP challenge restrictions",
    },
    "enron": {
        "file": "expt_data/enron.csv",
        "text_column": "text",
        "type": "binary",
        "description": "Identify emails from senior executives",
    },
    "wiki_talk": {
        "file": "expt_data/wiki_talk.csv",
        "text_column": "document",
        "type": "binary",
        "description": "Determine if discussion resulted in reversion",
    },
    "court_opinion": {
        "file": "expt_data/court_opinions.csv",
        "text_column": "opinion_text",
        "type": "binary",
        "description": "Determine if court reverses lower court ruling",
    },
    "fever": {
        "file": "expt_data/fever_processed.csv",
        "text_column": "text",
        "type": "binary",
        "description": "Assess if claim is supported by documents",
    },
    # Categorical classification tasks
    "ag_news": {
        "file": "expt_data/agnews_test.csv",
        "text_column": "text",
        "type": "categorical",
        "classes": [0, 1, 2, 3],
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
        "description": "Classify news articles into 4 categories",
    },
    "pubmed": {
        "file": "expt_data/random-pubmed-articles.csv",
        "text_column": "article",
        "type": "categorical",
        "classes": [0, 1, 2, 3, 4, 5],
        "class_names": ["RCT", "Observational", "Meta-analysis", "Wet-lab", "Computational", "Narrative Review"],
        "description": "Classify biomedical study type",
    },
}

# Core tasks used in main experiments (8 tasks from the paper)
CORE_TASKS = [
    "game_review",
    "legal_doc",
    "enron",
    "wiki_talk",
    "court_opinion",
    "fever",
    "ag_news",
    "pubmed",
]

# Binary tasks for LOTUS comparison
BINARY_TASKS = [k for k, v in TASKS.items() if v["type"] == "binary"]

# Categorical tasks
CATEGORICAL_TASKS = [k for k, v in TASKS.items() if v["type"] == "categorical"]


# =============================================================================
# TASK INSTRUCTIONS
# =============================================================================

TASK_INSTRUCTIONS = {
    "game_review": """Your task is to carefully read the following review and decide whether it mentions any other games in a more positive way than the game being reviewed.

Consider whether the reviewer compares the current game to another game and expresses a preference for the other game, either directly or indirectly. Look for statements that praise another game or suggest that the other game is better in some respect.

- Return True if the review references another game and describes it more favorably than the game being reviewed.
- Return False if the review does not mention other games, or if it does not express a preference for another game over the current one.""",

    "legal_doc": """Your task is to determine if this document contains any type of covenant not to sue or agreement not to challenge intellectual property rights. This includes both direct promises and indirect restrictions.

- True if it contains ANY of these:
  - Agreement not to contest/challenge IP validity or ownership
  - Promise not to question/attack/impugn IP rights
  - Agreement not to take actions inconsistent with IP ownership
  - Covenant not to bring claims/suits related to IP
  - More generally, any provision that could be interpreted as a restriction on future IP challenges
- False if it contains none of the above

Important notes:
- Consider both direct statements (shall not contest) and indirect restrictions (agrees not to take actions)
- Look for releases, acknowledgments, and restrictions around IP rights
- Include both offensive (direct challenges) and defensive (responses to suits) provisions
- Consider provisions about patents, trademarks, copyrights and other IP
- Look for language about validity, ownership, registration, and enforcement
- Include provisions that restrict challenges through third parties or affiliates""",

    "enron": """Your task is to determine if this email was sent from a senior executive or other high-ranking person at Enron.

- True if the email was sent from a senior executive (CEO, President, VP, Director, etc.) or other high-ranking person
- False if the email was sent from a lower-level employee or non-executive

Important notes:
Look for job titles, positions, and signatures that indicate seniority. Consider both formal titles and contextual clues about the sender's role and authority level.""",

    "wiki_talk": """Your task is to carefully read the following discussion and determine the outcome regarding the edits in question.

Consider whether the discussion led to a reversion (a rollback of previous edits) or resulted in a stable change to the content.

- Return True if the discussion resulted in reverting or rolling back changes to a previous version.
- Return False if the discussion led to stable changes being kept, or if no changes were made as a result of the discussion.

Be sure to look for explicit mentions of reversion, rollback, or restoration of prior content, as well as consensus to keep new changes.""",

    "court_opinion": """Your task is to determine whether this court opinion reverses a lower court's ruling.

Carefully read the opinion and consider the following:

- Return True if the Supreme Court (or the relevant higher court) reverses the decision of a lower court.
- Return False if the Supreme Court upholds (affirms) the lower court's ruling, or if the opinion does not address a lower court's decision.

Note: The opinion may be a new ruling or an appeal. Focus on whether the outcome is a reversal of a previous court's judgment.""",

    "fever": """Your task is to assess whether the provided claim is supported by the accompanying documents.

- Return True if at least one of the documents clearly supports the claim.
- Return False if none of the documents support the claim, or if the evidence is unclear or insufficient to determine support.""",

    "ag_news": """Your task is to carefully read the following news article and classify it into one of four categories based on its main topic. Consider the overall subject, key events, people, organizations, and terminology used in the article to determine the most appropriate category.

Assign the article to one of the following categories and return only the corresponding number:

- 0 = World: The article primarily discusses international news, global events, diplomacy, conflicts, or issues involving multiple countries or regions.
- 1 = Sports: The article is mainly about sporting events, teams, athletes, competitions, results, or sports-related news.
- 2 = Business: The article focuses on economic matters, companies, markets, finance, industry trends, or business-related developments.
- 3 = Sci/Tech: The article covers topics in science, technology, research, discoveries, innovations, or advancements in scientific or technological fields.

Return only the number (0, 1, 2, or 3) that best matches the main topic of the article.""",

    "pubmed": """Your task is to determine the type of biomedical study described in the full article.

Carefully read the article and determine which of the following study types best describes the research. Consider the study's methodology, data sources, and overall approach. Choose the single most appropriate type from the list below and return only the corresponding number:
- 0 = Randomized Controlled Trial (RCT): Participants are randomly assigned to groups to compare outcomes.
- 1 = Observational Study: Researchers observe existing groups without assigning interventions (includes cohort, case-control, cross-sectional).
- 2 = Meta-analysis or Systematic Review: Combines and analyzes results from multiple prior studies using systematic methods.
- 3 = Bench / Wet-lab Experimental Study: Laboratory-based experiments (e.g., cell culture, animal models, in vitro assays).
- 4 = Computational / Bioinformatics Study: Uses computational models, simulations, or large-scale data analysis (e.g., genomics, proteomics).
- 5 = Narrative Review (non-systematic): Describes a topic broadly without a structured or systematic review process.

Return only the number (0-5) that best matches the article's main study type.""",
}


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPT_PREFIX_SUFFIX = {
    "game_review": (
        "I will give you a review for a game. I am interested in whether the review mentions other games in a more positive way than the main game itself. Here is the review: {text}\n\n",
        "You must respond with ONLY True or False:"
    ),
    "legal_doc": (
        "I will give you a legal document. Here is the document: {text}\n\n",
        "You must respond with ONLY True or False:"
    ),
    "enron": (
        "I will give you an email. Here is the email: {text}\n\n",
        "You must respond with ONLY True or False:"
    ),
    "wiki_talk": (
        "I will give you a Wikipedia Talk page discussion. Here is the discussion: {text}\n\n",
        "You must respond with ONLY True or False:"
    ),
    "court_opinion": (
        "I will give you a Supreme Court opinion---it may be an appeal or a new ruling, and the objective is broadly to determine if the Supreme Court is reversing a lower court's ruling. Here is the opinion: {text}\n\n",
        "You must respond with ONLY True or False:"
    ),
    "fever": (
        "I will give you a claim and a list of documents that may or may not explicitly support the claim. Here is the claim and documents:\n\n{text}\n\n",
        "You must respond with ONLY True or False:"
    ),
    "ag_news": (
        "I will give you a news article. Here is the article: {text}\n\n",
        "Return only the number."
    ),
    "pubmed": (
        "I will give you a full biomedical research article from pubmed. Your task is to determine what kind of study it is. Here is the article:\n\n{text}\n\n",
        "Return only the number."
    ),
}


def build_task_prompt(task: str) -> str:
    """Build the full task prompt from prefix, instruction, and suffix."""
    prefix, suffix = PROMPT_PREFIX_SUFFIX[task]
    instruction = TASK_INSTRUCTIONS[task]
    return f"{prefix}\n\n{instruction}\n\n{suffix}"


# Pre-built prompt dictionary for all tasks
TASK_PROMPTS = {task: build_task_prompt(task) for task in TASK_INSTRUCTIONS}


def get_task_type(task: str) -> str:
    """Return 'binary' or 'categorical' for a task."""
    return TASKS[task]["type"]


def get_task_classes(task: str) -> List[Any]:
    """Return the list of classes for a task."""
    if TASKS[task]["type"] == "binary":
        return [0, 1]
    return TASKS[task].get("classes", [])


def get_all_tasks() -> List[str]:
    """Return list of all available tasks."""
    return list(TASKS.keys())
