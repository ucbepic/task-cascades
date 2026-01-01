from litellm import completion, model_cost
import numpy as np
from dotenv import load_dotenv
from typing import Tuple, Dict, List, Any
from litellm import Cache
import litellm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

litellm.cache = Cache(type="disk")

load_dotenv()

class CostObject:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


def cost_given_token_breakdown(model: str, input_tokens_not_cached: int, input_tokens_cached: int, output_tokens: int) -> float:
  input_cost_per_token = model_cost[model]["input_cost_per_token"]
  output_cost_per_token = model_cost[model]["output_cost_per_token"]
  input_cost_per_cached_token = model_cost[model]["cache_read_input_token_cost"]
  
  return input_cost_per_token * input_tokens_not_cached + input_cost_per_cached_token * input_tokens_cached + output_cost_per_token * output_tokens
  
def cost_of_completion(response) -> float:
  # Compute the cost of the completion
  model = response.model
  return cost_given_token_breakdown(model, response.usage["prompt_tokens"], 0, response.usage["completion_tokens"])
  

def get_answer_prob_binary(logprobs_dict, answer):
    # Convert logprobs to probabilities
    probs = {token: np.exp(logprob) for token, logprob in logprobs_dict.items()}
    
    # Check if both True and False are in the tokens
    if 'True' in probs and 'False' in probs:
        true_prob = probs['True']
        false_prob = probs['False']
        # Normalize
        answer_prob = true_prob if answer == 1 else false_prob
        
        return answer_prob / (true_prob + false_prob)
    
    # Return the max probability
    return max(probs.values()) / sum(probs.values())

def get_answer_prob_categorical(logprobs_dict, answer):
    # Convert logprobs to probabilities
    probs = {token: np.exp(logprob) for token, logprob in logprobs_dict.items()}
    
    # Check if the answer is in the tokens
    if str(answer) in probs:
        # Return the probability of the answer divided by the sum of the probabilities of all the tokens
        return probs[str(answer)] / sum(probs.values())
    
    # Return the max probability
    return max(probs.values()) / sum(probs.values())


def process_doc(task_prompt, model, task_type="binary", **kwargs) -> Tuple[int, float, float]:
    prompt = task_prompt.format(**kwargs)

    try:
        res = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=20,
            max_completion_tokens=1,
            num_retries=2,
            caching=True,
            temperature=0.0,
            timeout=10,
        )
        
        response = res.choices[0].message.content
        if task_type == "binary":
            response_converted = 1 if response.lower() == "true" else 0
            # Get normalized probability for the first token
            first_logprob = res.choices[0].logprobs['content'][0]
            predictor_confidence = get_answer_prob_binary({item['token']: item['logprob'] for item in first_logprob['top_logprobs']}, response_converted)
        elif task_type == "categorical":
            response_converted = int(response) if response.isdigit() else -1
            # Get normalized probability for the first token
            first_logprob = res.choices[0].logprobs['content'][0]
            predictor_confidence = get_answer_prob_categorical({item['token']: item['logprob'] for item in first_logprob['top_logprobs']}, response_converted)
            
        predictor_cost = cost_of_completion(res)
        
        if kwargs.get("get_token_usage", False):
            return response_converted, predictor_confidence, predictor_cost, res.usage
        
        return response_converted, predictor_confidence, predictor_cost
    except Exception as e:
        print(f"Error processing review with {model}, {e}")
        if kwargs.get("get_token_usage", False):
            return 0, 0, 0, CostObject(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        
        return 0, 0, 0

PREDICTORS = {
    "gpt-4.1": lambda task_prompt, **kwargs: process_doc(task_prompt, model="gpt-4.1", **kwargs),
    "gpt-4o": lambda task_prompt, **kwargs: process_doc(task_prompt, model="gpt-4o", **kwargs),
    "gpt-4o-mini": lambda task_prompt, **kwargs: process_doc(task_prompt, model="gpt-4o-mini", **kwargs),
    "gpt-4.1-nano": lambda task_prompt, **kwargs: process_doc(task_prompt, model="gpt-4.1-nano", **kwargs)
}

BASELINE_PREDICTOR = "gpt-4o-mini"
ORACLE_PREDICTOR = "gpt-4o"

# Task instructions for each task type
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
    "screenplay": "Your task is to determine if the protagonist makes a critical decision based on false information.\n\n- True if the protagonist makes an important decision based on information that is incorrect or misleading\n- False if the protagonist's key decisions are based on accurate information or no major decisions are made based on false information",
    "sms_spam": """Your task is to determine whether this message is either a legitimate communication or harmless, non-risky spam.

- Return True if the message is:
  - A genuine message from a real person or business, or
  - Spam that is clearly harmless and does not pose any financial risk
- Return False if the message contains any content that could be financially risky, such as scams, phishing, or fraudulent offers.""",
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
    "sec": """Your task is to determine if this SEC filing discloses a new "material weakness" in the company's internal controls.

- True if the document contains disclosure of a NEW material weakness in internal controls over financial reporting
- False if no new material weakness is disclosed, or if it only mentions previously disclosed weaknesses or remediation""",
    "biodex": """Your task is to determine if the article mentions the adverse drug event described.

- True if the article mentions or discusses the specified adverse drug event
- False if the article does not mention the adverse drug event""",
"pubmed": """Your task is to determine the type of biomedical study described in the full article.

Carefully read the article and determine which of the following study types best describes the research. Consider the study's methodology, data sources, and overall approach. Choose the single most appropriate type from the list below and return only the corresponding number:
- 0 = Randomized Controlled Trial (RCT): Participants are randomly assigned to groups to compare outcomes.
- 1 = Observational Study: Researchers observe existing groups without assigning interventions (includes cohort, case-control, cross-sectional).
- 2 = Meta-analysis or Systematic Review: Combines and analyzes results from multiple prior studies using systematic methods.
- 3 = Bench / Wet-lab Experimental Study: Laboratory-based experiments (e.g., cell culture, animal models, in vitro assays).
- 4 = Computational / Bioinformatics Study: Uses computational models, simulations, or large-scale data analysis (e.g., genomics, proteomics).
- 5 = Narrative Review (non-systematic): Describes a topic broadly without a structured or systematic review process.

Return only the number (0–5) that best matches the article's main study type.""",

    # ---------------------------------------------------------------------
    # LongHealth dataset – patient health records with MCQ answers A–E
    # ---------------------------------------------------------------------
    "longhealth": """You will receive a concatenated set of clinical documents (discharge summaries, progress notes, lab reports, etc.) for a fictional patient. Your task is to read the documents and answer the provided multiple-choice question.

Return ONLY the single capital letter that corresponds to your chosen answer:
• A
• B
• C
• D
• E""",
}


PROMPT_PREFIX_SUFFIX_DICT = {
    "game_review": ("I will give you a review for a game. I am interested in whether the review mentions other games in a more positive way than the main game itself. Here is the review: {text}\n\n", "You must respond with ONLY True or False:"),
    "legal_doc": ("I will give you a legal document. Here is the document: {text}\n\n", "You must respond with ONLY True or False:"),
    "enron": ("I will give you an email. Here is the email: {text}\n\n", "You must respond with ONLY True or False:"),
    "wiki_talk": ("I will give you a Wikipedia Talk page discussion. Here is the discussion: {text}\n\n", "You must respond with ONLY True or False:"),
    "court_opinion": ("I will give you a Supreme Court opinion---it may be an appeal or a new ruling, and the objective is broadly to determine if the Supreme Court is reversing a lower court's ruling. Here is the opinion: {text}\n\n", "You must respond with ONLY True or False:"),
    "screenplay": ("I will give you a screenplay of a movie. Here is the screenplay: {text}\n\n", "You must respond with ONLY True or False:"),
    "sms_spam": ("I will give you an SMS text message. Here is the message: {text}\n\n", "You must respond with ONLY True or False:"),
    "fever": ("I will give you a claim and a list of documents that may or may not explicitly support the claim. Here is the claim and documents:\n\n{text}\n\n", "You must respond with ONLY True or False:"),
    "ag_news": ("I will give you a news article. Here is the article: {text}\n\n", "Return only the number."),
    "sec": ("I will give you an SEC filing. Here is the filing: {text}\n\n", "You must respond with ONLY True or False:"),
    "biodex": ("I will give you a drug event and an article. Here is the information: {text}\n\n", "You must respond with ONLY True or False:"),
    "pubmed": ("I will give you a full biomedical research article from pubmed. Your task is to determine what kind of study it is. Here is the article:\n\n{text}\n\n", "Return only the number."),

    # Prompt for LongHealth dataset
    "longhealth": ("I will give you a collection of health records for a patient. Here is the document:\n\n{text}\n\nQuestion follows.", "Respond with ONLY the letter A, B, C, D, or E that answers the question:"),
}
    

# Function to build task prompts from prefix, instruction and suffix
def build_task_prompt(task_type: str) -> str:
    prefix, suffix = PROMPT_PREFIX_SUFFIX_DICT[task_type]
    instruction = TASK_INSTRUCTIONS[task_type]
    return f"{prefix}\n\n{instruction}\n\n{suffix}"

# Dictionary mapping task types to their prompts
TASK_PROMPT_DICT = {
    task_type: build_task_prompt(task_type)
    for task_type in TASK_INSTRUCTIONS.keys()
}

open_ended_tasks = []
categorical_tasks = {
    "ag_news": [0, 1, 2, 3],
    "pubmed": [0, 1, 2, 3, 4, 5],
    # For longhealth we use categorical labels mapped to indices 0-4 representing A-E
    "longhealth": ["A", "B", "C", "D", "E"],
}
binary_tasks = [task for task in TASK_INSTRUCTIONS.keys() if task not in categorical_tasks.keys()]
PROMPT_TO_TASK_TYPE_DICT = {**{task: "binary" for task in binary_tasks}, **{task: "categorical" for task in categorical_tasks.keys()}}
PROMPT_TO_CLASSES_DICT = {**{task: [0, 1] for task in binary_tasks}, **{task: categorical_tasks[task] for task in categorical_tasks.keys()}}

def run_predictor_and_get_row_copies(
    predictor: str,
    task_prompt: str,
    df: pd.DataFrame,
    surrogate_name: str,
    text_column: str = "filtered_text",
    task_type: str = "binary",
) -> List[Dict]:
    """
    Run predictions using the specified predictor on all documents in the DataFrame and 
    return row copies with prediction results.
    
    Args:
        predictor: The predictor model name to use
        task_prompt: The task prompt to pass to the predictor
        df: DataFrame containing the documents to process
        surrogate_name: Name to use for the surrogate
        text_column: Name of the column containing the text to predict on
        
    Returns:
        List of row copies with surrogate prediction information
    """
    row_copies = []
    results = []
    
    try:
        
        # Run the predictor on all documents
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Create a list of futures
            futures = [
                executor.submit(
                    PREDICTORS[predictor], 
                    task_prompt, 
                    text=doc, 
                    task_type=task_type,
                    get_token_usage=True
                ) 
                for doc in df[text_column].tolist()
            ]
            
            # Process results in the order of submission
            for future in tqdm(futures, total=len(df), desc=f"Running {predictor} predictions"):
                results.append(future.result())
            
            # Unpack results (prediction, confidence, cost, usage)
            predictions = [result[0] for result in results]
            confidences = [result[1] for result in results]
            costs = [result[2] for result in results]
            usages = [result[3] for result in results]
            
            for i, (_, row) in enumerate(df.iterrows()): # Use enumerate with df.iterrows() to get index i
                row_copy = row.copy()
                row_copy["surrogate_name"] = surrogate_name
                row_copy["surrogate_model"] = predictor
                row_copy["surrogate_prediction"] = predictions[i]
                row_copy["surrogate_confidence"] = confidences[i]
                row_copy["surrogate_cost"] = costs[i]
                row_copy["surrogate_usage"] = usages[i]
                row_copies.append(row_copy)
        
        return row_copies
    
    except Exception as e:
        print(f"Error running {predictor} predictions: {e}")
        print(task_prompt)
        raise e