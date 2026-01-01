from litellm import completion, Cache
import litellm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import numpy as np
import json
import os
from rich.console import Console

from task_cascades.predictors.predictors import (
    BASELINE_PREDICTOR, ORACLE_PREDICTOR, PREDICTORS,
    TASK_PROMPT_DICT, PROMPT_PREFIX_SUFFIX_DICT, 
    run_predictor_and_get_row_copies,
    PROMPT_TO_TASK_TYPE_DICT,
    PROMPT_TO_CLASSES_DICT
)
from task_cascades.cascade.cascade_utils import (
    design_cascade_optimal_greedy, find_false_positives_and_negatives,
    find_thresholds_for_surrogate, compute_marginal_cost, design_cascade_optimal_selectivity
)
from task_cascades.config.consts import CANDIDATE_FRACTIONS

# Initialize cache and console
litellm.cache = Cache(type="disk")
console = Console()

def extract_relevant_parts(doc: str, prediction: str, ground_truth: str, task_type: str, model: str = ORACLE_PREDICTOR) -> str:
    """
    Ask LLM to extract parts of the document that explain the classification error.
    """
    prompt = f"""For this task:
{TASK_PROMPT_DICT[task_type]}

Given this document:
{doc}

The correct answer is {ground_truth}, but the model incorrectly predicted {prediction}.

Extract ONLY exact chunks from the document that explain why this is a misclassification. Format your response as:

CHUNKS:
"<exact chunk 1>"
"<exact chunk 2>"
...

EXPLANATION:
Brief explanation of why these chunks show that {ground_truth} is the correct classification, not {prediction}.
"""
    
    try:
        res = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            num_retries=2,
            caching=True
        )
        return res.choices[0].message.content
    except Exception as e:
        print(f"Error extracting relevant parts: {e}")
        return doc

def generate_predicate_prompt(task_type: str, ordering: List[Tuple[str, str, float]], candidates: List[Tuple[str, str, float]], class_examples: List[Dict[Tuple[str, str, float], pd.DataFrame]], feedback: str = "", train_df: pd.DataFrame = None, num_surrogates: int = 5, provide_feedback: bool = True) -> str:
    """
    Generate a prompt asking the LLM to propose predicates based on failure analysis.
    Shows examples of misclassifications for each class and candidate.
    
    Args:
        task_type: The type of task (e.g., 'spam', 'sentiment', etc.)
        ordering: List of candidate tuples in the cascade ordering
        candidates: List of all candidate tuples (surrogate_name, surrogate_model, doc_fraction)
        class_examples: List of dictionaries (one per class), each mapping candidate tuples to DataFrames of misclassified examples
        feedback: Optional feedback from previous iterations
        train_df: Training DataFrame, used for sampling examples when ordering is empty
        
    Returns:
        A prompt string for the LLM to generate predicates
    """
    
    # Get classes for this task and create class-specific instructions
    classes = PROMPT_TO_CLASSES_DICT[task_type]
    
    # Determine output format and examples based on the classes
    if len(classes) == 2 and set(classes) == {0, 1}:
        # Binary task - use True/False
        output_format = "True or False"
        example_outputs = ["True", "False"]
        class_description = "binary classification"
        specialization_note = ""
    else:
        # Multi-class task - use actual class numbers
        output_format = " or ".join([str(c) for c in classes])
        example_outputs = [str(classes[0]), str(classes[1]) if len(classes) > 1 else str(classes[0])]
        class_description = f"multi-class classification with classes {output_format}"
        specialization_note = f"""

IMPORTANT FOR MULTI-CLASS: Your surrogate tasks should focus on the EASIEST cases to identify. You don't need to handle all classes perfectly - focus on the cases you can classify with high confidence and coverage. For example:
- A surrogate might be excellent at catching obvious class {classes[0]} cases
- Another surrogate might reliably identify clear class {classes[1]} examples  
- The goal is high precision AND high coverage for the easy subset of each class"""
    
    # Get the original task prompt
    original_prompt = TASK_PROMPT_DICT[task_type]
    
    # Include any feedback if available
    feedback_section = ""
    if feedback:
        feedback_section = f"\n## Feedback from Previous Iterations\n{feedback}\n"
    
    # Format the misclassification examples by class - but focus on broader patterns
    examples_section = "\n## Analysis of Current Model Failures\n"
    
    if provide_feedback:
        if not ordering and train_df is not None:
            # If no cascade yet, show some sample data to help understand the task
            examples_section += "\nTo help you understand this task, here are some sample examples:\n"
            for class_label in classes[:2]:  # Show max 2 classes to avoid overwhelming
                class_sample = train_df[train_df["label"] == class_label].sample(min(3, len(train_df[train_df["label"] == class_label])))
                if not class_sample.empty:
                    examples_section += f"\n**Examples of CLASS {class_label}:**\n"
                    for i, (_, row) in enumerate(class_sample.iterrows()):
                        text = row["filtered_text"]
                        if len(text) > 800:
                            text = text[:800] + "..."
                        examples_section += f"{i+1}. {text}\n"
        
        # Process misclassification examples but focus on patterns rather than specifics
        if class_examples and any(class_dict for class_dict in class_examples):
            examples_section += f"\n**CURRENT FAILURE PATTERNS:**\n"
            examples_section += f"The current models are struggling with certain types of examples. Rather than targeting these specific failures, think about what EASIER subsets of the main task could catch many of these cases reliably.\n"
            
            for class_idx, true_class_label in enumerate(classes):
                if class_idx < len(class_examples):
                    class_dict = class_examples[class_idx]
                    
                    if class_dict:  # Only add section if we have examples
                        examples_section += f"\n**Examples that should be CLASS {true_class_label} but aren't being caught:**\n"
                        
                        for candidate, examples_df in class_dict.items():
                            if len(examples_df) > 0:
                                # Show all examples for this candidate and class
                                # Collect all documents for parallel processing
                                extraction_tasks = []
                                for idx, (_, row) in enumerate(examples_df.iterrows()):
                                    doc = row["filtered_text"]
                                    prediction = row.get("surrogate_prediction", "unknown")
                                    ground_truth = row["label"]
                                    extraction_tasks.append((doc, str(prediction), str(ground_truth), task_type))
                                
                                # Process all extractions in parallel
                                def extract_task(args):
                                    doc, prediction, ground_truth, task_type = args
                                    return extract_relevant_parts(doc, prediction, ground_truth, task_type)
                                
                                with ThreadPoolExecutor(max_workers=2) as executor:
                                    extraction_results = list(tqdm(executor.map(extract_task, extraction_tasks), 
                                                                  total=len(extraction_tasks), 
                                                                  desc="Extracting relevant parts"))
                                
                                # Process the results
                                for idx, (relevant_parts, (_, row)) in enumerate(zip(extraction_results, examples_df.iterrows())):
                                    # Parse the response to get just the chunks if available
                                    if "CHUNKS:" in relevant_parts and "EXPLANATION:" in relevant_parts:
                                        chunks_section = relevant_parts.split("CHUNKS:")[1].split("EXPLANATION:")[0].strip()
                                        explanation_section = relevant_parts.split("EXPLANATION:")[1].strip()
                                        examples_section += f"- Key parts: {chunks_section}\n"
                                        if explanation_section:
                                            examples_section += f"  Why: {explanation_section}\n"
                                    else:
                                        # Fallback to truncated original text if extraction fails
                                        doc = row["filtered_text"]
                                        text = doc[:600] + "..." if len(doc) > 600 else doc
                                        examples_section += f"- {text}\n"
            
            examples_section += f"\nInstead of targeting these specific examples, think: What are the OBVIOUS cases of each class that a simple model could reliably identify?\n"
    else:
        # When not providing feedback, just give the task description without examples
        examples_section = f"\n## Task Understanding\n\nYour goal is to create surrogate tasks that are much simpler than the main task but have strong predictive power.\n"

    cascade_explanation = """
## Understanding Surrogate Tasks with Strong Predictive Power

We want to break complex tasks into simple surrogate tasks that are much easier to detect than the original task. 

A good surrogate task has **strong predictive power**, meaning either or both:

1. **When the surrogate task is True, the probability of the original task being False is very low**
   Example: If there's a medication mentioned, it's rarely a non-medical document

2. **When the surrogate task is False, the probability of the original task being True is very low**  
   Example: If there's no clinical terminology, it's rarely about adverse drug events

**CRITICAL: Each surrogate should detect a COMPLETELY DIFFERENT aspect, not rephrase the same detection.**

**Two Ways Surrogates Work Together:**

1. **Conjunction (AND)** - A sequence where ALL must be True
   - Use when multiple simple checks together give high confidence
   - Each False result strongly suggests the final answer is False
   - Example: "Has medication AND has negative outcome AND has causal language"

2. **Disjunction (OR)** - Independent tasks where ANY True is sufficient  
   - Use when any single True strongly indicates the answer
   - Each True result strongly suggests the final answer is True
   - Example: "Has drug name OR has dosage information OR has prescription details"

**Examples of Different Detection Types:**

- **Entity Detection**: "Is there a medication mentioned?" (prerequisite check)
- **Event Detection**: "Is there a negative health outcome described?" (outcome identification)  
- **Relationship Detection**: "Is there causal or temporal language?" (connection verification)
- **Context Detection**: "Is this a clinical setting?" (domain verification)
- **Attribute Detection**: "Are there severity indicators?" (property assessment)

Each should be **extremely simple to detect with a quick scan** and have **strong predictive power**.
"""

    # Format the final prompt
    prompt = f"""{cascade_explanation}

## Your Classification Task

{original_prompt}

This is a {class_description} task where you need to output: {output_format}

{feedback_section}

{examples_section}

## Your Assignment: Create {num_surrogates} Simple Surrogate Tasks

Based on the analysis above, propose {num_surrogates} surrogate tasks that are **much easier to detect** than the original task.

Each surrogate task should:
1. **Be extremely simple to detect with a quick scan**
2. **Return only True/False (for binary tasks) OR the exact class label numbers WITH their corresponding class names in parentheses (for multi-class tasks)** 
3. **Always explicitly state what to output for each outcome (e.g., 'output 0', 'output 1', or 'output True/False')**
4. **Have strong predictive power** based on the patterns above
5. **Detect a COMPLETELY DIFFERENT aspect** (not rephrase the same detection)
6. **Be very short (ideally a single concise sentence, even a shortened version of the original classification instruction)**
7. **For surrogate #1 specifically: it MUST be a very shortened version of the original classification instruction**

**Strong Predictive Power means:**
- When surrogate is True → high confidence about original task result
- When surrogate is False → high confidence about original task result

**{num_surrogates} Different Detection Types (aim for diversity):**

1. **Entity Detection** - "Is [specific thing] present?"
   - Medications, people, companies, locations, objects

2. **Event Detection** - "Did [specific event] happen?" 
   - Actions, outcomes, processes, changes, incidents

3. **Relationship Detection** - "Is there [specific connection]?"
   - Causality, temporal order, comparison, association

4. **Context Detection** - "Is this [specific setting/domain]?"
   - Clinical, business, personal, formal, technical contexts

5. **Attribute Detection** - "Does it have [specific property]?"
   - Severity, urgency, formality, specificity, emotional tone

**ABSOLUTELY FORBIDDEN: {num_surrogates} variations of the same detection type!**

Bad example (all entity detection):
- "Is there a medication mentioned?"
- "Is there a drug name mentioned?" 
- "Is there a pharmaceutical mentioned?"
← All detect the same thing, just rephrased!

{specialization_note}

**CRITICAL: Each surrogate must be a DIFFERENT TYPE of check with HIGH COVERAGE!**

**You must use EXACTLY this format (no markdown, no bullets, no extra formatting):**

PROMPT: <your complete classification instruction>
RATIONALE: <explain the simplified aspect and expected coverage>

**Examples (copy this exact format):**"""

    # Add diverse examples based on task type - showing 5 DIFFERENT types of checks
    if len(classes) == 2 and set(classes) == {0, 1}:
        # Binary task examples with True/False - showing task-specific decomposition
        examples_text = """For a complex task like "Are there adverse side effects to a medication mentioned in this document?":

PROMPT: Is there any medication or drug mentioned in this text? If yes, output True. Otherwise, output False.
RATIONALE: PREREQUISITE CHECK - can't have adverse drug effects without mentioning a drug first, simpler than full task

PROMPT: Is there any negative health outcome or side effect described? If yes, output True. Otherwise, output False.
RATIONALE: SUB-TASK FOR SPECIFIC CLASS - detecting negative outcomes is simpler than connecting them to medications

PROMPT: Is there any causal or temporal language linking events? If yes, output True. Otherwise, output False.
RATIONALE: SUB-TASK FOR SPECIFIC CLASS - causal connections are key to adverse events, easier to detect than full causality

PROMPT: Does this text describe a patient case or clinical scenario? If no, output False. Otherwise, output True.
RATIONALE: ELIMINATION FILTER - non-clinical texts unlikely to contain adverse drug events

PROMPT: Are there any medical severity terms or clinical assessment language? If yes, output True. Otherwise, output False.
RATIONALE: SEMANTIC INDICATOR - severity language indicates clinical significance, part of adverse event detection"""
    else:
        # Multi-class examples with class numbers - showing task-specific decomposition
        examples_text = f"""For a complex task like \"Classify this customer review as Positive (0), Neutral (1), or Negative (2)\":

PROMPT: Classify this customer review as Positive (0), Neutral (1), or Negative (2).
RATIONALE: SHORTENED ORIGINAL - concise restatement of the full task with the same class outputs. Make sure to include the class descriptions AND class numbers.

PROMPT: Does this review use strong negative words such as 'terrible', 'awful', or 'refund'? If yes, output 2. Otherwise, output 0.
RATIONALE: SUB-TASK FOR SPECIFIC CLASS - presence of strong negative cues points to Negative class, simpler than full sentiment analysis

PROMPT: Does this review include clearly positive words like 'love', 'excellent', or 'fantastic'? If yes, output 0. Otherwise, output 2.
RATIONALE: SUB-TASK FOR SPECIFIC CLASS - positive cues strongly indicate Positive class

PROMPT: Does the review mention star ratings of 4 or 5? If yes, output 0. Otherwise, output 1.
RATIONALE: ELIMINATION FILTER - high star ratings suggest Positive sentiment, else ambiguous/neutral

PROMPT: Does the review describe an average or mixed experience with phrases like 'okay', 'average', or 'fine'? If yes, output 1. Otherwise, output 0.
RATIONALE: SEMANTIC INDICATOR - average language often maps to Neutral sentiment"""

    prompt += examples_text + f"""

**Key Requirements for Simple Surrogate Tasks:**
- Each surrogate must detect a DIFFERENT ASPECT from the 5 types listed above
- NO two surrogates should detect the same thing with different words
- Focus on HIGH COVERAGE - should apply to many examples, not just edge cases  
- Use SIMPLE detection that can be done with a quick scan
- Each should have strong predictive power for the original task

**Mandatory Diversity Check:**
Before submitting, verify you have diverse detection types across your {num_surrogates} surrogates:
- ENTITY DETECTION (is specific thing present?)
- EVENT DETECTION (did specific event happen?)  
- RELATIONSHIP DETECTION (is there specific connection?)
- CONTEXT DETECTION (is this specific setting/domain?)
- ATTRIBUTE DETECTION (does it have specific property?)

**Format each as (no bold text or decoration):**
PROMPT: <the prompt text without any placeholders>
RATIONALE: <explain if conjunction/disjunction and how its predictions correlate with the original task>

**Absolutely Avoid These Mistakes:**
- Creating 5 variations of the same detection type (e.g., all entity detection)
- Using the same detection with different synonyms
- Making all surrogates about the same fundamental aspect
- Complex logic that's hard to detect with a quick scan

**Remember:** The goal is {num_surrogates} DIFFERENT detection types with strong predictive power!
"""

    return prompt

def print_predicate_prompt_with_rich(prompt: str) -> None:
    """
    Pretty-print the predicate prompt using rich formatting
    
    Args:
        prompt: The prompt text to format and print
    """
    console = Console()
    
    # Split the prompt into sections
    sections = prompt.split("\n\n")
    
    # Print the prompt with rich formatting
    for section in sections:
        if section.startswith("## "):
            # Format as a header
            header_text = section.strip("## ")
            console.print(f"[bold cyan]{section}[/bold cyan]")
        elif section.startswith("1. ") or section.startswith("2. ") or section.startswith("3. ") or section.startswith("4. "):
            # Format as a list
            console.print(f"[yellow]{section}[/yellow]")
        elif "Example" in section and "PROMPT:" in section:
            # Format as an example
            console.print(f"[green]{section}[/green]")
        elif section.startswith("### Candidate:"):
            # Format as a candidate section
            console.print(f"[bold magenta]{section}[/bold magenta]")
        else:
            # Format as regular text
            try:
                console.print(section.replace("[/h1]", "").replace("[/b]", ""))
            except Exception as e:
                print(section)
        
        console.print("")  # Add a blank line between sections

def get_predicate_proposals(prompt: str, prompt_prefix: str, prompt_suffix: str, messages_history: List[Dict[str, str]] = None) -> List[str]:
    """
    Get predicate proposals from the LLM using a message history approach.
    Retries up to 5 times if no prompts are extracted.
    
    Args:
        prompt: Initial prompt or follow-up message if messages_history is provided
        prompt_prefix: Prefix to be added to each extracted predicate
        prompt_suffix: Suffix to be added to each extracted predicate
        messages_history: Optional list of previous messages in the conversation
        
    Returns:
        List of predicate prompts with prefix and suffix
    """
    console = Console()
    
    # Initialize or use existing message history
    if messages_history is None:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = messages_history.copy()
        messages.append({"role": "user", "content": prompt})
    
    for attempt in range(5):
        try:
            console.print(f"[bold cyan]Attempt {attempt + 1}:[/bold cyan] Getting predicate proposals...")
            res = completion(
                model="o1-mini", 
                # model="o3",
                # model="gpt-4o",
                messages=messages,
                num_retries=2,
                caching=True if attempt == 0 else False
            )
            
            response = res.choices[0].message.content
            
            # Add the assistant's response to the message history for future iterations
            if messages_history is not None:
                messages_history.append({"role": "assistant", "content": response})
            
            # Parse response to extract prompts and rationales - expect exact format
            prompts = []
            current_prompt = None
            current_rationale = None
            
            for line in response.split('\n'):
                line = line.strip()
                
                # Look for exact PROMPT: format
                if line.startswith('PROMPT:'):
                    # If we already have a complete prompt-rationale pair, save it
                    if current_prompt and current_rationale:
                        console.print(f"[bold green]PROMPT:[/bold green] {current_prompt}")
                        console.print(f"[bold blue]RATIONALE:[/bold blue] {current_rationale}\n")
                        
                        # Create the full prompt with prefix and suffix
                        new_prompt = prompt_prefix + "\n\n" + current_prompt + "\n\n" + prompt_suffix
                        prompts.append(new_prompt)
                    
                    # Extract new prompt
                    current_prompt = line.replace('PROMPT:', '').strip()
                    current_rationale = None
                
                # Look for exact RATIONALE: format
                elif line.startswith('RATIONALE:') and current_prompt:
                    current_rationale = line.replace('RATIONALE:', '').strip()
            
            # Handle the last prompt-rationale pair
            if current_prompt and current_rationale:
                console.print(f"[bold green]PROMPT:[/bold green] {current_prompt}")
                console.print(f"[bold blue]RATIONALE:[/bold blue] {current_rationale}\n")
                
                # Create the full prompt with prefix and suffix
                new_prompt = prompt_prefix + "\n\n" + current_prompt + "\n\n" + prompt_suffix
                prompts.append(new_prompt)
            
            if prompts:
                console.print(f"[bold green]Successfully extracted {len(prompts)} prompts[/bold green]")
                return prompts
            
            console.print(f"[yellow]Attempt {attempt + 1}: No prompts extracted from response, retrying...[/yellow]")
            console.print(f"[dim]Response preview: {response[:1000]}...[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error on attempt {attempt + 1}: {e}[/red]")
            if attempt == 4:
                return []
    return []

def find_surrogates(
    train_df: pd.DataFrame,
    task: str,
    target_accuracy: float,
    num_iterations: int = 2,
    random_seed: int = 42,
    include_selectivity: bool = True,
    provide_feedback: bool = True,
    num_surrogate_requests: int = 5,
    guarantee_accuracy: bool = False,
) -> Dict:
    """
    Train and save predicates using an iterative agent-based approach with message history.
    
    Args:
        train_df: DataFrame containing training documents
        task: The classification task
        target_accuracy: Target accuracy for the cascade
        num_iterations: Number of iterations to run (default: 3)
        random_seed: Random seed for reproducibility (default: 42)
        include_selectivity: Whether to include selectivity cascade method (default: True)
        provide_feedback: Whether to provide feedback in iterations (default: True)
        num_surrogate_requests: Number of surrogates to request from agent (default: 5)
        guarantee_accuracy: Whether to guarantee accuracy in the cascade (default: False)
    
    Returns:
        Dictionary containing the best cascade results across all iterations
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Use the full training dataset
    train_data_df = train_df.copy().reset_index(drop=True)
    
    console.print(f"[bold]Using {len(train_data_df)} training examples[/bold]")
    
    prompt_prefix, prompt_suffix = PROMPT_PREFIX_SUFFIX_DICT[task]
    task_prompt = TASK_PROMPT_DICT[task]
    task_type_binary_or_multiclass = PROMPT_TO_TASK_TYPE_DICT[task]
    
    all_executions = []
    all_candidates = []
    surrogate_counter = 2  # Start with s2 (s1 is baseline)
    all_iteration_predicates = []
    
    # Initialize message history for the agent conversation
    message_history = []
    
    # Helper function to run a single predictor task - makes parallel execution easier
    def run_predictor_task(predictor, prompt, df, surrogate_name):
        return run_predictor_and_get_row_copies(predictor, prompt, df, surrogate_name, task_type=PROMPT_TO_TASK_TYPE_DICT[task])
    
    # Run initial baseline and oracle predictors in parallel
    prediction_tasks = [
        (BASELINE_PREDICTOR, task_prompt, train_data_df, "s1"),  # Baseline
        (ORACLE_PREDICTOR, task_prompt, train_data_df, "s1"),    # Oracle
    ]
    
    console.print(f"[bold cyan]Running initial predictors in parallel...[/bold cyan]")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda args: run_predictor_task(*args),
            prediction_tasks
        ))
    
    # Flatten results and add to all_executions
    for result in results:
        all_executions.extend(result)
    
    # Initialize all_executions_df with baseline and oracle results
    all_executions_df = pd.DataFrame(all_executions)
    
    # Create initial candidates from s1
    for doc_fraction in CANDIDATE_FRACTIONS:
        for predictor in PREDICTORS:
            if predictor == ORACLE_PREDICTOR and doc_fraction == 1.0:
                # No need to add oracle on full doc as a candidate
                continue
            all_candidates.append(("s1", predictor, doc_fraction))
    
    # Get initial cascade results as baseline
    baseline_cascade_greedy = design_cascade_optimal_greedy(
        all_executions_df,
        all_candidates,
        target_accuracy,
        task,
        # guarantee_accuracy=guarantee_accuracy,
    )
    best_cascade_greedy = baseline_cascade_greedy.copy()
    
    console.print(f"[bold]Initial greedy cascade cost:[/bold] {baseline_cascade_greedy['total_cost']:.4f}, [bold]accuracy:[/bold] {baseline_cascade_greedy['accuracy']:.4f}")
    
    if include_selectivity:
        baseline_cascade_selectivity = design_cascade_optimal_selectivity(all_executions_df, all_candidates, target_accuracy, task)
        best_cascade_selectivity = baseline_cascade_selectivity.copy()
        console.print(f"[bold]Initial selectivity cascade cost:[/bold] {baseline_cascade_selectivity['total_cost']:.4f}, [bold]accuracy:[/bold] {baseline_cascade_selectivity['accuracy']:.4f}")
    else:
        best_cascade_selectivity = None
    
    # Iteration loop
    for iteration in range(num_iterations):
        console.print(f"\n[bold yellow]{'='*20} ITERATION {iteration+1}/{num_iterations} {'='*20}[/bold yellow]")
        
        # Get false positives and negatives from current cascade
        class_examples = find_false_positives_and_negatives(
            all_executions_df, 
            best_cascade_greedy["ordering"], 
            best_cascade_greedy["thresholds"], 
            task,
            5
        )
        
        # For the first iteration, generate an initial prompt
        if iteration == 0:
            # If not providing feedback, use simpler examples and request more surrogates
            if not provide_feedback:
                initial_prompt = generate_predicate_prompt(
                    task_type=task,
                    ordering=[],  # Empty ordering for no feedback mode
                    candidates=[],
                    class_examples=[],
                    train_df=train_data_df,
                    num_surrogates=num_surrogate_requests,
                    provide_feedback=False
                )
            else:
                initial_prompt = generate_predicate_prompt(
                    task_type=task,
                    ordering=best_cascade_greedy["ordering"],
                    candidates=all_candidates,
                    class_examples=class_examples,
                    train_df=train_data_df,
                    num_surrogates=num_surrogate_requests,
                    provide_feedback=True
                )
            
            # Print the prompt with rich formatting
            console.print("[bold green]Initial Predicate Prompt:[/bold green]")
            print_predicate_prompt_with_rich(initial_prompt)
            
            # Start the message history with this initial prompt
            message_history = [{"role": "user", "content": initial_prompt}]
            
            # Get new predicate proposals for the first iteration
            console.print(f"[bold cyan]Getting initial predicate proposals...[/bold cyan]")
            predicate_proposals = get_predicate_proposals(initial_prompt, prompt_prefix, prompt_suffix, message_history)
            
            # If not providing feedback, end after first iteration
            if not provide_feedback:
                console.print(f"[bold yellow]Single iteration mode - ending after first iteration[/bold yellow]")
        else:
            # For subsequent iterations, create a follow-up prompt with feedback
            feedback = generate_feedback_for_next_iteration(
                all_iteration_predicates,
                best_cascade_greedy["ordering"],
                iteration,
                task,
                all_executions_df,
                best_cascade_greedy
            )
            
            # Rich print the feedback for better readability
            from rich.panel import Panel
            from rich.text import Text
            
            feedback_text = Text(feedback)
            console.print(Panel(feedback_text, title="Surrogate Performance Feedback", border_style="cyan"))
            
            # Determine the rationale description based on the task classes
            classes = PROMPT_TO_CLASSES_DICT[task]
            if len(classes) == 2 and set(classes) == {0, 1}:
                rationale_description = "how its True/False predictions correlate with the original task"
            else:
                output_format = " or ".join([str(c) for c in classes])
                rationale_description = f"how its {output_format} predictions correlate with the original task"
            
            follow_up_prompt = f"""I've analyzed the performance of your previous surrogate task suggestions. Here's my feedback:

{feedback}

Based on the above feedback and the current cascade performance, could you propose 5 NEW surrogate tasks that might further reduce the cost? 

Please focus on approaches that are different from those that didn't work well. If some of your previous approaches were effective, consider refining those or creating more specific variants.

Half should be keyword-based or pattern-based; the other half should be more logical or semantic.

Keep each PROMPT very short (ideally a single concise sentence, even a shortened version of the original classification instruction).
The FIRST surrogate you return should be exactly such a shortened version of the original task prompt, and it must explicitly specify the output mapping—include each class name with its number (e.g., Positive (0), Neutral (1), Negative (2)) or True/False for binary.

Remember to format each as:
PROMPT: <the prompt text without any placeholders>
RATIONALE: <explain if conjunction/disjunction and {rationale_description}>

No bolds or italics around the PROMPT or RATIONALE, since we will be parsing with regex."""
            
            # Get new predicate proposals using the existing message history
            console.print(f"[bold cyan]Getting follow-up predicate proposals for iteration {iteration+1}...[/bold cyan]")
            predicate_proposals = get_predicate_proposals(follow_up_prompt, prompt_prefix, prompt_suffix, message_history)
        
        if not predicate_proposals:
            console.print(f"[bold red]No new predicates proposed in iteration {iteration+1}, ending early.[/bold red]")
            break
            
        console.print(f"[bold green]Got {len(predicate_proposals)} new predicate proposals[/bold green]")
        
        # Store predicates for this iteration
        iteration_predicates = []
        
        # Create a list to hold all new prediction tasks
        all_new_prediction_tasks = []
        
        # Prepare prediction tasks for all the new predicates
        for i, predicate in enumerate(predicate_proposals):
            surrogate_name = f"s{surrogate_counter}"
            surrogate_counter += 1
            
            console.print(f"[cyan]Preparing predicate {i+1}/{len(predicate_proposals)}: {surrogate_name}[/cyan]")
            iteration_predicates.append((surrogate_name, predicate))

            # Subset train_data_df to only include the min doc fraction
            train_data_df_subset_min_doc_fraction = train_data_df[train_data_df["fraction"] == min(CANDIDATE_FRACTIONS)]
            
            # Add all prediction tasks for this predicate
            all_new_prediction_tasks.extend([
                (BASELINE_PREDICTOR, predicate, train_data_df, surrogate_name),  # Baseline
                # (ORACLE_PREDICTOR, predicate, train_data_df_subset_min_doc_fraction, surrogate_name),    # Oracle
            ])
            
            # Add new candidates
            for doc_fraction in CANDIDATE_FRACTIONS:
                all_candidates.append((surrogate_name, BASELINE_PREDICTOR, doc_fraction))
                if doc_fraction == 1.0:
                    # No need to add oracle on full doc as a candidate
                    continue
                # if doc_fraction == min(CANDIDATE_FRACTIONS):
                #     all_candidates.append((surrogate_name, ORACLE_PREDICTOR, doc_fraction))
        
        # Execute all prediction tasks in parallel
        console.print(f"[bold cyan]Running {len(all_new_prediction_tasks)} prediction tasks in parallel...[/bold cyan]")
        results = []
        for args in all_new_prediction_tasks:
            results.append(run_predictor_task(*args))
            
        # Flatten results and add to all_executions
        for result in results:
            all_executions.extend(result)
        
        # Store all predicates across iterations
        all_iteration_predicates.extend(iteration_predicates)
        
        # Update executions dataframe
        all_executions_df = pd.DataFrame(all_executions)
        
        # Compute new cascade results
        old_cost_greedy = best_cascade_greedy["total_cost"]
        old_accuracy_greedy = best_cascade_greedy["accuracy"]
        
        new_cascade_greedy = design_cascade_optimal_greedy(
            all_executions_df,
            all_candidates,
            target_accuracy,
            task,
            # guarantee_accuracy=guarantee_accuracy,
        )
        
        new_cost_greedy = new_cascade_greedy["total_cost"]
        new_accuracy_greedy = new_cascade_greedy["accuracy"]
        
        # Compute improvements for greedy method
        cost_improvement_greedy = old_cost_greedy - new_cost_greedy
        accuracy_improvement_greedy = new_accuracy_greedy - old_accuracy_greedy
        
        console.print(f"[bold magenta]Iteration {iteration+1} results:[/bold magenta]")
        console.print(f"  [green]Greedy method:[/green] Cost improvement: {cost_improvement_greedy:.4f} ({old_cost_greedy:.4f} -> {new_cost_greedy:.4f})")
        console.print(f"  [green]Greedy method:[/green] Accuracy: {old_accuracy_greedy:.4f} -> {new_accuracy_greedy:.4f}")
        
        if include_selectivity:
            old_cost_selectivity = best_cascade_selectivity["total_cost"]
            old_accuracy_selectivity = best_cascade_selectivity["accuracy"]
            
            new_cascade_selectivity = design_cascade_optimal_selectivity(all_executions_df, all_candidates, target_accuracy, task)
            
            new_cost_selectivity = new_cascade_selectivity["total_cost"]
            new_accuracy_selectivity = new_cascade_selectivity["accuracy"]
            
            cost_improvement_selectivity = old_cost_selectivity - new_cost_selectivity
            accuracy_improvement_selectivity = new_accuracy_selectivity - old_accuracy_selectivity
            
            console.print(f"  [blue]Selectivity method:[/blue] Cost improvement: {cost_improvement_selectivity:.4f} ({old_cost_selectivity:.4f} -> {new_cost_selectivity:.4f})")
            console.print(f"  [blue]Selectivity method:[/blue] Accuracy: {old_accuracy_selectivity:.4f} -> {new_accuracy_selectivity:.4f}")
        
        # Keep track of the best cascade for each method
        if new_cost_greedy < best_cascade_greedy["total_cost"]:
            best_cascade_greedy = new_cascade_greedy
            console.print(f"  [bold green]New best greedy cascade found![/bold green]")
        else:
            console.print(f"  [yellow]No improvement in greedy cascade cost.[/yellow]")
            
        if include_selectivity:
            if new_cost_selectivity < best_cascade_selectivity["total_cost"]:
                best_cascade_selectivity = new_cascade_selectivity
                console.print(f"  [bold blue]New best selectivity cascade found![/bold blue]")
            else:
                console.print(f"  [yellow]No improvement in selectivity cascade cost.[/yellow]")
        
        # If not providing feedback, end after first iteration
        if not provide_feedback:
            break
    
    # Store all predicates used across iterations
    all_predicates = []
    for surrogate_name, predicate_text in all_iteration_predicates:
        all_predicates.append(predicate_text)
    
    # Create mapping from surrogate name to prompt
    surrogate_to_prompt = {"s1": task_prompt}
    for (surrogate_name, predicate_text) in all_iteration_predicates:
        surrogate_to_prompt[surrogate_name] = predicate_text
    
    # Add common fields to the best greedy cascade
    best_cascade_greedy["all_predicates"] = all_predicates
    best_cascade_greedy["message_history"] = message_history
    best_cascade_greedy["surrogate_to_prompt"] = surrogate_to_prompt
    best_cascade_greedy["method"] = "greedy"

    # Build the result dictionary
    result = {
        "greedy": best_cascade_greedy,
        "surrogate_to_prompt": surrogate_to_prompt,
        "all_predicates": all_predicates,
        "message_history": message_history,
    }

    # ===== CONDITIONAL: Compute Greedy Cascade with Accuracy Guarantee =====
    if guarantee_accuracy:
        best_cascade_greedy_guaranteed = design_cascade_optimal_greedy(
            all_executions_df,
            all_candidates,
            target_accuracy,
            task,
            guarantee_accuracy=True,
        )

        # Attach shared metadata to guaranteed cascade
        best_cascade_greedy_guaranteed["all_predicates"] = all_predicates
        best_cascade_greedy_guaranteed["message_history"] = message_history
        best_cascade_greedy_guaranteed["surrogate_to_prompt"] = surrogate_to_prompt
        best_cascade_greedy_guaranteed["method"] = "greedy_guaranteed"
        
        result["greedy_guaranteed"] = best_cascade_greedy_guaranteed

    if include_selectivity:
        best_cascade_selectivity["all_predicates"] = all_predicates
        best_cascade_selectivity["message_history"] = message_history
        best_cascade_selectivity["surrogate_to_prompt"] = surrogate_to_prompt
        best_cascade_selectivity["method"] = "selectivity"
        result["selectivity"] = best_cascade_selectivity

    return result

def generate_feedback_for_next_iteration(
    all_iteration_predicates: List[Tuple[str, str]],
    final_ordering: List[Tuple[str, str, float]],
    current_iteration: int,
    task: str,
    all_executions_df: pd.DataFrame = None,
    best_cascade: Dict = None
) -> str:
    """
    Generate detailed feedback on surrogate performance for the next iteration.
    
    Args:
        all_iteration_predicates: List of all (surrogate_name, predicate_text) tuples across iterations
        final_ordering: The cascade ordering after the current iteration
        current_iteration: The current iteration number
        all_executions_df: DataFrame containing all execution results
        best_cascade: Current best cascade configuration
        
    Returns:
        Detailed feedback text summarizing surrogate performance with examples
    """
    # Extract surrogate names that were included in the cascade
    selected_surrogates = set(name for name, _, _ in final_ordering if name.startswith("s") and name != "s1")
    
    # Group candidates by iteration based on surrogate name (s2, s3... from first iteration, etc.)
    iteration_groups = {}
    
    for surrogate_name, predicate_text in all_iteration_predicates:
        # Extract iteration number from surrogate name
        num = int(surrogate_name[1:])  # Remove 's' prefix and convert to int
        iteration_index = (num - 2) // 5  # Assuming 5 predicates per iteration, 0-indexed
        
        if iteration_index not in iteration_groups:
            iteration_groups[iteration_index] = []
        
        iteration_groups[iteration_index].append((surrogate_name, predicate_text))
    
    # Get predicates from current iteration only
    if current_iteration - 1 in iteration_groups:
        current_predicates = iteration_groups[current_iteration - 1]
    else:
        current_predicates = []
    
    # Find which predicates were used and which weren't
    successful_predicates = []
    unsuccessful_predicates = []
    
    for surrogate_name, predicate_text in current_predicates:
        if surrogate_name in selected_surrogates:
            successful_predicates.append((surrogate_name, predicate_text))
        else:
            unsuccessful_predicates.append((surrogate_name, predicate_text))
    
    # Generate feedback
    feedback = f"ITERATION {current_iteration} SURROGATE PERFORMANCE EVALUATION:\n"
    
    # Add system cost improvement information
    if best_cascade and "total_cost" in best_cascade:
        feedback += f"\nCASCADE SYSTEM PERFORMANCE:\n"
        feedback += f"- Current Total System Cost: {best_cascade['total_cost']:.4f}\n"
        feedback += f"- Current Accuracy: {best_cascade['accuracy']:.4f}\n"
    
    if successful_predicates:
        feedback += "\nSURROGATE TASKS THAT IMPROVED THE CASCADE:\n"
        for name, text in successful_predicates:
            # Extract just the main prompt text 
            main_prompt = text
            for prefix_suffix_pair in PROMPT_PREFIX_SUFFIX_DICT.values():
                main_prompt = main_prompt.replace(prefix_suffix_pair[0], "").replace(prefix_suffix_pair[1], "").strip()
            
            feedback += f"✓ {name}: {main_prompt}\n"
            
            # Show some performance metrics and misclassification examples
            if all_executions_df is not None:
                surrogate_data = all_executions_df[all_executions_df["surrogate_name"] == name]
                if not surrogate_data.empty:
                    accuracy = (surrogate_data["surrogate_prediction"] == surrogate_data["label"]).mean()
                    feedback += f"  → Accuracy: {accuracy:.3f}, Used in cascade\n"
                    
                    # Show examples of misclassifications this surrogate made
                    misclassified = surrogate_data[surrogate_data["surrogate_prediction"] != surrogate_data["label"]]
                    if not misclassified.empty:
                        feedback += f"  → Examples this surrogate still missed:\n"
                        classes = PROMPT_TO_CLASSES_DICT[task]
                        
                        # Collect all extraction tasks for parallel processing
                        all_extraction_tasks = []
                        class_row_mapping = {}
                        
                        for class_label in classes:
                            class_errors = misclassified[misclassified["label"] == class_label].sort_values(by="surrogate_confidence", ascending=False).head(15)
                            if not class_errors.empty:
                                class_row_mapping[class_label] = []
                                for i, (_, row) in enumerate(class_errors.iterrows()):
                                    doc = row["filtered_text"]
                                    predicted = row["surrogate_prediction"]
                                    all_extraction_tasks.append((doc, str(predicted), str(class_label), task))
                                    class_row_mapping[class_label].append((i, row))
                        
                        # Process all extractions in parallel
                        if all_extraction_tasks:
                            def extract_task(args):
                                doc, prediction, ground_truth, task_type = args
                                return extract_relevant_parts(doc, prediction, ground_truth, task_type)
                            
                            with ThreadPoolExecutor(max_workers=32) as executor:
                                extraction_results = list(tqdm(executor.map(extract_task, all_extraction_tasks), 
                                                              total=len(all_extraction_tasks), 
                                                              desc="Extracting relevant parts (successful)"))
                            
                            # Process results by class
                            result_index = 0
                            for class_label in classes:
                                if class_label in class_row_mapping:
                                    feedback += f"    Class {class_label} errors:\n"
                                    for i, row in class_row_mapping[class_label]:
                                        predicted = row["surrogate_prediction"]
                                        relevant_parts = extraction_results[result_index]
                                        result_index += 1
                                        
                                        # Parse the response to get focused feedback
                                        if "CHUNKS:" in relevant_parts and "EXPLANATION:" in relevant_parts:
                                            chunks_section = relevant_parts.split("CHUNKS:")[1].split("EXPLANATION:")[0].strip()
                                            explanation_section = relevant_parts.split("EXPLANATION:")[1].strip()
                                            feedback += f"    {i+1}. Predicted {predicted}, should be {class_label}:\n"
                                            feedback += f"        Key parts: {chunks_section}...\n"
                                            if explanation_section:
                                                feedback += f"        Why: {explanation_section}...\n"
                                        else:
                                            # Fallback to truncated original text if extraction fails
                                            doc = row["filtered_text"]
                                            text_snippet = doc[:500] + "..." if len(doc) > 500 else doc
                                            feedback += f"    {i+1}. Predicted {predicted}, should be {class_label}: {text_snippet}\n"
            
        feedback += "\n"
    
    if unsuccessful_predicates:
        feedback += "\nSURROGATE TASKS THAT WERE NOT EFFECTIVE:\n"
        for name, text in unsuccessful_predicates:
            # Extract just the main prompt text
            main_prompt = text
            for prefix_suffix_pair in PROMPT_PREFIX_SUFFIX_DICT.values():
                main_prompt = main_prompt.replace(prefix_suffix_pair[0], "").replace(prefix_suffix_pair[1], "").strip()
            
            feedback += f"✗ {name}: {main_prompt}\n"
            
            # Show why it might not have worked
            # if all_executions_df is not None:
            #     surrogate_data = all_executions_df[all_executions_df["surrogate_name"] == name]
            #     if not surrogate_data.empty:
            #         accuracy = (surrogate_data["surrogate_prediction"] == surrogate_data["label"]).mean()
            #         coverage = len(surrogate_data) / len(all_executions_df.drop_duplicates("uuid"))
            #         feedback += f"  → Accuracy: {accuracy:.3f}, Coverage: {coverage:.3f}, Not cost-effective\n"
                    
            #         # Show examples of misclassifications to understand what went wrong
            #         misclassified = surrogate_data[surrogate_data["surrogate_prediction"] != surrogate_data["label"]]
            #         if not misclassified.empty:
            #             feedback += f"  → Examples of errors (why it wasn't selected):\n"
            #             classes = PROMPT_TO_CLASSES_DICT[task]
                        
            #             # Collect all extraction tasks for parallel processing
            #             all_extraction_tasks = []
            #             class_row_mapping = {}
                        
            #             for class_label in classes:
            #                 class_errors = misclassified[misclassified["label"] == class_label]
            #                 if not class_errors.empty:
            #                     class_row_mapping[class_label] = []
            #                     for i, (_, row) in enumerate(class_errors.iterrows()):
            #                         doc = row["filtered_text"]
            #                         predicted = row["surrogate_prediction"]
            #                         all_extraction_tasks.append((doc, str(predicted), str(class_label), task))
            #                         class_row_mapping[class_label].append((i, row))
                        
            #             # Process all extractions in parallel
            #             if all_extraction_tasks:
            #                 def extract_task(args):
            #                     doc, prediction, ground_truth, task_type = args
            #                     return extract_relevant_parts(doc, prediction, ground_truth, task_type)
                            
            #                 with ThreadPoolExecutor(max_workers=24) as executor:
            #                     extraction_results = list(tqdm(executor.map(extract_task, all_extraction_tasks), 
            #                                                   total=len(all_extraction_tasks), 
            #                                                   desc="Extracting relevant parts (unsuccessful docs in cascade)"))
                            
            #                 # Process results by class
            #                 result_index = 0
            #                 for class_label in classes:
            #                     if class_label in class_row_mapping:
            #                         feedback += f"    Class {class_label} errors:\n"
            #                         for i, row in class_row_mapping[class_label]:
            #                             predicted = row["surrogate_prediction"]
            #                             relevant_parts = extraction_results[result_index]
            #                             result_index += 1
                                        
            #                             # Parse the response to get focused feedback
            #                             if "CHUNKS:" in relevant_parts and "EXPLANATION:" in relevant_parts:
            #                                 chunks_section = relevant_parts.split("CHUNKS:")[1].split("EXPLANATION:")[0].strip()
            #                                 explanation_section = relevant_parts.split("EXPLANATION:")[1].strip()
            #                                 feedback += f"    {i+1}. Predicted {predicted}, should be {class_label}:\n"
            #                                 feedback += f"        Key parts: {chunks_section[:500]}...\n"
            #                                 if explanation_section:
            #                                     feedback += f"        Why: {explanation_section[:200]}...\n"
            #                             else:
            #                                 # Fallback to truncated original text if extraction fails
            #                                 doc = row["filtered_text"]
            #                                 text_snippet = doc[:500] + "..." if len(doc) > 500 else doc
            #                                 feedback += f"    {i+1}. Predicted {predicted}, should be {class_label}: {text_snippet}\n"
            #         else:
            #             feedback += f"  → High accuracy but likely too low coverage or high cost\n"
            
        feedback += "\n"
    
    feedback += "RECOMMENDATIONS FOR YOUR NEXT SURROGATE TASKS:\n"
    
    if successful_predicates:
        feedback += "1. BUILD ON SUCCESS: The successful predicates above show effective patterns\n"
        feedback += "2. CREATE VARIATIONS: Try similar approaches with different keywords or logic\n"
    else:
        feedback += "1. TRY SIMPLER APPROACHES: Previous predicates may have been too complex\n"
        feedback += "2. FOCUS ON OBVIOUS PATTERNS: Look for clear, simple indicators\n"
    
    feedback += "3. TARGET SPECIFIC CLASSES: Specialize in detecting one class very reliably\n"
    feedback += "4. USE KEYWORD-BASED DETECTION: Simple word/phrase matching often works well\n"
    feedback += "5. AVOID OVERLY COMPLEX LOGIC: Small models work best with simple rules\n"
    
    return feedback