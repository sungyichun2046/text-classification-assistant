import asyncio
import logging
import os
from datetime import datetime
from typing import List

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score

from app import chunker
from config import OPEN_API_KEY

MAX_REQUESTS_PER_TIME = 100
MODEL = "gpt-4o-mini"

DATASET = "spam"


class Feedback(BaseModel):
    """ Feedback object for one data returned by LLM."""
    index: int  # text index in result_df
    text: str  # text classified
    category: str  # category decided on UI
    explanation: str  # explanation of choice of category
    rating: int  # rating given by LLM as judge
    rating_reason:  str  # reason of rating
    suggested_instruction: str  # suggested in struction to add to improve categorizatin result


class Feedbacks(BaseModel):
    """ List of Feedback objects."""
    feedbacks: List[Feedback]


def main():
    """ Compute metrics containing F-score, accuracy and llm_judge_score.

    If these metrics are correlated, we can trust reliability of llm_judge_score"""
    # Load dataset containing annotated category
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )

    annotated_col = "Category"
    annotated_df = pd.read_csv(f"data/{DATASET}/annotated.csv", encoding='utf-8')
    annotated_df = annotated_df.reset_index()
    # Load csv containing openai generated category
    result_df = pd.read_csv(f"data/{DATASET}/results.csv")
    result_df = result_df.reset_index()
    start = datetime.now()

    # Add annotated category to result, they use same indexes
    result_df = result_df.merge(annotated_df[['index', annotated_col]], on='index').drop('index', axis=1)
    result_df = result_df.reset_index()
    result_df = result_df.rename(columns={annotated_col: 'expected'})
    # Compute F-score, accuracy
    accuracy = accuracy_score(result_df["expected"], result_df["category"])
    f_score = f1_score(result_df["expected"], result_df["category"], average="weighted")

    # Compute llm_judge_score
    result_df = ask_llm(result_df)
    logging.info(
        "LLM requesting execution time = %s", datetime.now() - start)

    average_llm_judge_score = result_df["rating"].mean()
    # Convert rating between 1 and 4 to percentage between 0 and 1, to be able to compared with accuracy and f1-score
    rescaled_llm_judge_score = rescale_number(
        value=average_llm_judge_score, original_min=1, original_max=4, new_min=0, new_max=1
    )
    # Convert in percentages
    logging.info(
        "Accuracy = %d, F1 score = %d, LLM judge score = %d",
        float(accuracy)*100, float(f_score)*100, float(rescaled_llm_judge_score)*100
    )

    # Store metrics and llm results
    cols = ["text", "category", "explanation", "rating", "rating_reason", "suggested_instruction"]
    result_df[cols].to_csv(f'data/{DATASET}/evaluation.csv', index=False)
    store_evaluation(
        dataset=DATASET, accuracy=accuracy*100, f_score=f_score*100,
        average_llm_judge_score=rescaled_llm_judge_score*100,
    )


def store_evaluation(
    dataset: pd.DataFrame(), accuracy: float, f_score: float, average_llm_judge_score: float
):
    """ Store date, dataset_name, accuracy, F1-score, average_llm_judge_score of each run.

    Instructions stored in evaluation.csv will be added in last prompt to improve result."""
    df = pd.DataFrame(
        [
            {
                "date": datetime.today().strftime("%Y-%m-%d"), "dataset": dataset, "accuracy": "%.2f" % accuracy,
                "f1_score": "%.2f" % f_score, "average_llm_judge_score": "%.2f" % average_llm_judge_score,
            }
        ]
    )
    cols = ["date", "dataset", "accuracy", "f1_score", "average_llm_judge_score"]
    fname = f'data/{dataset}/metrics.csv'

    # Append in csv
    if not os.path.isfile(fname):
        df.to_csv(fname, header=cols)
    else:  # else it exists so append without writing the header
        df.to_csv(fname, mode='a', header=False)


def rescale_number(
    value: float, original_min: float, original_max: float, new_min: float, new_max: float,
):
    """ Rescale a number from one range to another.

    :param value: value to be rescaled
    :param original_min: original minimum
    :param original_max: original maximum
    :param new_min: new minimum
    :param new_max: new_maximum
    :return:
    """
    return ((value - original_min) / (original_max - original_min)) * \
           (new_max - new_min) + new_min


def ask_llm(result_df):
    cols = ["index", "text", "category", "explanation"]
    list_texts = result_df[cols].values.tolist()
    texts_for_requests = [
        texts for texts in chunker(chunk_size=MAX_REQUESTS_PER_TIME - 10, iterable=list_texts)
    ]
    user_prompt = """
    {texts} is a list of lists containing index, text, category, explanation.
    To provide a 'rating' for each categorization how well the category chosen based on explanation between
    {list_categories}" and add one short suggested_instruction, to add in original prompt to improve general
    classification result, not only for {dataset}.
    Give your answer on a scale of 1 to 4: 1: irrelevant explanation or it should another category ;
    2: undetected ambiguity in the text ; 3: could be improved ; 4:correct category based on clear
    explanation."""
    list_categories = result_df["category"].unique()
    prompts = [
        get_prompts_user_bulk(
            chunk_size=10, request_size=3, list_texts=texts, user_prompt=user_prompt,
            list_categories=list_categories,
        ) for texts in texts_for_requests
    ]
    logging.info("Prompt list size = %d", len(prompts))

    results = asyncio.run(judge_all(prompts))
    merged_df = pd.concat(results, ignore_index=True)

    # Find failed categorized data in merged_df
    missing = [n for n in result_df['index'].to_list() if n not in merged_df['index'].to_list()]
    if len(missing) != 0:
        logging.warning("Index of failed data to rerun : %s", missing)
        # rerun
        cols = ["index", "text", "category", "explanation"]
        list_texts = [tuple(lst) for lst in result_df.loc[result_df.index.isin(missing)][cols].values.tolist()]
        bulks = get_prompts_user_bulk(
            chunk_size=10, request_size=3, list_texts=list_texts,
            list_categories=list_categories, user_prompt=user_prompt,
        )
        feedbacks_df = asyncio.run(judge_all([bulks]))[0]
        retried_df = pd.concat([merged_df, feedbacks_df], ignore_index=True).sort_values(by='index')
        # recheck missing
        missing_after_retried = [n for n in retried_df['index'].to_list() if n not in result_df['index'].to_list()]
        if len(missing_after_retried) > 0:
            logging.error("Retry failed, missing data index = ", missing_after_retried)
            return retried_df
    return merged_df


def get_prompts_user_bulk(
        chunk_size: int, request_size: int, list_texts: List[str],
        user_prompt: List[str], list_categories):
    """ Generate prompts_user_bulk as list of requests containing texts chunks.

    :param chunk_size: number of texts in each chunk
    :param request_size: number of chunks in each request
    :param list_texts: list of texts
    :return: prompts_user_bulk as list of lists
    """
    requests = [request for request in chunker(chunk_size=chunk_size * request_size, iterable=list_texts)]
    chunks = [[user_prompt.format(texts=chunks, dataset=DATASET, row_nb=len(chunks), list_categories=list_categories)
               for chunks in chunker(
        chunk_size=chunk_size, iterable=chunks)] for chunks in requests]

    return chunks


async def judge_all(prompts: List[str]) -> List[str]:
    """ Async function to process prompts in parallel and preserve order.

    The list of prompts is created synchronously.
    Async categorization runs all tasks concurrently but asyncio.gather() preserves the original order. To merge
    multiple results into one.

    :param prompts: list of lists
    :return:
    """
    semaphore = asyncio.Semaphore(MAX_REQUESTS_PER_TIME)
    tasks = [judge_by_llm(prompts_user_bulk, semaphore) for prompts_user_bulk in prompts]
    results = await asyncio.gather(*tasks)  # Preserves order of input
    return results


async def judge_by_llm(prompts_user_bulk, semaphore) -> None:
    """ Asynchronous function to simulate categorization by LLM.

        :param prompts_user_bulk: list of prompts
        :return: semaphore = asyncio.Semaphore(MAX_REQUESTS_PER_TIME)
    """
    prompt_system = "You are a judge specialized in text classification ."

    client = AsyncOpenAI(api_key=OPEN_API_KEY, base_url="https://llm-api.allobrain.com/")

    # Create a list of coroutine tasks for each batch of prompts
    tasks = [
        generate_text(
            prompt_system,
            prompts_user,
            client,
            semaphore,
        )
        for prompts_user in prompts_user_bulk
    ]

    # Run all tasks concurrently and wait for their completion
    results = await asyncio.gather(*tasks)

    # Parse response:
    feedback_df = pd.DataFrame(
        [dict(feedback) for res in results for feedback in res.feedbacks]
    )
    return feedback_df


# Asynchronous function to send a batch of prompts to the OpenAI API
async def generate_text(
        prompt_system: str,
        prompts_user: List[str],
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
) -> str:
    # Combine all user prompts into a single string with formatting
    # instructions
    prompts_user = f"""
    Each text should be classified separately in the output.
    Explanations should be unified to facilitate induction and less than 15 words.
    Finally I want to have at least top 3 explanations for each category by myself.

    Questions:
    {" - ".join(prompts_user)}
    """

    # Acquire semaphore to respect rate limits
    async with semaphore:
        # Make the API call using the beta chat.completions endpoint
        response = await client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                # Set system behavior
                {"role": "system", "content": prompt_system},
                # User input (bulk questions)
                {"role": "user", "content": prompts_user},
            ],
            # Expect structured output
            response_format=Feedbacks,
            temperature=0,
        )

        # Parse structured answers
        content = response.choices[0].message.parsed
        return content

if __name__ == "__main__":
    main()
