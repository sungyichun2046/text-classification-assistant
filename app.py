""" Classify text based on user defined categories using multiple async OpenAI requests simultaneously"""

import asyncio
import logging
from datetime import datetime
from io import StringIO
from itertools import islice
from typing import Iterable, List

import pandas as pd
import streamlit as st
import tiktoken
from openai import AsyncOpenAI
from pydantic import BaseModel
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

from config import OPEN_API_KEY

# Set the maximum number of concurrent API requests allowed
MAX_REQUESTS_PER_TIME = 100

# Semaphore limits the number of simultaneous API
# requests to avoid hitting rate limits
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


MODEL = "gpt-4o-mini"
# set a lower value of MAX_CONTENT_LENGTH to be safe
MAX_CONTEXT_LENGTH = 128000 - 10000


# Define the expected structure of the API response using Pydantic for data validation
class Category(BaseModel):
    """ Category object attributes.

    index: intput text index
    text: string
    category: category decided by LLM
    """
    index: int
    text: str
    category: str
    explanation: str


class Categories(BaseModel):
    """ List of categories.

    categories: list of Category object.
    """
    categories: List[Category]


async def categorize_by_llm(prompts_user_bulk: List, semaphore: asyncio.Semaphore) -> None:
    """ Asynchronous function to simulate categorization by LLM.

    :param prompts_user_bulk: list of user prompts to classify texts
    :param semaphore: synchronization primitive to limit the number of simultaneous operations in a section of code.
    :return:
    """
    prompt_system = "You are a helpful text classification assistant."

    # Initialize the OpenAI async client using the API
    try:
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

    except AsyncOpenAI.BadRequestError as error:
        logging.error("Request too large, error = %s", error)
        with st.chat_message("assistant"):
            st.markdown("Sorry I can't classify data with your defined list")
    else:
        # Parse response
        categories_df = pd.DataFrame([dict(cat) for res in results for cat in res.categories])

    return categories_df


def main(summarize=False):
    """ Load csv file, categorize text data based on list of categories defined by user."""

    # Title
    st.markdown(
        """
        <h1 style="color: blue; text-align: right;
            font-size: 48px;
            text-shadow: 2px 2px 2px LightBlue;">Classification Assistant</h1>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )

    load_csv_data_sidebar()

    st.session_state.text_col = "text"
    if list_categories := st.chat_input(
            "Enter your list of categories separated by comma (e.g. positive,negative,neutral):",
            disabled=not (st.session_state.data_file)

    ):
        st.chat_message("user").markdown(list_categories)
        nb_tokens = count_tokens(col_name=st.session_state.text_col)

        # Count tokens number to decide if summarize texts
        if summarize:
            if nb_tokens > MAX_CONTEXT_LENGTH:
                logging.warning(
                    "Tokens number = %d (max context window size = %d), use summaries instead texts",
                    nb_tokens, MAX_CONTEXT_LENGTH
                )

                summarize()
                st.session_state.text_col = 'summary'
        # Multiple sending of requests based on MAX_REQUESTS_PER_TIME

        # list_texts are list of lists containing index and text
        list_texts = st.session_state.texts_df[['index', st.session_state.text_col]].values.tolist()
        texts_for_requests = [
            texts for texts in chunker(chunk_size=MAX_REQUESTS_PER_TIME-10, iterable=list_texts)
        ]
        start_time = datetime.now()

        user_prompt = """
            Text enclosed in angle brackets, <{texts}>, is a list of (index, text).
            Classify each text into classes {list_categories}"""
        prompts = [
            get_prompts_user_bulk(
                chunk_size=10, request_size=3, list_texts=texts, list_categories=list_categories,
                user_prompt=user_prompt,
            ) for texts in texts_for_requests
        ]

        logging.info("Prompts list size = %d", len(prompts))
        results = asyncio.run(categorize_all(prompts))
        merged_df = pd.concat(results, ignore_index=True)

        # Find failed categorized data in merged_df
        missing = [n for n in st.session_state.texts_df['index'].to_list() if n not in merged_df['index'].to_list()]

        datetime.now() - start_time
        logging.info(
            "Total llm requesting time = %s seconds on %d data",
            datetime.now() - start_time, merged_df.shape[0]
        )
        if len(missing) != 0:
            logging.error("Failed categorized data to categorize again: %s", missing)

        # Display result, could be downloaded
        with st.chat_message("assistant"):
            st.dataframe(merged_df)


async def categorize_all(prompts: List[str]) -> List[str]:
    """ Async function to process prompts in parallel and preserve order.

    The list of prompts is created synchronously.
    Async categorization runs all tasks concurrently but asyncio.gather() preserves the original order. To merge
    multiple results into one.

    :param prompts: list of prompts.
    :return:
    """
    semaphore = asyncio.Semaphore(MAX_REQUESTS_PER_TIME)
    tasks = [categorize_by_llm(prompts_user_bulk, semaphore) for prompts_user_bulk in prompts]
    results = await asyncio.gather(*tasks)  # Preserves order of input

    return results


def load_csv_data_sidebar():
    if "data_file" not in st.session_state:
        st.session_state.data_file = None

    # initialise the system prompt
    prompt_template = ""
    with st.sidebar:

        chosen_file = st.file_uploader(
            "Choose a csv file where first column containing texts to classify", type=["csv", "xls", "xlsx"]
        )

        # Read as string
        if st.session_state.data_file != chosen_file:
            st.session_state.data_file = chosen_file

            # Convert to a string based IO:
            stringio = StringIO(st.session_state.data_file.getvalue().decode("utf-8"))

            csv_data = stringio.read()
            string_io = StringIO(csv_data)

            # Convert it to dataframe
            st.session_state.texts_df = pd.read_csv(string_io, usecols=[0])
            logging.info("Input data size = %d", st.session_state.texts_df.shape[0])

            st.session_state.texts_df.columns = [st.session_state.text_col]
            # Add index as a column
            st.session_state.texts_df = st.session_state.texts_df.reset_index()
            st.write(st.session_state.texts_df.shape)

            st.header("Data")
            st.markdown(
                f"""<div style="color: blue; text-align: right;">Reading: {st.session_state.data_file.name}</div>""",
                unsafe_allow_html=True)

        if st.session_state.data_file is not None:
            st.dataframe(pd.read_csv(chosen_file))

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": prompt_template}]
    else:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            if isinstance(message, dict):
                if message["role"] != 'system':
                    with st.chat_message(message["role"]):
                        st.markdown(message['content'])
            elif isinstance(message, list):
                pass
            else:  # ChatCompletionMessage object
                if message.role != 'system':
                    with st.chat_message(message.role):
                        st.markdown(message.content)


def summarize() -> None:
    """ Summarize text.

    :return:
    """
    summarizer_start = datetime.now()
    # Summarization failed on all upper text
    col = st.session_state.text_col
    st.session_state.texts_df.loc[
        st.session_state.texts_df[col].str.isupper(), col
    ] = st.session_state.texts_df[col].str.lower()
    st.session_state.texts_df['summary'] = st.session_state.texts_df[col].apply(add_summary)

    logging.info("Summarizer execution time = %s", datetime.now() - summarizer_start)


def add_summary(text: str, lan='english', summary_size=1) -> str:
    """ Add column summary in dataframe.

    :param text: text to summarize
    :param lan: language
    :param summary_size: summary size
    :return: summary
    """
    parser = PlaintextParser.from_string(text, Tokenizer(lan))
    summarizer = LexRankSummarizer()

    list_summary = summarizer(document=parser.document, sentences_count=summary_size)

    # Convert list to string, return original text if summarization fails
    summary = text
    try:
        if len(list_summary) > summary_size:
            summary = '.'.join(str(sent) for sent in list_summary)
        elif len(list_summary) == summary_size:
            summary = str(list_summary[0])
    except Exception as error:
        logging.error(
            "Summarization failed : text = %s, summary = %s because of %s", text, list_summary, error
        )
        pass
    return summary


def count_tokens(col_name: str) -> int:
    """  Count tokens size to avoid OpenAI ContextWindowExceededError.

    :param col_name: column name to count tokens
    :return: number of tokens in this column
    """
    response_format_schemas_tokens_nb = 100
    encoding = tiktoken.encoding_for_model(MODEL)
    text_data = st.session_state.texts_df[col_name].to_list()

    text_input = ' '.join([text for text in text_data])
    num_tokens_text = len(encoding.encode(str(text_input))) + response_format_schemas_tokens_nb
    return num_tokens_text


def get_prompts_user_bulk(
        chunk_size: int, request_size: int, list_texts: List[str], list_categories: List[str], user_prompt: str
) -> List[str]:
    """ Generate prompts_user_bulk as list of requests containing texts chunks.

    :param chunk_size: number of texts in each chunk
    :param request_size: number of chunks in each request
    :param list_texts: list of texts
    :param list_categories: list of categories
    :return: prompts_user_bulk as list of lists
    """
    requests = [request for request in chunker(chunk_size=chunk_size * request_size, iterable=list_texts)]
    chunks = [[user_prompt.format(texts=chunks, list_categories=list_categories) for chunks in chunker(
        chunk_size=chunk_size, iterable=chunks)] for chunks in requests]
    return chunks


def chunker(chunk_size: int, iterable: Iterable) -> tuple:
    """ Split iterable into chunks based on chunk_size

    :param chunk_size: int
    :param iterable:
    :return:
    """
    iterable = iter(iterable)
    while True:
        x = tuple(islice(iterable, chunk_size))
        if not x:
            return
        yield x


# Asynchronous function to send a batch of prompts to the OpenAI API
async def generate_text(
    prompt_system: str,
    prompts_user: List[str],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> str:
    """ Get categories as list of Categories.

    :param prompt_system: prompt of system role
    :param prompts_user: prompt of user role
    :param client: openai client
    :param semaphore: synchronization primitive to limit the number of simultaneous operations in a section of code.
    :return: parsed text
    """
    # Combine all user prompts into a single string with formatting instructions
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
            response_format=Categories,
            temperature=0,
        )

        # Parse structured answers
        content = response.choices[0].message.parsed
        return content


if __name__ == '__main__':
    main()
