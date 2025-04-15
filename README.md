## text-classification-assistant with streamlit, AsyncOpenAI and Docker

[text_assistant_demo.webm](https://github.com/user-attachments/assets/de8c7937-2bc5-4fd0-9611-c34ec9b5063b)

### User interface using Streamlit
 * Add config.py to the root of this project
```commandline
OPEN_API_KEY='sk-********************'
```
 * Without Docker
   * Install dependencies: ```pip install -r requirements.txt```
   * Run ```streamlit run app.py```
 * With Docker
   * ```docker build -t classifier:latest -f Dockerfile .```
   * ```docker run -d --name classifier-container -p 8000:8000 classifier:latest```

 * Test
   * Select one csv/excel file where the first column is text data to classify  
   * Entry list of categories separated by comma and classify them

### Auto evaluation with accuracy, F1-score and LLM as judge score
 * Categorize `data/spam/annotated.csv` using UI, use `spam,ham` as list of categories, to be able to evaluate result
 * Download the output csv in `data/spam/results.csv`
 * Run ```python evaluation.py```

### Results 
 * Evaluation result: ```Accuracy = 97%, F1 score = 97%, LLM judge score = 99% on Spam-Ham dataset containing 5572 data.```
 * The execution time is 1 minute on 5572 data on user interface
 * Folder `data`
   * `results.csv`: output of UI, input of evaluation.py, containing `index`, `text,category`, `explanation` (explanation of choice of category) 
   * `annotated.csv`: input of evaluation.py, containing `Message` (text to classify) and `Category` (annotations: spam/ham)
   * `evaluation.csv`: output containing `text`, `category`, `explanation`, `rating`, `rating_reason`, `suggested_instruction`
   * `metrics.csv`: output containing date `text`, `category`, `explanation`, `rating`, `rating_reason`, `suggested_instruction`

### Project structure
```commandline
├── app.py
├── config.py
├── data
│   └── spam
│       ├── annotated.csv 
│       ├── evaluation.csv 
│       ├── metrics.csv
│       └── results.csv
├── Dockerfile
├── evaluation.py
├── README.md
```





