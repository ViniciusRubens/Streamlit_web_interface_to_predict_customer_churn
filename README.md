# Construction and Deploy for a Machine Learning Model to Predict Customer Churn using Random Forest and Streamlit

### About the project
This project delivers a robust, end-to-end Machine Learning solution designed to forecast customer churn. Spanning from problem conceptualization to model deployment, the system utilizes GPU to train a **Random Forest Classifier** model capable of identifying customers with a high probability of discontinuing service.

### Problem
Customer churn is a critical metric for any subscription-based business. High churn rates directly impact revenue and indicate dissatisfaction. Identifying patterns in historical data manually is inefficient and reactive. The challenge is to predict **who** will leave and **why**, before it happens.

### Solution
We developed a predictive system that:
1.  Ingests historical customer data (demographics, usage patterns, contract details).
2.  Processes data.
3.  Trains a **Random Forest Classifier** to learn interactions between variables.
4.  Provides a user-friendly Web Interface (Streamlit) for real-time predictions on new customer profiles.

### Tech stack
* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, Numpy, cuDF
* **Machine Learning:** 
    * **Training:** RAPIDS cuML (Random Forest Classifier on GPU)
    * **Evaluation:** Scikit-Learn (Metrics)
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit
* **Environment Management:** Conda

---

### Data Source & Pipeline

**Source:** The dataset consists of synthetic data mimicking real-world telecom customer profiles. It includes features such as:
* **Demographics:** Age.
* **Services:** Monthly usage, Monthly bill value, Plan type (Basic, Standard, Premium).
* **Engagement:** Contract duration, Customer satisfaction score.

**Pipeline:**
1.  **Ingestion:** Loading csv file.
3.  **Preprocessing:** * Scaling numerical variables using `StandardScaler`.
    * One-Hot Encoding for categorical variables (Plans, Contract Types).
4.  **Split:** Separation into Train and Test sets.
5.  **Modeling:** Training the Random Forest algorithm.

### Model

```bash
{
    "test_metrics": {
        "accuracy": 0.7833,
        "confusion_matrix": [
            [
                150,
                32
            ],
            [
                33,
                85
            ]
        ],
        "classification_report": {
            "0": {
                "precision": 0.819672131147541,
                "recall": 0.8241758241758241,
                "f1-score": 0.821917808219178,
                "support": 182.0
            },
            "1": {
                "precision": 0.7264957264957265,
                "recall": 0.7203389830508474,
                "f1-score": 0.723404255319149,
                "support": 118.0
            },
            "accuracy": 0.7833333333333333,
            "macro avg": {
                "precision": 0.7730839288216338,
                "recall": 0.7722574036133358,
                "f1-score": 0.7726610317691636,
                "support": 300.0
            },
            "weighted avg": {
                "precision": 0.7830227453178273,
                "recall": 0.7833333333333333,
                "f1-score": 0.7831691440784999,
                "support": 300.0
            }
        }
    }
}
```

---

## Getting Started

Follow these instructions to set up a local copy of the project.

### Prerequisites

In this project I used `conda` for environment management.
* Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Installation


1.  **Clone the repository** (or download the files to a local folder).
    ```bash
    git clone git@github.com:ViniciusRubens/Streamlit_web_interface_to_predict_customer_churn.git
    cd your-repository-name
    ```

2.  **Create a new conda environment** (this example uses `project_env` as the name):
    ```bash
    conda create --name project_env python=3.12
    ```

3.  **Activate the new environment:**
    ```bash
    conda activate project_env
    ```

4.  **Install pip** into the environment:
    ```bash
    conda install pip
    ```

5.  **Install the required dependencies** from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

6. **Install cuML and cuDF dependencies**:
    ```bash
    #obs.: Runs nvidia-smi command to see your cuda version
    conda install -c rapidsai -c conda-forge -c nvidia     cudf cuml cuda-version=12.0
    ```

---

## Usage


With your `project_env` environment still active, run the application using the following command to start Streamlit application:

```bash
streamlit run app.py
```

You will see something like that in browser:
![](/assets/images/churn_home.png)

---

## How it Works (Step-by-Step)

1.  **Data Entry:** The user inputs customer metrics (e.g., Age, Monthly Usage, Contract Type) via the Streamlit Web Interface.

2.  **Preprocessing:** The system automatically applies **Standardization** (using the saved `StandardScaler`) and **One-Hot Encoding** to match the model's training schema.

3.  **Inference:** The pre-trained **Random Forest Classifier** processes the transformed data to calculate the probability of churn.

4.  **Output:** The application returns a binary classification (**Churn / No Churn**) with immediate visual feedback for decision-making.

---

## Cleanup

To deactivate and remove the conda environment (optional).

1.  **Deactivate the environment:**
    ```bash
    conda deactivate
    ```

2.  **Remove the environment (optional):**
    ```bash
    conda remove --name project_env --all
    ```

---

## License

Distributed under the MIT License. See `LICENSE` file for more information.


















# Projeto 2 - Prevendo o Churn de Clientes com RandomForest - Da Concepção do Problema ao Deploy

# Abra o terminal ou prompt de comando, navegue até a pasta com os arquivos e execute o comando abaixo para criar um ambiente virtual:

conda create --name dsadeploymlp2 python=3.13

# Ative o ambiente:

conda activate dsadeploymlp2 (ou: source activate dsadeploymlp2)

# Instale o pip e as dependências:

conda install pip
pip install -r requirements.txt 

# Execute o comando abaixo para acessar o jupyter notebook e treinar o modelo:

jupyter notebook

# Execute o comando abaixo para o deploy do modelo:

streamlit run deploy.py

# Use os comandos abaixo para desativar o ambiente virtual e remover o ambiente (opcional):

conda deactivate
conda remove --name dsadeploymlp2 --all



(project_4) vinicius-rubens@vinicius-rubens:~/dev/Project_4/modelling/src$ conda install -c rapidsai -c conda-forge -c nvidia \
    cudf cuml cuda-version=12.0

nvidia-smi

