---
license: apache-2.0
task_categories:
- token-classification
language:
- en
size_categories:
- 1K<n<10K
---
# Dataset Card for Sarcasm Detection Dataset

## Dataset Details

### Dataset Description

The Sarcasm Detection Dataset is designed for identifying instances of sarcasm in text. The dataset aims to address difficulties in sarcasm detection due to the subjective and contextual nature of language. 

## Uses

### Direct Use

The dataset can be used for training machine learning models to detect sarcasm in text, which has applications in sentiment analysis, social media monitoring, and natural language understanding tasks.

## Dataset Structure

The dataset consists of text examples labeled as sarcastic or non-sarcastic. Each example is accompanied by metadata indicating sarcasm markers and linguistic patterns.

## Dataset Creation

### Curation Rationale

The dataset was curated to provide a diverse collection of sarcastic and non-sarcastic text examples, aiming to capture the complexities of sarcasm in natural language.

### Source Data

#### Data Collection and Processing

The data collection process involved sourcing text samples from various sources, including social media, online forums, and news articles. Each sample was manually annotated as sarcastic or non-sarcastic by human annotators.

### Annotations [optional]

#### Annotation process

Annotations were performed by human annotators who were provided with guidelines for identifying sarcasm in text. Interannotator agreement was measured to ensure consistency in labeling.

## Bias, Risks, and Limitations

The dataset may contain biases inherent in the selection and annotation process, including cultural biases and subjective interpretations of sarcasm.

### Recommendations

Users are advised to consider the limitations of the dataset when training and evaluating sarcasm detection models.

## Citation [optional]
Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). A Large Self-Annotated Corpus for Sarcasm. In LREC 2018 (pp. 1-6).
Rahman M O, Hossain M S, Junaid T S, et al. Predicting prices of stock market using gated recurrent units (GRUs) neural networks[J]. Int. J. Comput. Sci. Netw. Secur, 2019, 19(1): 213-222.
Yu Y, Si X, Hu C, et al. A review of recurrent neural networks: LSTM cells and network architectures[J]. Neural computation, 2019, 31(7): 1235-1270.
Gole, M., Nwadiugwu, W. P., & Miranskyy, A. (2023). On Sarcasm Detection with OpenAI GPT-based Models.
B. Sonare, J. H. Dewan, S. D. Thepade, V. Dadape, T. Gadge and A. Gavali, "Detecting Sarcasm in Reddit Comments: A Comparative Analysis," 2023 4th International Conference for Emerging Technology (INCET), Belgaum, India, 2023, pp. 1-6, doi: 10.1109/INCET57972.2023.10170613.


