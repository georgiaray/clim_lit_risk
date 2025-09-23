**CLR methodology**

**Creating original dataframe (create_df.ipynb)**

In creating the original data frame, we started from the Sabin Center
database and then:

- Only retained information on the company sued, the date of the suit,
  the name of the case, the impact, and whether it was positive or
  negative

- Paired this up with companies_df ([Note to self: not sure where this
  is from]{.mark}) from which we kept name, date, and case name

  - I think it could be some part of the Sabin Center database as well

- Removed rows after a certain point because they are more like a
  scratchpad, not data

- Combine the two databases so we have a dataframe that includes company
  name, case name, filing date, decision date, impact, and whether it
  was positive or negative

- Saved this as [litigation_w_decision.csv]{.mark}

**Getting 10ks from EDGAR website (get_data.py)**

To build our dataset of company disclosures, we:

- Started with a CSV of relevant companies containing ticker symbols and
  CIK numbers (SEC identifiers; in this case the csv was
  [american_companies.csv]{.mark} and entered manually)

  - For each company, retrieved all available 10-K annual reports from
    the SEC\'s EDGAR database

  - Used the SEC\'s RSS feed system to get filing metadata (accession
    numbers and dates)

- Downloaded the actual filing documents by:

  - Finding the best HTML version of each filing (largest file size,
    excluding technical/metadata files)

- Cleaned the raw filing text by:

  - Removing XBRL tags and other technical markup

  - Joining fragmented lines back together (SEC formatting often breaks
    sentences across lines)

  - Trimming everything before \"SECURITIES AND EXCHANGE COMMISSION\" to
    focus on the actual filing content

- Saved each cleaned filing as a text file named by ticker and filing
  date (e.g., \"AAPL_2023-09-30.txt\")

- These files are then saved in [a folder called 10ks]{.mark} which are
  further subsetted by the company ticker

**Organizing 10ks (data_exploration.ipynb)**

Chose relevant companies and their data by:

- Extracting only filings 2014 onwards

<!-- -->

- Combined multiple filings per company per year into single text
  records

- Assessed data coverage by creating a pivot table showing which
  companies have filings for which years and selected those companies
  with complete coverage across our timeline

- Saved coverage analysis as \"[10k_coverage.csv]{.mark}\" for reference

<!-- -->

- Saved filtered dataset as \"[full_data_filtered.csv]{.mark}\"

**Choosing the best models (testing.ipynb)**

To choose the best LLM for our main RAG pipeline complete with
cross-model validation we first needed to select the best three,
balancing performance and cost. To do this, we:

- Tested the 10 cheapest models on OpenRouter against our manually
  labeled ground truth data (60 manually labelled samples)

- Enhanced classification with RAG (Retrieval-Augmented Generation)

  - Used embeddings to find 6 most similar examples for each test case

    - Groundtruth embeddings saved as gt_embeddings.npy

- Evaluated each model on accuracy, precision, recall, and F1 score

- Calculated estimated costs for processing our full dataset (66
  companies × \~1,100 chunks each)

- Created cost vs. performance comparison to identify optimal model for
  our budget

Output:

- Results saved as [model_testing_results.csv]{.mark} ranked by F1 score

- Cost-performance visualization to guide final model selection

  - 3-lunaris-8b, gemma-3-4b-it, and llama-3.2-3b-instruct chosen as the
    best three

**Perform analysis (RAG.ipynb)**

To apply our selected model to the full dataset, we:

- Loaded complete 10-K dataset and ground truth examples

- Chunked all text into manageable pieces and removed duplicates

- Generated embeddings for both document chunks and ground truth
  examples using the nomic embedding model

  - Saved embeddings to disk for reuse ([doc_embeddings.npy]{.mark},
    [gt_embeddings.npy]{.mark})

- Implemented two-step RAG classification process:

  - Step 1: Retrieve top 100 most similar document chunks

  - Step 2: Select 5 best ground truth examples for few-shot learning

  - Chosen because the two-step RAG approach is more efficient than
    comparing every chunk to all examples

- Processed companies systematically, organized by company and year

  - Built in restart capability from specific company/year (starting
    from \"SUN\" company, 2019)

  - Saved results in nested folder structure for each model

And then this was repeated three times for each model; results saved in
folders called [SAO_RAG_results]{.mark}, [META_RAG_results]{.mark}, and
[GEMMA_rag_results]{.mark}

**Analysis and assessment (assessing_RAG_results.ipynb)**

To analyze our climate litigation classification results, we:

- Loaded existing litigation data from the Sabin Center database for 10
  companies of interest

  - Removed duplicate cases (keeping latest decision dates) and filtered
    to our target companies

  - Calculated active litigation counts by year (cases filed but not yet
    decided)

- Processed RAG classification outputs from three different models (SAO,
  GEMMA, META)

  - Extracted binary classifications (0/1) from model responses using
    regex parsing

  - [Note to self: Some responses couldn\'t be parsed and were marked
    for manual review]{.mark}

- Compared model agreement and disagreement patterns

  - Found [86% complete agreement across all three models]{.mark}

  - [90% agreement when allowing two-of-three consensus]{.mark}

  - [Note to self: also need to measure precision and recall because we
    have an overabundance of negative cases in this data]{.mark}

- Created visualizations comparing actual litigation cases vs.
  disclosure statements over time

  - Plotted active cases, new filings, and detected disclosures for
    trend analysis

- Analyzed disagreement cases to understand model differences

  - Sampled disagreement cases where SAO differed from the other two
    models

  - Saved samples for manual validation by domain expert

    - [Domain only disagrees with SAO (when SAO is alone) 8.33% of the
      time (2/24)]{.mark}

**Helper metods can be found in utils.py and are used throughout various
notebooks and py files**
