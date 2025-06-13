import re
import tqdm
import numpy as np
import os
import time
import pandas as pd

def tokenize_and_chunk(row, tokenizer, max_tokens=512, text_col='text'):
    sentences = re.split(r'(?<=[.!?]) +', row[text_col])
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        token_count = len(tokenizer.tokenize(sentence))

        if current_tokens + token_count <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += token_count
        else:
            if current_chunk:
                chunk_row = row.to_dict()
                chunk_row[text_col] = ' '.join(current_chunk)
                chunks.append(chunk_row)
            current_chunk = [sentence]
            current_tokens = token_count

    if current_chunk:
        chunk_row = row.to_dict()
        chunk_row[text_col] = ' '.join(current_chunk)
        chunks.append(chunk_row)

    return chunks

def encode_in_batches(texts, model, batch_size=10):
    embeddings = []
    valid_indices = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        local_embeddings = []
        for j, text in enumerate(batch):
            global_index = i + j
            try:
                embedding = model.encode(
                    text,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                local_embeddings.append(embedding)
                valid_indices.append(global_index)
            except Exception as e:
                tqdm.write(f"❌ Skipped text index {global_index} due to error: {e}")
                tqdm.write(f"Text content (first 300 chars): {str(text)[:300]}")

        embeddings.extend(local_embeddings)

    return np.array(embeddings), valid_indices

SYSTEM_PROMPT = """You are a legal and environmental disclosure expert. Your task is to determine whether a paragraph of text qualifies as climate litigation.
 
Climate litigation refers to legal actions that materially concern climate change science, policy, or law. These include, but are not limited to:
- Lawsuits targeting false or misleading climate claims (e.g. greenwashing)
- Legal actions over a company’s contribution to climate-related impacts
- Efforts to force climate alignment through human rights or fiduciary duty arguments
- Failure to disclose climate-related risks or impacts
- Breaches of climate-related regulations
- Litigation seeking damages for harms caused by climate change
- Legal challenges to regulatory approvals on the basis of climate misalignment
 
Your classification must be binary:
- climate_litigation: 1 if the paragraph relates to litigation that is specifically about climate change
- climate_litigation: 0 otherwise
 
Be especially careful not to classify the following as climate litigation:
- Environmental lawsuits unrelated to climate change, such as:
  - Pollution from toxic substances (e.g., PFAS, oil spills, nuclear hazard)
  - Air pollution regulations that do not focus on climate mitigation
  - Destruction of ecosystems not linked to climate change
  - Breaches of water, soil, clean air or conservation laws without reference to climate change 
 
Do not classify as climate litigation the litigation against governments or public authorities or that aims to challenge a law, regulation or public policy, unless the litigation also targets a private actor or company. Legal challenges to climate-related laws and regulations brought solely against public authorities (e.g., EPA), where no private company is sued or legally challenged, are not considered climate litigation.
 
Include statements about the adoption or proposal of new rules and laws that facilitate climate litigation, such as the adoption of strict liability statutes that make companies responsible for their historical greenhouse gas emission and statutes that lower the evidentiary requirements of climate litigation cases, notably around the causality link between climate damages and greenhouse gas emissions.
 
Do not classify as climate litigation simply because the case mentions sustainability, ESG, or environmental risk. Focus only on litigation where climate change itself is central to the legal reasoning, claims, or remedies sought.

Air on the side of caution and do not classify as climate litigation unless the text clearly indicates a direct connection to climate change issues. If in doubt, classify as 0 (not climate litigation).
"""

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_candidate_chunks(embedding_model, df, k=100):
    """Retrieve chunks most likely to contain climate litigation using cosine similarity"""
    litigation_query = "Paragraphs discussing actual, pending, threatened, or potential legal action or liability related to climate change or greenhouse gas emissions. This includes, but is not limited to, lawsuits, court proceedings, appeals, enforcement actions by public authorities (such as investigations, notices of violation, fines, or sanctions), consent decrees, complaints, legal risks, and references to litigation or liability exposure."
    
    # Embed the query
    query_embedding = embedding_model.encode([litigation_query])[0]
    
    # Calculate similarities
    similarities = []
    for idx, row in df.iterrows():
        similarity = cosine_similarity(query_embedding, row['embedding'])
        similarities.append({'index': idx, 'similarity': similarity})
    
    # Sort by similarity and get top k
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    top_indices = [item['index'] for item in similarities[:k]]
    
    candidate_chunks = df.loc[top_indices].copy()
    candidate_chunks['retrieval_similarity'] = [item['similarity'] for item in similarities[:k]]
    
    print(f"Retrieved {len(candidate_chunks)} candidate chunks for classification")
    return candidate_chunks

def retrieve_similar_examples(query_text, embedding_model, groundtruth_df, k=6):
    """
    Retrieve k most similar ground-truth examples, half where 'Climate Litigation' == 1 and half where == 0.
    Assumes groundtruth_df has a 'Climate Litigation' column with values 1 (positive) or 0 (negative).
    """
    import math

    # Embed the query text
    query_embedding = embedding_model.encode([query_text])[0]

    # Determine counts for each class
    n_pos = math.ceil(k / 2)
    n_neg = k - n_pos

    # Split ground truth by the 'Climate Litigation' label
    pos_df = groundtruth_df[groundtruth_df['climate_litigation'] == 1].copy()
    neg_df = groundtruth_df[groundtruth_df['climate_litigation'] == 0].copy()

    # Helper to compute top-k by similarity
    def top_k_subset(df_subset, top_k):
        sims = []
        for idx, row in df_subset.iterrows():
            sim = cosine_similarity(query_embedding, row['embedding'])
            sims.append({'index': idx, 'similarity': sim})
        sims.sort(key=lambda x: x['similarity'], reverse=True)
        return sims[:top_k]

    # Get top examples from each class
    top_pos = top_k_subset(pos_df, n_pos)
    top_neg = top_k_subset(neg_df, n_neg)

    # Combine and gather indices
    combined = top_pos + top_neg
    selected_indices = [item['index'] for item in combined]

    # Slice and attach similarity
    similar_examples = groundtruth_df.loc[selected_indices].copy()
    similar_examples['retrieval_similarity'] = [item['similarity'] for item in combined]

    return similar_examples

def classify_with_rag(chunk_text, retrieved_examples, client):
    """Classify a chunk using dynamically retrieved examples"""
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add retrieved examples as few-shot context
        for _, row in retrieved_examples.iterrows():
            messages.append({
                "role": "user",
                "content": f"Paragraph: {row['text']}\nIs this climate litigation? Respond with 'climate_litigation: 1' or 'climate_litigation: 0'"
            })
            messages.append({
                "role": "assistant", 
                "content": f"climate_litigation: {row['climate_litigation']}"
            })

        # Add the actual query
        messages.append({
            "role": "user",
            "content": f"Paragraph: {chunk_text}\nIs this climate litigation? Respond with 'climate_litigation: 1' or 'climate_litigation: 0'"
        })

        response = client.chat.completions.create(
            model="qwen/qwen-2.5-7b-instruct",
            messages=messages,
            temperature=0
        )

        response_dict = response.to_dict()
        content = response_dict['choices'][0]['message']['content']

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        print(f"Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
        return content

    except Exception as e:
        print(f"[ERROR] Failed to classify paragraph: {chunk_text[:80]}...\nException: {e}")
        return "SKIPPED"
    
def run_rag_classification_for_company(embedding_model,
        company_df,
        company_name,
        retrieval_k=100,
        example_k=5,
        start_index=0,
        output_dir="rag_results"
    ):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🔍 Processing company: {company_name} with {len(company_df)} chunks")

    # Debug: show columns
    print(company_df.columns)
    
    # Stage 1: Retrieve top-k candidate chunks within company only
    litigation_query = (
        "Paragraphs about lawsuits or legal actions involving climate change, "
        "greenwashing, regulatory breaches, or liability for emissions."
    )
    query_embedding = embedding_model.encode([litigation_query])[0]
    
    # Compute similarities
    similarities = []
    for idx, row in company_df.iterrows():
        sim = cosine_similarity(query_embedding, row['embedding'])
        similarities.append({'index': idx, 'similarity': sim})
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    top_indices = [item['index'] for item in similarities[:retrieval_k]]

    candidate_chunks = company_df.loc[top_indices].copy()
    candidate_chunks['retrieval_similarity'] = [item['similarity'] for item in similarities[:retrieval_k]]
    
    # Stage 2 & 3: Classify
    results = []
    print(f"Classifying {len(candidate_chunks)} candidate chunks for {company_name}")
    for i, (_, row) in enumerate(candidate_chunks.iterrows()):
        if i < start_index:
            continue

        chunk_text = row['text']
        similar_examples = retrieve_similar_examples(chunk_text, k=example_k)
        classification = classify_with_rag(chunk_text, similar_examples)

        result = {
            'company': company_name,
            'original_index': row.name,
            'year': row.get('year', None),
            'text': chunk_text,
            'climate_litigation': classification,
            'retrieval_similarity': row['retrieval_similarity'],
            'num_examples_used': len(similar_examples)
        }
        results.append(result)

        print(
            f"[{company_name}][Year {result['year']}] Chunk {i+1}/{len(candidate_chunks)}: "
            f"{chunk_text[:50]}... -> {classification}"
        )

        # Periodic save
        if (i + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_path = os.path.join(
                output_dir,
                f"rag_results_{company_name}_{result['year']}.csv"
            )
            temp_df.to_csv(temp_path, index=False)
            print(f"[{company_name}][Year {result['year']}] Progress saved after {i+1} chunks")

        time.sleep(3)

    # Final save
    results_df = pd.DataFrame(results)
    final_path = os.path.join(
        output_dir,
        f"rag_results_{company_name}_{results_df['year'].iloc[0] if not results_df.empty else 'NA'}.csv"
    )
    results_df.to_csv(final_path, index=False)
    print(f"✅ Done with {company_name} Year {results_df['year'].iloc[0] if not results_df.empty else 'NA'}. "
          f"Results saved to {final_path}")
    return results_df