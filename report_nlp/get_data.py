import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path
import pandas as pd
import feedparser
import re

headers = {'User-Agent': 'Your Name your.email@example.com'}

def get_10k_accessions(cik, max_entries=100):
    feed_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&owner=exclude&count={max_entries}&output=atom"
    response = requests.get(feed_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    entries = soup.find_all("entry")

    results = []
    for entry in entries:
        acc_tag = entry.find("accession-number")
        date_tag = entry.find("filing-date")
        if acc_tag and date_tag:
            accession_number = acc_tag.text.strip()
            filing_date = date_tag.text.strip()
            results.append((accession_number, filing_date))
    return results

def find_best_html(cik, accession_number):
    acc_num_nodash = accession_number.replace('-', '')
    base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_num_nodash}/"
    index_url = base_url + "index.json"

    response = requests.get(index_url, headers=headers)
    response.raise_for_status()
    files = response.json()['directory']['item']

    candidates = [
        (f['name'], int(f['size'])) for f in files
        if f['name'].lower().endswith('.htm') and
           not any(x in f['name'].lower() for x in ['cal', 'def', 'lab', 'pre', 'xsd', 'xml'])
    ]

    print(f"Found {len(candidates)} HTML candidates for {accession_number}")
    if not candidates:
        return None

    best_file = max(candidates, key=lambda x: x[1])[0]
    return base_url + best_file

def join_lines(text):
    lines = text.splitlines()
    merged = []
    buffer = ""
    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                merged.append(buffer)
                buffer = ""
            continue
        if len(line) < 30 and buffer:
            buffer += " " + line
        else:
            if buffer:
                merged.append(buffer)
            buffer = line
    if buffer:
        merged.append(buffer)
    return "\n".join(merged)

def trim_to_sec_start(text):
    marker = "SECURITIES AND EXCHANGE COMMISSION"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx:]

def download_and_clean_10k(html_url):
    response = requests.get(html_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup.find_all(lambda tag: tag.name and tag.name.startswith("ix:")):
        tag.unwrap()

    text = soup.get_text(separator="\n", strip=True)
    text = join_lines(text)
    trimmed_text = trim_to_sec_start(text)
    return trimmed_text

def process_company(ticker, cik):
    output_dir = Path("10k_2") / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    accessions = get_10k_accessions(cik)
    if not accessions:
        print(f"No 10-K filings found for {ticker} / CIK {cik}")
        return

    print(f"Found {len(accessions)} 10-K filings for {ticker}. Downloading...")

    for acc_num, date in accessions:
        try:
            html_url = find_best_html(cik, acc_num)
            if not html_url:
                print(f"No suitable HTML file found for accession {acc_num}")
                continue
            print(f"Downloading from: {html_url}")
            text = download_and_clean_10k(html_url)
            file_name = f"{ticker}_{date}.txt"
            file_path = output_dir / file_name
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved cleaned 10-K to {file_path}")
        except Exception as e:
            print(f"Failed to download or process {acc_num}: {e}")

def main():
    mode = input("Use CSV or manual entry? (csv/manual): ").strip().lower()
    
    if mode == "csv":
        input_csv = Path("relevant_companies.csv")
        df = pd.read_csv(input_csv, encoding='ISO-8859-1')
        for _, row in df.iterrows():
            ticker = str(row['ticker']).strip().upper()
            cik = str(row['CIK']).strip().zfill(10)
            print(f"\nProcessing {ticker} (CIK: {cik})")
            try:
                process_company(ticker, cik)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    elif mode == "manual":
        ticker = input("Enter company ticker (e.g., AAPL): ").strip().upper()
        cik = input("Enter 10-digit CIK (digits only): ").strip().zfill(10)
        process_company(ticker, cik)
    else:
        print("Invalid input. Please type 'csv' or 'manual'.")

if __name__ == "__main__":
    main()
