import pandas as pd
import requests
import json
import argparse
import os
import sys

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:20b" # Corrected name with colon

def get_summary_from_ollama(prompt):
    """Sends a prompt to the local Ollama instance and returns the generated text."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 500, # Increased to allow for "thinking" overhead
            "temperature": 0.1
        }
    }
    
    try:
        print(f"Sending request to Ollama ({MODEL_NAME})...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        response.raise_for_status()
        
        resp_json = response.json()
        full_output = resp_json.get("response", "").strip()
        
        # If the model uses a "Summary: " prefix, try to extract just that part
        if "Summary:" in full_output:
            output = full_output.split("Summary:")[-1].strip()
        else:
            output = full_output
            
        if not output:
            print("Warning: Ollama returned an empty response.")
            # Check if it's in the 'thinking' block (some Ollama versions/models)
            thinking = resp_json.get("thinking", "")
            if thinking and "Summary:" in thinking:
                output = thinking.split("Summary:")[-1].strip()
                print("Extracted summary from thinking block.")
            
        return output
    except Exception as e:
        print(f"Ollama Error: {e}")
        return f"Error: {e}"

def summarize_game_reviews(appid, csv_path="scraped_reviews.csv", max_reviews=30):
    """Loads reviews for a game and uses Ollama to summarize the pros and cons."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading reviews for AppID {appid}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
        
    game_reviews = df[df['appid'] == appid]
    
    if game_reviews.empty:
        print(f"No reviews found for AppID {appid}.")
        return

    pos_reviews = game_reviews[game_reviews['voted_up'] == True]['review_text'].dropna().head(max_reviews).tolist()
    neg_reviews = game_reviews[game_reviews['voted_up'] == False]['review_text'].dropna().head(max_reviews).tolist()

    # --- Summarize Positive ---
    if pos_reviews:
        print(f"Summarizing {len(pos_reviews)} positive reviews...")
        pos_text = "\n".join([f"- {str(r)[:300]}" for r in pos_reviews])
        pos_prompt = (
            f"Here are several positive user reviews for a video game:\n\n{pos_text}\n\n"
            "Based on these reviews, summarize the 'Good' aspects of the game in exactly one sentence of no more than 20 words. "
            "End your response with 'Summary: [Your Sentence]'."
        )
        good_summary = get_summary_from_ollama(pos_prompt)
    else:
        good_summary = "No positive reviews available."

    # --- Summarize Negative ---
    if neg_reviews:
        print(f"Summarizing {len(neg_reviews)} negative reviews...")
        neg_text = "\n".join([f"- {str(r)[:300]}" for r in neg_reviews])
        neg_prompt = (
            f"Here are several negative user reviews for a video game:\n\n{neg_text}\n\n"
            "Based on these reviews, summarize the 'Bad' aspects of the game in exactly one sentence of no more than 20 words. "
            "End your response with 'Summary: [Your Sentence]'."
        )
        bad_summary = get_summary_from_ollama(neg_prompt)
    else:
        bad_summary = "No negative reviews available."

    print("\n" + "="*50)
    print(f"Ollama Summary for AppID {appid}")
    print("="*50)
    print(f"The Good: {good_summary}")
    print(f"The Bad:  {bad_summary}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize game reviews using local Ollama instance.")
    parser.add_argument("appid", type=int, help="Steam AppID of the game.")
    parser.add_argument("--csv", default="scraped_reviews.csv", help="Path to reviews CSV.")
    parser.add_argument("--limit", type=int, default=30, help="Max reviews to pass to LLM per category.")
    
    args = parser.parse_args()
    summarize_game_reviews(args.appid, args.csv, args.limit)
