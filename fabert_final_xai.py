import pandas as pd
import torch
import os
import json # اضافه شده برای خواندن فایل هسته دانش
from transformers import BertTokenizer, BertForSequenceClassification

# --- 1. تنظیمات و بارگذاری ---
model_path = "./final_fabert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# بارگذاری گراف دانش (A-Box)
try:
    df_kg = pd.read_csv('graph.csv') 
    df_kg['Title'] = df_kg['Title'].str.strip()
except:
    print("❌ Error: Knowledge Graph (graph.csv) not found!")

# بارگذاری آنتولوژی (T-Box) از فایل JSON [جدید]
try:
    with open('ontology.json', 'r', encoding='utf-8') as f:
        ONTOLOGY = json.load(f)
    print("✅ Ontology loaded successfully.")
except Exception as e:
    print(f"❌ Error loading ontology.json: {e}")

# --- 2. توابع هوش مصنوعی ---

def get_sentiment_prob(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()

STOP_WORDS = ['خیلی', 'واقعا', 'بود', 'است', 'شد', 'این', 'آن', 'رو', 'در']

def find_impact_word(comment):
    words = comment.split()
    if len(words) < 2: return words[0] if words else ""
    
    base_score = get_sentiment_prob(comment)
    impacts = []
    
    for i in range(len(words)):
        if words[i] in STOP_WORDS: continue 
        
        temp_comment = " ".join(words[:i] + words[i+1:])
        new_score = get_sentiment_prob(temp_comment)
        impact = abs(new_score - base_score)
        impacts.append((words[i], impact))
    
    if not impacts: return words[0]
    return sorted(impacts, key=lambda x: x[1], reverse=True)[0][0]

# --- 3. موتور استدلال XAI ---

def xai_live_demo():
    print("\n" + "█"*65)
    print("   🚀 SYSTEMIC XAI DEMO: DATA-DRIVEN REASONING (JSON + CSV)")
    print("█"*65)
    
    product_query = input("\n📌 Product Name (e.g., A56): ")
    user_comment = input("💬 User Comment: ")
    
    initial_prob = get_sentiment_prob(user_comment)
    is_positive = initial_prob > 0.5
    sentiment_label = "POSITIVE" if is_positive else "NEGATIVE"
    
    print(f"\n🔍 [Step 1] Sentiment: {sentiment_label} ({initial_prob:.2%})")
    
    impact_word = find_impact_word(user_comment)
    print(f"🎯 [Step 2] Anchor Word Found: '{impact_word}'")
    
    try:
        # جستجوی محصول در گراف دانش
        product_data = df_kg[df_kg['Title'].str.contains(product_query, na=False, case=False)].iloc[0]
        
        # شناسایی خودکار جنبه فنی از طریق آنتولوژی JSON
        target_node = None
        for node, data in ONTOLOGY.items():
            if any(k in user_comment for k in data['keywords']):
                target_node = node
                break
        
        if not target_node:
            print("⚠️ Result: No matching technical aspect found in Ontology.")
            return

        # استخراج فکت فنی از گراف دانش بر اساس کلید معرفی شده در JSON
        tech_col = ONTOLOGY[target_node]['tech_column']
        tech_value = product_data.get(tech_col, "N/A")
        
        # منطق استدلال تقابلی (Counterfactual Logic)
        if is_positive:
            # تخریب مثبت به منفی برای اثبات حساسیت
            replacement = ONTOLOGY[target_node]['neg_adj']
        else:
            # اصلاح منفی به مثبت با استفاده از فکت گراف دانش
            pos_desc = ONTOLOGY[target_node]['pos_adj']
            replacement = f"{pos_desc} ({tech_value})"
        
        modified_comment = user_comment.replace(impact_word, replacement)
        
        final_prob = get_sentiment_prob(modified_comment)
        final_label = "POSITIVE" if final_prob > 0.5 else "NEGATIVE"

        # --- نمایش گزارش نهایی ---
        print("\n" + "-"*65)
        print("📝 DYNAMIC XAI REASONING REPORT")
        print("-"*65)
        print(f"Target Feature: {target_node} [Source: ontology.json]")
        print(f"KG Evidence: {tech_value} [Source: graph.csv]")
        print(f"Action: Swapped '{impact_word}' with '{replacement}'")
        print(f"New Sentiment: {final_label} ({final_prob:.2%})")
        
        shift = abs(final_prob - initial_prob)
        print(f"📊 Semantic Impact: {shift:.2%}")
        print("="*65)

    except Exception as e:
        print(f"\n❌ Execution Error: {e}")

if __name__ == "__main__":
    while True:
        xai_live_demo()
        if input("\nAnalyze another? (y/n): ").lower() != 'y': break