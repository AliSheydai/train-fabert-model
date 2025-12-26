import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Configuration and Model Loading (Paths preserved as requested)
model_path = "./final_fabert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. Knowledge Graph Loading
try:
    df_kg = pd.read_csv('graph.csv')
    df_kg['Title'] = df_kg['Title'].str.strip() 
except FileNotFoundError:
    print("❌ Error: 'graph.csv' file not found!")

def get_sentiment(text):
    """Extract sentiment (Binary: Negative or Positive)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    
    # Updated to Binary Labels as per your requirement
    labels = ["Negative", "Positive"] 
    return labels[prediction]

def xai_explain(product_title, comment):
    """
    XAI Engine: Links sentiment to KG technical specs
    """
    # A) Binary Sentiment Detection
    sentiment_label = get_sentiment(comment)
    
    # B) Knowledge Graph Lookup
    try:
        specs = df_kg[df_kg['Title'].str.contains(product_title, na=False)].iloc[0]
    except IndexError:
        return f"⚠️ Error: Product '{product_title}' not found in Knowledge Graph."

    # C) Ontology of Relationships
    ontology = {
        'Battery': ['باتری', 'شارژ', 'دوام', 'دشارژ', 'آمپر'],
        'RAM': ['رم', 'هنگ', 'لگ', 'کندی', 'سرعت', 'حافظه موقت'],
        'Camera': ['دوربین', 'عکس', 'لنز', 'وضوح', 'مگاپیکسل', 'فیلم'],
        'Chipset': ['تراشه', 'پردازنده', 'داغ', 'گرم', 'چیپست'],
        'Storage': ['حافظه داخلی', 'فضا', 'پر شده', 'ذخیره']
    }

    explanations = []
    
    # D) Logic Fixed: Conditional statements are now INSIDE the loop
    for tech_node, keywords in ontology.items():
        if any(word in comment for word in keywords):
            tech_value = specs.get(tech_node, "Unknown")
            
            if sentiment_label == "Negative":
                explanations.append(f"❌ Technical Critique: User dissatisfaction regarding '{tech_node}' (Value: {tech_value}) analyzed against KG specs.")
            else:
                explanations.append(f"✅ Technical Confirmation: High performance of '{tech_node}' (Value: {tech_value}) confirmed by user sentiment.")

    # E) Final Reporting
    if not explanations:
        return f"🤖 FaBERT Analysis: Sentiment '{sentiment_label}' detected, but no matching technical aspects found in KG."
    
    report = f"\n{'='*65}\n"
    report += f"📝 FINAL XAI ANALYSIS REPORT (Binary Sentiment)\n"
    report += f"{'='*65}\n"
    report += f"📌 Product: {specs['Title'][:60]}...\n"
    report += f"🎭 Predicted Sentiment: {sentiment_label}\n"
    report += f"{'-'*65}\n"
    report += "\n".join(explanations)
    report += f"\n{'='*65}"
    
    return report

# --- Test ---
test_comment = "این گوشی باتریش واقعا افتضاحه"
product_name = "A56" 

print(xai_explain(product_name, test_comment))