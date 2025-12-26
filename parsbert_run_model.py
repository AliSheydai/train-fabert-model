import os
import sys
import time
from transformers import pipeline

# --- VISUAL SETTINGS (ANSI Colors) ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# Clear screen for a fresh start
os.system('cls' if os.name == 'nt' else 'clear')

# --- 1. BANNER & SETUP ---
def print_banner():
    print(Colors.CYAN + Colors.BOLD + "="*60)
    print("   ParsBERT SENTIMENT ANALYZER |  XAI PROJECT DEMO")
    print("   Powered by: HooshvareLab & Hugging Face")
    print("="*60 + Colors.RESET)

print_banner()

# Model Path (Ensure this matches your folder name)
model_path = "./final_parsbert_model"

# Check path
if not os.path.exists(model_path):
    print(Colors.RED + f"\n[!] CRITICAL ERROR: Model path '{model_path}' not found." + Colors.RESET)
    sys.exit(1)

print(Colors.YELLOW + "\n[SYSTEM] Initializing AI Core..." + Colors.RESET)
print(Colors.YELLOW + f"[SYSTEM] Loading ParsBERT from: {model_path}" + Colors.RESET)

# --- 2. LOAD MODEL ---
try:
    # device=-1 for CPU
    classifier = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=-1)
    # Fake loading delay for effect
    time.sleep(1) 
    print(Colors.GREEN + "[SUCCESS] ParsBERT Model Loaded! Engine Ready." + Colors.RESET)
    print("-" * 60)
except Exception as e:
    print(Colors.RED + f"[ERROR] Failed to load model: {e}" + Colors.RESET)
    sys.exit(1)

# --- 3. PREDICTION FUNCTION WITH UI ---
def draw_bar(score, length=20):
    fill = int(length * score)
    bar = '█' * fill + '░' * (length - fill)
    return bar

def predict(text, auto_mode=False):
    if not text.strip(): return

    start_time = time.time()
    result = classifier(text)
    end_time = time.time()
    
    label = result[0]['label']
    score = result[0]['score']
    process_time = (end_time - start_time) * 1000 # ms

    # Visual Logic (ParsBERT output logic)
    # Mapping POSITIVE/LABEL_1 to Green, else Red
    if label in ['POSITIVE', 'LABEL_1']:
        readable_label = "POSITIVE (Satisfied)"
        color = Colors.GREEN
        icon = "😊"
    else:
        readable_label = "NEGATIVE (Dissatisfied)"
        color = Colors.RED
        icon = "😡"

    # --- THE FANCY OUTPUT ---
    print(f"\n{Colors.BOLD}Input Text:{Colors.RESET} \"{text}\"")
    print(color + "┌" + "─"*50 + "┐")
    print(f"│ {icon} Sentiment : {Colors.BOLD}{readable_label:<25}{color} │")
    print(f"│ 🎯 Confidence: {score:.4f}  [{draw_bar(score)}]   │")
    print(f"│ ⚡ Latency   : {process_time:.2f} ms                             │")
    print("└" + "─"*50 + "┘" + Colors.RESET)
    
    if auto_mode:
        time.sleep(0.5)

# --- 4. RUN DEMO ---
print(Colors.CYAN + "\n>>> RUNNING AUTOMATED DIAGNOSTICS (Parsbert EXAMPLES):" + Colors.RESET)
test_sentences = [
    # --- (Easy Positive) ---
    "واقعا عالیه، کیفیت ساختش نسبت به قیمتش خیلی خوبه.",
    "من که خیلی راضی بودم، رنگش دقیقا مثل عکسش بود.",
    
    # --- (Easy Negative) ---
    "افتضاح بود، اصلا پیشنهاد نمیکنم پولتون رو دور نریزید.",
    "حیف پول، اصلا شبیه چیزی که سفارش دادم نبود.",

    # --- (Mixed/Complex) ---
    "کیفیتش خوبه اما قیمتش خیلی گرونه و ارزش خرید نداره.",

    # --- (Slang  ---
    "خیلی خفنه، دمتون گرم!",
    "بدک نبود",
    "ترکوندین با این سرعت ارسال، عالی بود.",

    # --- (Short Text) ---
    "محشر.",
    "مزخرف.",
    "معمولی.",
    "پیشنهاد میشه."
]
for sentence in test_sentences:
    predict(sentence, auto_mode=True)

# --- 5. INTERACTIVE LOOP ---
print(Colors.CYAN + "\n" + "="*60)
print("   INTERACTIVE MODE ENGAGED")
print("   Type a Persian sentence to analyze (or 'exit' to quit)")
print("="*60 + Colors.RESET)

while True:
    try:
        user_input = input(Colors.BLUE + "\nUSER@ParsBERT:~$ " + Colors.RESET)
        if user_input.lower() in ["exit", "quit"]:
            print(Colors.YELLOW + "\n[SYSTEM] Shutting down..." + Colors.RESET)
            break
        predict(user_input)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Force Exit.")
        break