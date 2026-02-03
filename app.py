import streamlit as st
import pandas as pd
import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification
from pyvis.network import Network
from streamlit.components.v1 import html
import re

# --- تنظیمات صفحه ---
st.set_page_config(page_title="FaBERT vs ParsBERT XAI", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    @import url('https://v1.fontapi.ir/css/Vazir');
    body, div, span, p, h1, h2, h3 { font-family: 'Vazir', sans-serif !important; direction: rtl; text-align: right; }
    .stTextInput > div > div > input { text-align: right; direction: rtl; }
    .highlight { background-color: #ffcccc; padding: 2px 5px; border-radius: 4px; font-weight: bold; }
    .model-box { border: 1px solid #ddd; padding: 15px; border-radius: 10px; background-color: #f9f9f9; }
    </style>
    """, unsafe_allow_html=True)

# --- ۱. بارگذاری مدل‌ها و داده‌ها ---
@st.cache_resource
def load_all_models():
    # مسیر مدل‌ها - مطمئن شوید پوشه‌ها در کنار فایل اپلیکیشن هستند
    fabert_path = "./final_fabert_model"
    parsbert_path = "./final_parsbert_model"
    
    # بارگذاری FaBERT
    tok_fa = BertTokenizer.from_pretrained(fabert_path)
    mod_fa = BertForSequenceClassification.from_pretrained(fabert_path)
    
    # بارگذاری ParsBERT
    tok_pars = BertTokenizer.from_pretrained(parsbert_path)
    mod_pars = BertForSequenceClassification.from_pretrained(parsbert_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod_fa.to(device).eval()
    mod_pars.to(device).eval()
    
    # بارگذاری دانش
    df_kg = pd.read_csv('graph.csv')
    with open('ontology.json', 'r', encoding='utf-8') as f:
        ontology = json.load(f)
        
    return (tok_fa, mod_fa), (tok_pars, mod_pars), device, df_kg, ontology

try:
    fabert, parsbert, device, df_kg, ONTOLOGY = load_all_models()
    st.sidebar.success("✅ هر دو مدل (FaBERT & ParsBERT) آماده هستند.")
except Exception as e:
    st.error(f"❌ خطا در بارگذاری مدل‌ها: {e}")
    st.stop()

# --- ۲. توابع پردازشی ---
def get_prob(text, model_data):
    tokenizer, model = model_data
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()

import re

def find_impact_word(comment, model_data):
    """
    پیدا کردن کلمه‌ای که بیشترین تأثیر را روی تغییر نظر مدل داشته است (Ablation Study).
    """
    if not comment or not isinstance(comment, str):
        return None

    # ۱. حذف علائم نگارشی برای توکن‌بندی دقیق (مثلاً "خوبه!" بشه "خوبه")
    # فقط حروف و اعداد و نیم‌فاصله فارسی باقی می‌مانند
    clean_comment = re.sub(r'[^\w\s\u200c]', ' ', comment)
    words = clean_comment.split()

    # ۲. لیست کلمات توقف توسعه‌یافته (برای جلوگیری از انتخاب کلمات ربط)
    STOP_WORDS = set([
        'خیلی', 'واقعا', 'بود', 'است', 'شد', 'این', 'آن', 'رو', 'در', 'از', 'به', 
        'با', 'که', 'برای', 'هم', 'یک', 'من', 'ما', 'شما', 'ها', 'های', 'تر', 'ترین',
        'را', 'می', 'نمی', 'دارد', 'کرد', 'نیست', 'هست', 'اما', 'ولی', 'اگر'
    ])

    # محاسبه امتیاز پایه (امتیاز جمله کامل)
    try:
        base_score = get_prob(comment, model_data)
    except:
        return words[0] if words else "Unknown"

    impacts = []

    for i, word in enumerate(words):
        # ۳. فیلتر کردن کلمات: کلمات توقف و کلمات خیلی کوتاه (زیر ۲ حرف)
        if word in STOP_WORDS or len(word) < 2:
            continue

        # ۴. تکنیک حذف کلمه (Leave-One-Out)
        # کلمه جاری را از لیست حذف می‌کنیم تا ببینیم جمله بدون آن چه حسی دارد
        temp_words = words[:i] + words[i+1:]
        temp_text = " ".join(temp_words)

        # اگر جمله خالی شد (مثلاً کامنت تک کلمه‌ای بوده)، ادامه نده
        if not temp_text.strip():
            continue

        try:
            # محاسبه اختلاف امتیاز (قدر مطلق تفاوت)
            new_score = get_prob(temp_text, model_data)
            diff = abs(new_score - base_score)
            impacts.append((word, diff))
        except:
            continue

    # ۵. انتخاب کلمه با بیشترین تأثیر
    # اگر هیچ کلمه‌ای پیدا نشد (مثلاً همش Stop word بود)، اولین کلمه معنادار رو برگردون
    if not impacts:
        return words[0] if words else "N/A"
    
    # مرتب‌سازی نزولی بر اساس میزان اختلاف (diff)
    best_word = sorted(impacts, key=lambda x: x[1], reverse=True)[0][0]
    
    return best_word

def create_enhanced_graph(ontology_data, target_aspect=None):
    # تنظیمات شبکه با فیزیک پیشرفته برای جلوگیری از درهم‌ریختگی
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="#2d3436", directed=True)
    
    # تنظیم فیزیک برای فاصله گرفتن گره‌ها (خیلی مهم برای زیبایی)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4)

    # اگر جنبه خاصی پیدا شده، فقط همان شاخه را نمایش بده
    if target_aspect and target_aspect in ontology_data:
        items = {target_aspect: ontology_data[target_aspect]}
    else:
        items = ontology_data # یا نمایش گراف کلی با فاصله زیاد

    for aspect, details in items.items():
        # ۱. گره مرکزی: جنبه فنی (آبی، بزرگ، لوزی)
        net.add_node(aspect, label=f"💎 {aspect}", color="#0984e3", size=40, shape="diamond", shadow=True)
        
        # ۲. گره‌های صفت (سبز برای مثبت، قرمز برای منفی)
        net.add_node(f"P_{aspect}", label=f"✅ {details['pos_adj']}", color="#00b894", size=25, shape="dot")
        net.add_node(f"N_{aspect}", label=f"❌ {details['neg_adj']}", color="#d63031", size=25, shape="dot")
        
        # اتصال صفت‌ها به مرکز
        net.add_edge(aspect, f"P_{aspect}", label="Positive Path", color="#55efc4", width=2)
        net.add_edge(aspect, f"N_{aspect}", label="Negative Path", color="#ff7675", width=2)
        
        # ۳. گره‌های کلمات کلیدی عامیانه (زرد، دایره کوچک، خط‌چین)
        for kw in details['keywords']:
            kw_id = f"KW_{kw}_{aspect}" # شناسه یکتا
            net.add_node(kw_id, label=kw, color="#fdcb6e", size=15, shape="ellipse", font={'size': 12})
            net.add_edge(kw_id, aspect, label="Mapped To", dashes=True, color="#dfe6e9")

    # ذخیره و تنظیمات نهایی
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {"gravitationalConstant": -100, "springLength": 150},
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    net.save_graph("enhanced_ontology.html")
    return "enhanced_ontology.html"

# --- ۳. رابط کاربری (UI) ---
st.title("⚖️ FaBERT vs ParsBERT: نبرد مدل‌ها در میدان XAI")

product_query = st.selectbox("📌 انتخاب محصول از گراف دانش:", df_kg['Title'].unique())
user_comment = st.text_area("💬 نظر کاربر را وارد کنید:", "گوشیم باتریش خیلی زود تموم میشه")

if st.button("شروع تحلیل مقایسه‌ای 🚀"):
    # استخراج اطلاعات محصول
    product_info = df_kg[df_kg['Title'] == product_query].iloc[0]
    
    # پیدا کردن جنبه فنی
    target_node = next((node for node, data in ONTOLOGY.items() if any(k in user_comment for k in data['keywords'])), None)

    col_fa, col_pars = st.columns(2)

    for i, (name, m_data) in enumerate([("FaBERT (Deep Review Analysis)", fabert), ("ParsBERT (General Persian)", parsbert)]):
        with (col_fa if i==0 else col_pars):
            st.markdown(f"### 🤖 {name}")
            
            # تحلیل اولیه
            prob = get_prob(user_comment, m_data)
            is_pos = prob > 0.5
            st.metric("احساس اولیه", "مثبت 😊" if is_pos else "منفی 😡", f"{prob:.2%}")
            
            # پیدا کردن کلمه کلیدی
            impact_word = find_impact_word(user_comment, m_data)
            st.write(f"🎯 کلمه کلیدی شناسایی شده: **{impact_word}**")
            
            if target_node:
                # منطق جایگزینی
                tech_val = product_info.get(ONTOLOGY[target_node]['tech_column'], "N/A")
                repl = ONTOLOGY[target_node]['neg_adj'] if is_pos else f"{ONTOLOGY[target_node]['pos_adj']} ({tech_val})"
                mod_comment = user_comment.replace(impact_word, repl)
                
                new_prob = get_prob(mod_comment, m_data)
                shift = abs(new_prob - prob)
                
                st.info(f"🏗 اصلاح شد به: \n\n {mod_comment}")
                st.metric("امتیاز پس از اصلاح", f"{new_prob:.2%}", f"تغییر: {shift:.2%}")
            else:
                st.warning("جنبه فنی یافت نشد.")

st.divider()
st.subheader("🌐 تحلیل بصری پیوند معنایی (Local XAI)")

# پیدا کردن جنبه فنی جمله به صورت داینامیک
detected_aspect = None
for node, data in ONTOLOGY.items():
    if any(k in user_comment for k in data['keywords']):
        detected_aspect = node
        break

with st.expander("🔍 باز کردن نقشه دانش استخراج شده", expanded=True):
    if detected_aspect:
        st.success(f"🎯 مدل بر روی جنبه فنی **{detected_aspect}** تمرکز کرده است.")
        path = create_enhanced_graph(ONTOLOGY, target_aspect=detected_aspect)
    else:
        st.warning("💡 کلمه کلیدی فنی یافت نشد. نمایش ساختار کلی آنتولوژی:")
        path = create_enhanced_graph(ONTOLOGY)
    
    with open(path, 'r', encoding='utf-8') as f:
        html(f.read(), height=550)