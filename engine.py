import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os

# 1. إعداد ذكاء قوقل (للنشر التلقائي)
# استبدل 'YOUR_API_KEY' بالمفتاح الذي حصلت عليه من Google AI Studio
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-pro')

# 2. تحميل نموذج فهم المعاني (للبحث الذكي)
# هذا النموذج يحول النصوص إلى أرقام ليفهم "المعنى"
search_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class SmartWiki:
    def __init__(self, db_file='wiki_data.csv'):
        self.db_file = db_file
        if os.path.exists(db_file):
            self.df = pd.read_csv(db_file)
        else:
            self.df = pd.DataFrame(columns=['title', 'content', 'category', 'date'])

    def auto_publish(self, topic):
        """نظام النشر التلقائي بالذكاء الاصطناعي"""
        prompt = f"اكتب مقالاً موسوعياً مفصلاً عن {topic} بتنسيق ويكيبيديا، يتضمن التصنيف المناسب."
        response = model.generate_content(prompt)
        content = response.text
        
        # إضافة المقال لقاعدة البيانات تلقائياً
        new_row = {'title': topic, 'content': content, 'category': 'General', 'date': pd.Timestamp.now()}
        self.df = self.df.append(new_row, ignore_index=True)
        self.df.to_csv(self.db_file, index=False)
        return f"تم نشر مقال جديد عن: {topic}"

    def ai_search(self, query):
        """نظام البحث بالمعنى (مش مطابقة حرفية)"""
        if self.df.empty: return "الموسوعة فارغة حالياً."
        
        # تحويل السؤال والمقالات إلى مصفوفات رقمية (Embeddings)
        query_embedding = search_model.encode(query)
        content_embeddings = search_model.encode(self.df['content'].tolist())
        
        # حساب التشابه بين السؤال وكل المقالات
        cosine_scores = util.cos_sim(query_embedding, content_embeddings)[0]
        
        # جلب أفضل نتيجة
        best_result_idx = cosine_scores.argmax().item()
        if cosine_scores[best_result_idx] > 0.4: # عتبة التشابه
            return self.df.iloc[best_result_idx]
        return "لم أجد نتائج مطابقة للمعنى."

# --- تشغيل المحرك ---
wiki = SmartWiki()

# مثال للنشر التلقائي:
# wiki.auto_publish("تاريخ الطيران الحربي")

# مثال للبحث الذكي:
# print(wiki.ai_search("كيف تطير الطائرات؟"))
