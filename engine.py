import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
from datetime import datetime

# إعداد Google Generative AI
genai.configure(api_key=os.getenv("pip install -r requirements.txt", "pip install -r requirements.txt"))
model = genai.GenerativeModel('gemini-pro')

# تحميل نموذج فهم المعاني
search_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class SmartWiki:
    def __init__(self, db_file='wiki_data.csv'):
        self.db_file = db_file
        self.load_database()
    
    def load_database(self):
        """تحميل قاعدة البيانات من ملف CSV"""
        try:
            if os.path.exists(self.db_file):
                self.df = pd.read_csv(self.db_file)
            else:
                self.df = pd.DataFrame(columns=['id', 'title', 'content', 'category', 'date', 'author', 'views', 'rating'])
        except Exception as e:
            print(f"خطأ في تحميل قاعدة البيانات: {e}")
            self.df = pd.DataFrame(columns=['id', 'title', 'content', 'category', 'date', 'author', 'views', 'rating'])

    def auto_publish(self, topic, category='عام'):
        """نظام النشر التلقائي بالذكاء الاصطناعي"""
        try:
            prompt = f"اكتب مقالاً موسوعياً مفصلاً عن {topic} بتنسيق ويكيبيديا، بطول 500-800 كلمة، يتضمن المقدمة والأقسام الرئيسية."
            response = model.generate_content(prompt)
            content = response.text
            
            # إنشاء معرّف فريد للمقال
            article_id = len(self.df) + 1
            
            # إضافة المقال لقاعدة البيانات
            new_row = pd.DataFrame({
                'id': [article_id],
                'title': [topic],
                'content': [content],
                'category': [category],
                'date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'author': ['AI'],
                'views': [0],
                'rating': [0.0]
            })
            
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            self.save_database()
            return f"✓ تم نشر مقال جديد عن: {topic}"
        except Exception as e:
            return f"✗ خطأ في النشر: {str(e)}"

    def ai_search(self, query):
        """نظام البحث بالمعنى (فهم المعاني وليس المطابقة الحرفية)"""
        if self.df.empty:
            return "الموسوعة فارغة حالياً."
        
        try:
            # تحويل السؤال والمقالات إلى مصفوفات رقمية
            query_embedding = search_model.encode(query)
            content_embeddings = search_model.encode(self.df['content'].tolist())
            
            # حساب التشابه
            cosine_scores = util.cos_sim(query_embedding, content_embeddings)[0]
            
            # جلب أفضل النتائج
            best_idx = cosine_scores.argmax().item()
            similarity_score = cosine_scores[best_idx].item()
            
            if similarity_score > 0.3:
                # تحديث عدد المشاهدات
                self.df.at[best_idx, 'views'] = self.df.at[best_idx, 'views'] + 1
                self.save_database()
                return self.df.iloc[best_idx].to_dict()
            return "لم أجد نتائج مطابقة للمعنى."
        except Exception as e:
            return f"خطأ في البحث: {str(e)}"

    def get_by_category(self, category):
        """الحصول على المقالات حسب التصنيف"""
        try:
            results = self.df[self.df['category'] == category].to_dict('records')
            return results if results else f"لا توجد مقالات في تصنيف: {category}"
        except Exception as e:
            return f"خطأ: {str(e)}"

    def get_trending(self, limit=5):
        """الحصول على المقالات الأكثر مشاهدة"""
        try:
            trending = self.df.nlargest(limit, 'views')[['title', 'views', 'rating']].to_dict('records')
            return trending if trending else "لا توجد مقالات بعد"
        except Exception as e:
            return f"خطأ: {str(e)}"

    def edit_article(self, article_id, new_content, category=None):
        """تعديل مقال موجود"""
        try:
            if article_id in self.df['id'].values:
                idx = self.df[self.df['id'] == article_id].index[0]
                self.df.at[idx, 'content'] = new_content
                if category:
                    self.df.at[idx, 'category'] = category
                self.save_database()
                return f"✓ تم تعديل المقال رقم {article_id}"
            return f"✗ المقال رقم {article_id} غير موجود"
        except Exception as e:
            return f"خطأ: {str(e)}"

    def delete_article(self, article_id):
        """حذف مقال"""
        try:
            if article_id in self.df['id'].values:
                self.df = self.df[self.df['id'] != article_id]
                self.save_database()
                return f"✓ تم حذف المقال رقم {article_id}"
            return f"✗ المقال رقم {article_id} غير موجود"
        except Exception as e:
            return f"خطأ: {str(e)}"

    def rate_article(self, article_id, rating):
        """تقييم مقال (من 1 إلى 5)"""
        try:
            if 1 <= rating <= 5 and article_id in self.df['id'].values:
                idx = self.df[self.df['id'] == article_id].index[0]
                self.df.at[idx, 'rating'] = rating
                self.save_database()
                return f"✓ تم تقييم المقال بـ {rating}/5"
            return "✗ تقييم غير صحيح (يجب أن يكون من 1 إلى 5)"
        except Exception as e:
            return f"خطأ: {str(e)}"

    def get_all_articles(self):
        """الحصول على جميع المقالات"""
        try:
            return self.df.to_dict('records')
        except Exception as e:
            return f"خطأ: {str(e)}"

    def get_statistics(self):
        """إحصائيات الموسوعة"""
        try:
            stats = {
                'عدد المقالات': len(self.df),
                'عدد التصنيفات': self.df['category'].nunique(),
                'إجمالي المشاهدات': self.df['views'].sum(),
                'متوسط التقييم': self.df['rating'].mean()
            }
            return stats
        except Exception as e:
            return f"خطأ: {str(e)}"

    def save_database(self):
        """حفظ قاعدة البيانات"""
        try:
            self.df.to_csv(self.db_file, index=False, encoding='utf-8')
        except Exception as e:
            print(f"خطأ في حفظ قاعدة البيانات: {e}")

# --- تشغيل المحرك ---
wiki = SmartWiki()

# أمثلة للاستخدام:
# wiki.auto_publish("تاريخ الطيران الحربي", "التاريخ")
# wiki.auto_publish("فيزياء الكم", "العلوم")
# print(wiki.ai_search("كيف تطير الطائرات؟"))
# print(wiki.get_trending())
# print(wiki.get_statistics())