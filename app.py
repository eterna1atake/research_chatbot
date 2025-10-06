import os
import time
import json
import re
import google.generativeai as genai
import streamlit as st
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import hashlib

# Configure API
genai.configure(api_key="AIzaSyCEJLy1bwbtXnjtdC6i2clawIEAgEcjwaw")

# Generation config
generation_config = {
    "temperature": 0.15,
    "top_p": 0.92,
    "top_k": 45,
    "max_output_tokens": 12000,
    "response_mime_type": "text/plain",
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

# Enhanced keyword mapping
KEYWORDS = {
    "รับสมัคร": ["สมัคร", "ลงทะเบียน", "apply", "admission", "enrollment", "เปิดรับ", "การรับ", "สมัครเข้า"],
    "เอกสาร": ["หลักฐาน", "documents", "ใบสำคัญ", "ใบรับรอง", "ประกอบ", "เอกสารประกอบ", "หลักฐานการสมัคร"],
    "ค่าเทอม": ["ค่าธรรมเนียม", "ค่าใช้จ่าย", "tuition", "fee", "ค่าเล่าเรียน", "ค่าลงทะเบียน", "ค่าใช้จ่ายการศึกษา"],
    "ปริญญาตรี": ["ป.ตรี", "bachelor", "undergraduate", "บัณฑิต", "ระดับปริญญาตรี"],
    "ปริญญาโท": ["ป.โท", "master", "graduate", "มหาบัณฑิต", "ระดับปริญญาโท"],
    "ปริญญาเอก": ["ป.เอก", "PhD", "doctorate", "ดุษฎีบัณฑิต", "ระดับปริญญาเอก"],
    "สาขา": ["สาขาวิชา", "แผนก", "program", "major", "หลักสูตร", "วิชาเอก", "สาขาวิชาเอก"],
    "วิศวกรรม": ["วิศวะ", "engineer", "engineering", "คณะวิศวกรรมศาสตร์"],
    "เทคโนโลยี": ["เทค", "technology", "เทคโนโลยีสารสนเทศ", "เทคโนโลยีอุตสาหกรรม"],
    "ครุศาสตร์": ["ครุ", "ศึกษาศาสตร์", "การศึกษา", "education", "คณะครุศาสตร์อุตสาหกรรม"],
    "ไฟฟ้า": ["electrical", "อิเล็กทรอนิกส์", "electronics", "วิศวกรรมไฟฟ้า"],
    "คอมพิวเตอร์": ["computer", "คอม", "สารสนเทศ", "IT", "วิศวกรรมคอมพิวเตอร์"],
    "เครื่องกล": ["mechanical", "กล", "วิศวกรรมเครื่องกล"],
    "โยธา": ["civil", "โครงสร้าง", "วิศวกรรมโยธา"],
    "อุตสาหการ": ["industrial", "อุตสาหกรรม", "วิศวกรรมอุตสาหการ"],
    "สอบ": ["การสอบ", "exam", "test", "ทดสอบ", "การทดสอบ"],
    "เกรด": ["เกรดเฉลี่ย", "GPA", "ผลการเรียน", "คะแนน", "เกรดขั้นต่ำ"],
    "ทุน": ["ทุนการศึกษา", "scholarship", "ทุนเรียนดี"],
    "ต่อ": ["เรียนต่อ", "ศึกษาต่อ", "continue", "การเรียนต่อ", "ต่อได้", "สมัครได้"],
    "คุณสมบัติ": ["เงื่อนไข", "qualification", "requirement", "ข้อกำหนด"],
    "ระยะเวลา": ["กำหนดเวลา", "duration", "ช่วงเวลา", "วันที่"],
    "วิธีการ": ["ขั้นตอน", "procedure", "กระบวนการ", "method"],
    "คณะ": ["faculty", "สำนัก", "วิทยาลัย"],
    "สาย": ["แผนการเรียน", "สายการเรียน", "แผน"]
}

# System Prompt ที่ฉลาดและเข้าใจบริบท
SYSTEM_PROMPT = """คุณเป็น AI ที่ฉลาดในการตอบคำถามเกี่ยวกับ KMUTNB

หลักการตอบที่สำคัญ:

**1. เข้าใจระดับความละเอียดที่ต้องการ:**

ถ้าถามแบบกว้างๆ → ตอบแบบกว้างๆ
- "ม.6 สายศิลป์ต่ออะไรได้บ้าง?" → ตอบเฉพาะชื่อคณะ
- "มีคณะอะไรบ้าง?" → ตอบเฉพาะชื่อคณะ
- "มีกี่สาขา?" → ตอบจำนวน + ชื่อสาขา

ถ้าถามแบบละเอียด → ตอบแบบละเอียด
- "สาขาคอมพิวเตอร์มีอะไรบ้าง?" → แจกแจงทุกหลักสูตร
- "คุณสมบัติการสมัครคืออะไร?" → ระบุคุณสมบัติทุกข้อ
- "ขั้นตอนการสมัครยังไง?" → แจกแจงทุกขั้นตอน

**2. รูปแบบการตอบตามประเภทคำถาม:**

**คำถามเกี่ยวกับคณะที่รับสมัคร:**
ตอบแค่ชื่อคณะ ไม่ต้องแจกแจงหลักสูตร
ตัวอย่าง: "ม.6 สายศิลป์ต่อได้ที่คณะครุศาสตร์อุตสาหกรรม"

**คำถามเกี่ยวกับจำนวนคณะ/สาขา:**
ตอบจำนวน + ชื่อคณะ/สาขา (ไม่ต้องบอกหลักสูตรย่อย)

**คำถามเกี่ยวกับหลักสูตรในคณะ:**
แจกแจงหลักสูตรทุกหลักสูตรพร้อมจัดกลุ่ม

**คำถามเกี่ยวกับคุณสมบัติ:**
แจกแจงคุณสมบัติทุกข้อ + ระบุตัวเลขชัดเจน

**คำถามเกี่ยวกับเอกสาร:**
ระบุเอกสารทุกรายการ

**คำถามเกี่ยวกับค่าใช้จ่าย:**
ระบุจำนวนเงิน + แยกรายการ

**3. การจัดรูปแบบ:**
- ถ้าตอบสั้นๆ → ใช้ประโยค ไม่ต้องใช้ bullet points
- ถ้าตอบหลายข้อ → ใช้ bullet points (•)
- แบ่งหมวดหมู่ชัดเจน
- เว้นบรรทัดเหมาะสม

**4. ข้อห้าม:**
- ห้ามใช้: ครับ ค่ะ นะ
- ห้ามอ้างอิง: หน้า ข้อ
- ห้ามเสริมข้อมูลที่ไม่เกี่ยวข้อง
- ห้ามตอบยาวเกินจำเป็น

**5. ตัวอย่างการตอบที่ดี:**

คำถาม: "จบ ม.6 สายศิลป์ต่ออะไรได้บ้าง?"
ตอบ: "สายศิลป์สามารถสมัครเข้าศึกษาต่อได้ที่คณะครุศาสตร์อุตสาหกรรม"

คำถาม: "คณะครุศาสตร์มีกี่สาขา?"
ตอบ: "คณะครุศาสตร์อุตสาหกรรมมี 22 หลักสูตร แบ่งเป็น:

**ปริญญาตรี: 7 หลักสูตร**
- หลักสูตรครุศาสตร์อุตสาหกรรมบัณฑิต (ค.อ.บ.) 4 ปี
- วิศวกรรมแมคคาทรอนิกส์และระบบอัตโนมัติ (TT)
...

**ปริญญาโท: 7 หลักสูตร**
...

**ปริญญาเอก: 8 หลักสูตร**
..."

คำถาม: "มีคณะอะไรบ้าง?"
ตอบ: "KMUTNB มีหลายคณะ อาทิ:
- คณะวิศวกรรมศาสตร์
- คณะครุศาสตร์อุตสาหกรรม
- คณะเทคโนโลยี
..."

สิ่งสำคัญ: **ตอบตรงคำถาม ไม่ยาวเกินจำเป็น และเข้าใจบริบท**
"""

# Initialize model with chat history support
def create_chat_session():
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        safety_settings=SAFETY_SETTINGS,
        generation_config=generation_config,
        system_instruction=SYSTEM_PROMPT,
    )
    return model.start_chat(history=[])

class DocumentProcessor:
    def __init__(self):
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_file_hash(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def load_from_cache(self, file_hash: str) -> str:
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def save_to_cache(self, file_hash: str, content: str):
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.txt")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def read_pdf_optimized(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return "Error: File not found"
        
        file_hash = self.get_file_hash(file_path)
        if file_hash:
            cached_content = self.load_from_cache(file_hash)
            if cached_content:
                return cached_content
        
        try:
            import fitz
            doc = fitz.open(file_path)
            
            full_text = ""
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    page_text = page.get_text("text")
                    
                    if page_text.strip():
                        page_text = self.clean_text(page_text)
                        full_text += f"\n{page_text}\n"
                        
                except Exception:
                    continue
            
            doc.close()
            
            if file_hash and full_text:
                self.save_to_cache(file_hash, full_text)
            
            return full_text
            
        except ImportError:
            return "Error: ต้องติดตั้ง PyMuPDF (pip install PyMuPDF)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\(หน้า\s*\d+[^)]*\)', '', text)
        text = re.sub(r'\(ข้อ\s*[\d.]+\)', '', text)
        text = re.sub(r'หน้า\s*\d+[^,\n]*', '', text)
        text = re.sub(r'ข้อ\s*[\d.]+[^,\n]*', '', text)
        text = re.sub(r'=== หน้า \d+ ===', '', text)
        text = re.sub(r'\x00', '', text)
        text = re.sub(r'\f', '\n', text)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()

class SmartSearcher:
    def __init__(self):
        self.max_chunk_size = 200000
    
    def analyze_query_type(self, query: str) -> dict:
        """วิเคราะห์ประเภทคำถามเพื่อกำหนดระดับความละเอียดในการตอบ"""
        query_lower = query.lower()
        
        analysis = {
            "detail_level": "medium",
            "question_type": "general",
            "needs_list": False
        }
        
        broad_patterns = [
            r'ต่ออะไรได้',
            r'มีคณะ(อะไร)?บ้าง',
            r'รับสาย(อะไร)?',
            r'สมัครได้(ที่)?ไหน'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in broad_patterns):
            analysis["detail_level"] = "low"
            analysis["question_type"] = "faculty_list"
            return analysis
        
        count_patterns = [
            r'(มี)?กี่(คณะ|สาขา|หลักสูตร)',
            r'จำนวน(คณะ|สาขา|หลักสูตร)',
            r'ทั้งหมดกี่'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in count_patterns):
            analysis["detail_level"] = "medium"
            analysis["question_type"] = "count"
            analysis["needs_list"] = True
            return analysis
        
        detailed_patterns = [
            r'(มี|เปิด)(สาขา|หลักสูตร)(อะไร|ไหน)?บ้าง',
            r'คุณสมบัติ',
            r'เอกสาร',
            r'ค่าใช้จ่าย',
            r'ค่าธรรมเนียม',
            r'ขั้นตอน',
            r'วิธีการ'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in detailed_patterns):
            analysis["detail_level"] = "high"
            analysis["question_type"] = "detailed"
            analysis["needs_list"] = True
        
        return analysis
    
    def expand_query(self, query: str) -> list:
        expanded = [query.lower()]
        query_lower = query.lower()
        
        for keyword, synonyms in KEYWORDS.items():
            if keyword in query_lower:
                expanded.extend([s.lower() for s in synonyms])
            for syn in synonyms:
                if syn.lower() in query_lower:
                    expanded.append(keyword.lower())
                    expanded.extend([s.lower() for s in synonyms])
        
        words = query_lower.split()
        for word in words:
            if len(word) > 2:
                expanded.append(word)
        
        return list(set(expanded))
    
    def find_relevant_chunks(self, content: str, query: str, max_chunks: int = 20) -> list:
        expanded_keywords = self.expand_query(query)
        paragraphs = re.split(r'\n\s*\n', content)
        scored_chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip() or len(paragraph.strip()) < 20:
                continue
                
            paragraph_lower = paragraph.lower()
            score = 0
            matched_keywords = set()
            
            for keyword in expanded_keywords:
                keyword_lower = keyword.lower()
                exact_matches = len(re.findall(rf'\b{re.escape(keyword_lower)}\b', paragraph_lower))
                if exact_matches > 0:
                    score += exact_matches * 15
                    matched_keywords.add(keyword_lower)
                
                partial_matches = paragraph_lower.count(keyword_lower) - exact_matches
                score += partial_matches * 5
                
                if keyword_lower in query.lower():
                    score += exact_matches * 10
            
            list_indicators = ["1.", "2.", "3.", "•", "-", "ก.", "ข.", "ค.", "ปริญญา", "หลักสูตร", "สาขา", "คณะ"]
            for indicator in list_indicators:
                if indicator in paragraph:
                    score += 5
            
            context_keywords = [
                "หลักสูตร", "สาขา", "คุณสมบัติ", "การรับสมัคร", "เงื่อนไข", 
                "ค่าใช้จ่าย", "ระยะเวลา", "ขั้นตอน", "วิธีการ", "เอกสาร",
                "ค่าธรรมเนียม", "ทุนการศึกษา", "การสอบ", "เกรด", "คณะ"
            ]
            context_bonus = sum(5 for ctx_word in context_keywords if ctx_word in paragraph_lower)
            score += context_bonus
            
            if len(paragraph) > 200:
                score += 3
            
            score += len(matched_keywords) * 5
            
            if score > 0:
                extended_chunk = ""
                start_idx = max(0, i-1)
                end_idx = min(len(paragraphs), i+2)
                
                for j in range(start_idx, end_idx):
                    if paragraphs[j].strip():
                        extended_chunk += paragraphs[j].strip() + "\n\n"
                
                scored_chunks.append((score, i, extended_chunk.strip()))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        selected_chunks = []
        total_length = 0
        seen_content = set()
        
        for score, para_num, chunk in scored_chunks[:max_chunks]:
            chunk_hash = hash(chunk[:100])
            if chunk_hash in seen_content:
                continue
            seen_content.add(chunk_hash)
            
            if total_length + len(chunk) > self.max_chunk_size:
                remaining_space = self.max_chunk_size - total_length
                if remaining_space > 500:
                    chunk = chunk[:remaining_space] + "..."
                    selected_chunks.append(chunk)
                break
                
            selected_chunks.append(chunk)
            total_length += len(chunk)
        
        return selected_chunks
    
    def search_and_answer(self, query: str, content: str, chat_session) -> str:
        if not content or content.startswith("Error:"):
            return "ไม่สามารถอ่านข้อมูลได้"
        
        query_analysis = self.analyze_query_type(query)
        relevant_chunks = self.find_relevant_chunks(content, query)
        
        if not relevant_chunks:
            return "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถาม"
        
        combined_content = "\n\n=== ข้อมูลส่วนที่เกี่ยวข้อง ===\n\n".join(relevant_chunks)
        
        if query_analysis["detail_level"] == "low":
            instruction = """
ตอบแบบสั้นและตรงประเด็น:
- ถ้าถามว่า "ต่ออะไรได้บ้าง" → ตอบเฉพาะชื่อคณะที่รับสมัคร (ไม่ต้องแจกแจงหลักสูตร)
- ถ้าถามว่า "มีคณะอะไรบ้าง" → ตอบเฉพาะชื่อคณะ
- ใช้ประโยคสั้นๆ หรือ bullet points ง่ายๆ
- ไม่ต้องระบุรายละเอียดหลักสูตรย่อย
"""
        elif query_analysis["detail_level"] == "medium":
            instruction = """
ตอบแบบปานกลาง:
- บอกจำนวนรวม (ถ้ามี)
- แจกแจงชื่อคณะ/สาขา/หลักสูตร
- จัดกลุ่มตามระดับการศึกษา (ถ้ามี)
- ใช้ bullet points
"""
        else:
            instruction = """
ตอบแบบละเอียด:
- แจกแจงทุกรายการที่เกี่ยวข้อง
- จัดกลุ่มและจัดหมวดหมู่ชัดเจน
- ระบุตัวเลข ข้อมูลที่สำคัญ
- ใช้ bullet points และเว้นบรรทัด
"""
        
        prompt = f"""
คำถาม: {query}

ข้อมูลจาก KMUTNB:
{combined_content}

คำสั่งการตอบ:
{instruction}

ข้อห้าม:
- ห้ามใช้: ครับ ค่ะ นะ
- ห้ามอ้างอิง: หน้า ข้อ เอกสาร
- ห้ามเสริมข้อมูลที่ไม่เกี่ยวข้อง
- ตอบเฉพาะที่คำถามถาม ไม่ต้องขยายความเกินจำเป็น

ตอบตามข้อมูลที่มี:
"""
        
        try:
            response = chat_session.send_message(prompt)
            return self.clean_response(response.text)
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"
    
    def clean_response(self, response: str) -> str:
        if not response:
            return "ไม่พบข้อมูลที่เกี่ยวข้อง"
        
        response = re.sub(r'\(หน้า\s*\d+[^)]*\)', '', response)
        response = re.sub(r'\(ข้อ\s*[\d.]+\)', '', response)
        response = re.sub(r'หน้า\s*\d+[^,\s]*', '', response)
        response = re.sub(r'ข้อ\s*[\d.]+[^,\s]*', '', response)
        
        unwanted_patterns = [
            r'จากเอกสาร[^.]*\.?',
            r'ตามเอกสาร[^.]*\.?',
            r'เอกสารระบุ[^.]*\.?',
            r'ระบุไว้ใน[^.]*\.?',
            r'\bครับ\b',
            r'\bค่ะ\b',
            r'\bนะ\b'
        ]
        
        for pattern in unwanted_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not any(line.startswith(word) for word in ['อ้างอิง', 'ดูจาก', 'ตามที่', 'จากข้อมูล', 'ตามข้อมูล']):
                line = re.sub(r'\s*\([^)]*หน้า[^)]*\)$', '', line)
                line = re.sub(r'\s*\([^)]*ข้อ[^)]*\)$', '', line)
                if line:
                    cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines)
        response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
        response = re.sub(r' +', ' ', response)
        
        return response.strip()

class RateLimiter:
    def __init__(self):
        if 'api_calls' not in st.session_state:
            st.session_state.api_calls = []
    
    def can_make_request(self) -> bool:
        current_time = time.time()
        st.session_state.api_calls = [
            call_time for call_time in st.session_state.api_calls 
            if current_time - call_time < 60
        ]
        return len(st.session_state.api_calls) < 20
    
    def add_request(self):
        st.session_state.api_calls.append(time.time())
    
    def get_wait_time(self) -> int:
        if not st.session_state.api_calls:
            return 0
        current_time = time.time()
        oldest_call = min(st.session_state.api_calls)
        return max(0, int(60 - (current_time - oldest_call)))

# Initialize components
doc_processor = DocumentProcessor()
searcher = SmartSearcher()
rate_limiter = RateLimiter()

def new_chat():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "สอบถามข้อมูลเกี่ยวกับมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ"}
    ]
    st.session_state["chat_session"] = create_chat_session()
    st.rerun()

# Page config
st.set_page_config(
    page_title="KMUTNB Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    
    # ไม่ต้องใช้ text_input เลย แค่กำหนดค่าตรงๆ
    file_path = "Dataset.pdf"

    if st.button("🔄 โหลดใหม่", use_container_width=True):
        if 'document_content' in st.session_state:
            del st.session_state.document_content
        st.rerun()
    if st.button("💬 New Chat", use_container_width=True):
        new_chat()
    
    if 'document_content' in st.session_state:
        content = st.session_state.document_content
        if content and not content.startswith("Error:"):
            paragraphs = len([p for p in re.split(r'\n\s*\n', content) if p.strip()])


# Main app
st.title("🎓 KMUTNB Chatbot")
st.caption("ระบบค้นหาข้อมูลอัจฉริยะสำหรับ KMUTNB")

# Load document
if 'document_content' not in st.session_state:
    if file_path.strip():
        with st.spinner("กำลังอ่านไฟล์ PDF..."):
            content = doc_processor.read_pdf_optimized(file_path)
            st.session_state.document_content = content
            
            if content.startswith("Error:"):
                st.error(f"❌ {content}")
            else:
                paragraphs = len([p for p in re.split(r'\n\s*\n', content) if p.strip()])

# Initialize chat session
if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = create_chat_session()

# Initialize messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "สอบถามข้อมูลเกี่ยวกับมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ"}
    ]

# Display messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("พิมพ์คำถาม..."):
    if 'document_content' not in st.session_state:
        st.error("❌ กรุณาโหลดไฟล์ก่อน")
        st.stop()
    
    if not rate_limiter.can_make_request():
        wait_time = rate_limiter.get_wait_time()
        st.error(f"❌ กรุณารอ {wait_time} วินาที")
        st.stop()
    
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูล..."):
            response = searcher.search_and_answer(
                prompt, 
                st.session_state.document_content,
                st.session_state["chat_session"]
            )
            rate_limiter.add_request()
            
            st.write(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})