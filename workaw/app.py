import os
import time
import json
import re
import google.generativeai as genai
import streamlit as st
from prompt import PROMPT_WORKAW
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import ResourceExhausted
from typing import Dict, List, Any

# Import enhanced document reader
try:
    from document_reader import EnhancedDocumentReader, get_kmutnb_summary, search_in_document
    ENHANCED_READER_AVAILABLE = True
except ImportError:
    ENHANCED_READER_AVAILABLE = False
    # ฟังก์ชันสำรองแบบเดิม
    def get_kmutnb_summary(file_path: str) -> str:
        try:
            if file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path.lower().endswith('.pdf'):
                try:
                    import fitz
                    doc = fitz.open(file_path)
                    content = ""
                    for page in doc:
                        content += page.get_text()
                    doc.close()
                except ImportError:
                    return "Error: PyMuPDF not installed. Please install: pip install PyMuPDF"
            else:
                return "Error: Unsupported file type. Please use .txt or .pdf"
            
            if len(content) > 15000:
                content = content[:15000] + "\n\n[เนื้อหาถูกตัดเพื่อประหยัด token]"
            
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

# Configure API
genai.configure(api_key="AIzaSyDnfUxgwBV4QaoXCo1hPHn4536BtlVAeq4")

# Enhanced generation config
generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=SAFETY_SETTINGS,
    generation_config=generation_config,
    system_instruction=PROMPT_WORKAW,
)

# Enhanced Rate limiting with better tracking
class EnhancedRateLimiter:
    def __init__(self):
        if 'api_calls' not in st.session_state:
            st.session_state.api_calls = []
        if 'api_errors' not in st.session_state:
            st.session_state.api_errors = []
        if 'total_tokens_used' not in st.session_state:
            st.session_state.total_tokens_used = 0
    
    def can_make_request(self):
        current_time = time.time()
        st.session_state.api_calls = [
            call_time for call_time in st.session_state.api_calls 
            if current_time - call_time < 60
        ]
        return len(st.session_state.api_calls) < 10
    
    def add_request(self, tokens_used: int = 0):
        st.session_state.api_calls.append(time.time())
        st.session_state.total_tokens_used += tokens_used
    
    def add_error(self, error_msg: str):
        st.session_state.api_errors.append({
            'time': time.time(),
            'error': error_msg
        })
        if len(st.session_state.api_errors) > 10:
            st.session_state.api_errors = st.session_state.api_errors[-10:]
    
    def time_until_next_request(self):
        if not st.session_state.api_calls:
            return 0
        oldest_call = min(st.session_state.api_calls)
        return max(0, 60 - (time.time() - oldest_call))

    def get_recent_errors(self):
        current_time = time.time()
        return [
            error for error in st.session_state.api_errors
            if current_time - error['time'] < 300
        ]

rate_limiter = EnhancedRateLimiter()

# Document management
class DocumentManager:
    def __init__(self):
        if 'document_content' not in st.session_state:
            st.session_state.document_content = None
        if 'document_metadata' not in st.session_state:
            st.session_state.document_metadata = {}
        if 'document_sections' not in st.session_state:
            st.session_state.document_sections = {}
        if 'document_keywords' not in st.session_state:
            st.session_state.document_keywords = []
        if 'last_file_path' not in st.session_state:
            st.session_state.last_file_path = None
    
    def load_document(self, file_path: str) -> tuple[str, str]:
        if st.session_state.last_file_path == file_path and st.session_state.document_content:
            return st.session_state.document_content, "✅ ใช้เอกสารที่โหลดไว้แล้ว"
        
        if not os.path.exists(file_path):
            search_paths = self._get_search_paths(file_path)
            
            for path in search_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            else:
                return None, f"ไม่พบไฟล์เอกสาร: {file_path}"
        
        try:
            if ENHANCED_READER_AVAILABLE:
                reader = EnhancedDocumentReader(file_path)
                content = reader.get_comprehensive_summary()
                
                st.session_state.document_metadata = reader.metadata
                st.session_state.document_sections = reader.sections
                st.session_state.document_keywords = list(reader.keywords)
                
                status = f"✅ โหลดเอกสารสำเร็จ (Enhanced Mode) - {len(content):,} ตัวอักษร"
                if reader.metadata:
                    status += f" | หน้า: {reader.metadata.get('pages', 'ไม่ทราบ')}"
            else:
                content = get_kmutnb_summary(file_path)
                status = f"✅ โหลดเอกสารสำเร็จ (Basic Mode) - {len(content):,} ตัวอักษร"
            
            if content.startswith("Error:"):
                return None, content
            
            st.session_state.document_content = content
            st.session_state.last_file_path = file_path
            
            return content, status
            
        except Exception as e:
            error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
            return None, error_msg
    
    def _get_search_paths(self, original_path: str) -> List[str]:
        current_dir = os.path.dirname(__file__) if __file__ else os.getcwd()
        filename = os.path.basename(original_path)
        
        search_paths = [
            original_path,
            os.path.join(current_dir, filename),
            os.path.join(current_dir, "dataset_reseach.pdf"),
            os.path.join(current_dir, "dataset.pdf"),
            os.path.join(current_dir, "data.pdf"),
            os.path.join(current_dir, "kmutnb.pdf"),
            os.path.join(current_dir, "kmutnb_data.pdf"),
            os.path.join(current_dir, "documents", filename),
            os.path.join(current_dir, "data", filename),
        ]
        
        return search_paths
    
    def search_document(self, search_term: str) -> str:
        if not st.session_state.document_content:
            return "❌ ไม่มีเอกสารที่โหลดไว้"
        
        if ENHANCED_READER_AVAILABLE and st.session_state.last_file_path:
            return search_in_document(st.session_state.last_file_path, search_term)
        else:
            content = st.session_state.document_content
            lines = content.split('\n')
            found_lines = []
            
            for i, line in enumerate(lines):
                if search_term.lower() in line.lower():
                    context_start = max(0, i-1)
                    context_end = min(len(lines), i+2)
                    context = lines[context_start:context_end]
                    found_lines.append(f"=== บรรทัดที่ {i+1} ===\n" + '\n'.join(context) + "\n")
            
            if found_lines:
                return f"🔍 พบคำว่า '{search_term}' ในเอกสาร {len(found_lines)} ตำแหน่ง:\n\n" + '\n'.join(found_lines[:5])
            else:
                return f"❌ ไม่พบคำว่า '{search_term}' ในเอกสาร"

doc_manager = DocumentManager()

def clear_history():
    st.session_state["messages"] = [
        {"role": "model", "content": "KMUTNB Chatbot สวัสดีค่ะ คุณลูกค้า สอบถามข้อมูลเกี่ยวกับ KMUTNB เรื่องใดคะ"}
    ]
    st.rerun()

def safe_api_call(api_function, max_retries=3):
    for attempt in range(max_retries):
        try:
            if not rate_limiter.can_make_request():
                wait_time = rate_limiter.time_until_next_request()
                if wait_time > 0:
                    st.warning(f"⏳ รอ {wait_time:.0f} วินาที เนื่องจาก rate limit")
                    time.sleep(wait_time + 1)
            
            result = api_function()
            rate_limiter.add_request(tokens_used=100)
            return result
            
        except ResourceExhausted as e:
            error_msg = f"API quota เกิน (ครั้งที่ {attempt + 1})"
            rate_limiter.add_error(error_msg)
            
            wait_time = 60 * (attempt + 1)
            if attempt < max_retries - 1:
                st.warning(f"⚠️ {error_msg}! รอ {wait_time} วินาที...")
                time.sleep(wait_time)
            else:
                return "ขออภัยค่ะ ระบบกำลังยุ่ง กรุณาลองใหม่ในอีกสักครู่นะคะ 🙏"
                
        except Exception as e:
            error_msg = f"API Error: {str(e)}"
            rate_limiter.add_error(error_msg)
            
            if attempt < max_retries - 1:
                st.warning(f"⚠️ {error_msg} (ลองใหม่ครั้งที่ {attempt + 2})")
                time.sleep(5 * (attempt + 1))
            else:
                return f"เกิดข้อผิดพลาด: {str(e)}"
    
    return "ไม่สามารถประมวลผลได้ในขณะนี้"

def enhanced_response_generation(prompt: str, document_content: str) -> str:
    question_type = analyze_question_type(prompt)
    enhanced_prompt = enhance_prompt_based_on_type(prompt, question_type)
    
    def generate_response():
        history = []
        
        history.append({
            "role": "user", 
            "parts": [{"text": f"เอกสารอ้างอิง:\n{document_content}"}]
        })
        
        recent_messages = st.session_state["messages"][-10:]
        for msg in recent_messages:
            history.append({
                "role": msg["role"], 
                "parts": [{"text": msg["content"]}]
            })
        
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(enhanced_prompt)
        return response.text
    
    return safe_api_call(generate_response)

def analyze_question_type(prompt: str) -> str:
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['ค้นหา', 'หา', 'search', 'find']):
        return 'search'
    elif any(word in prompt_lower for word in ['เปรียบเทียบ', 'compare', 'ต่าง', 'เหมือน']):
        return 'compare'
    elif any(word in prompt_lower for word in ['อธิบาย', 'explain', 'คืออะไร', 'ทำไม', 'อย่างไร']):
        return 'explain'
    elif any(word in prompt_lower for word in ['รายชื่อ', 'list', 'มีอะไรบ้าง', 'ทั้งหมด']):
        return 'list'
    elif any(word in prompt_lower for word in ['ตัวอย่าง', 'example', 'เช่น']):
        return 'example'
    else:
        return 'general'

def enhance_prompt_based_on_type(prompt: str, question_type: str) -> str:
    enhancements = {
        'search': "กรุณาค้นหาข้อมูลที่เกี่ยวข้องในเอกสารและตอบอย่างละเอียด: ",
        'compare': "กรุณาเปรียบเทียบและวิเคราะห์ความแตกต่างอย่างชัดเจน: ",
        'explain': "กรุณาอธิบายอย่างละเอียดและให้ตัวอย่างประกอบ: ",
        'list': "กรุณาจัดทำรายการที่ครบถ้วนและเรียงลำดับ: ",
        'example': "กรุณาให้ตัวอย่างที่ชัดเจนและหลากหลาย: ",
        'general': "กรุณาตอบคำถามอย่างละเอียดและครบถ้วน: "
    }
    
    return enhancements.get(question_type, '') + prompt

# Page config
st.set_page_config(
    page_title="KMUTNB Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with enhanced controls
with st.sidebar:
    
    
    if st.button("Clear History", use_container_width=True):
        clear_history()

    
    requests_count = len([
        call for call in st.session_state.get('api_calls', [])
        if time.time() - call < 60
    ])
   
    
   

# Main app
st.title("💬 KMUTNB Enhanced Chatbot")
st.write("ระบบ AI ตอบคำถามเกี่ยวกับ KMUTNB อย่างละเอียดและแม่นยำ")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "model",
            "content": "KMUTNB Chatbot สวัสดีค่ะ คุณลูกค้า สอบถามข้อมูลเกี่ยวกับ KMUTNB เรื่องใดคะ ระบบได้รับการปรับปรุงให้อ่านเอกสารได้ละเอียดและตอบคำถามได้แม่นยำมากขึ้นแล้วค่ะ",
        }
    ]

# Load document
file_path = "/Users/zayxaxto/Documents/kmutnb_chatbot/workaw/dataset_reseach.pdf"
file_content, load_status = doc_manager.load_document(file_path)

# Display load status
if file_content is None:
    st.error(f"❌ {load_status}")
    st.info("💡 วิธีแก้ไข: ตรวจสอบ path ของไฟล์หรือวางไฟล์ในโฟลเดอร์เดียวกันกับ app.py")
    
    with st.expander("📁 ตำแหน่งที่ระบบจะค้นหาไฟล์"):
        search_paths = doc_manager._get_search_paths(file_path)
        for path in search_paths:
            status = "✅" if os.path.exists(path) else "❌"
            st.markdown(f"**{status}** `{path}`")
else:
    st.success(f"✅ {load_status}")

# Display messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("💭 Type your question"):
    if file_content is None:
        st.error("❌ กรุณาโหลดเอกสารก่อนใช้งาน")
        st.stop()
    
    if not rate_limiter.can_make_request():
        wait_time = rate_limiter.time_until_next_request()
        st.error(f"⏳ กรุณารอ {wait_time:.0f} วินาที ก่อนส่งข้อความใหม่")
        st.stop()
    
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("model"):
        with st.spinner("🤔 กำลังประมวลผลอย่างละเอียด..."):
            
            if prompt.lower().startswith("ค้นหา:") or prompt.lower().startswith("search:"):
                search_term = prompt.split(":", 1)[1].strip()
                response_text = doc_manager.search_document(search_term)
            else:
                response_text = enhanced_response_generation(prompt, file_content)
            
            st.write(response_text)
            st.session_state["messages"].append({"role": "model", "content": response_text})

