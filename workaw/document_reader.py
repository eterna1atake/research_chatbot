import os
import re
from typing import Optional, List, Dict, Tuple
from docx import Document
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from heapq import nlargest
from collections import Counter
import hashlib
import json

# ดาวน์โหลด NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class EnhancedDocumentReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content_cache = {}
        self.metadata = {}
        self.sections = {}
        self.keywords = set()
        
    def validate_file(self) -> bool:
        """ตรวจสอบไฟล์อย่างละเอียด"""
        if not os.path.exists(self.file_path):
            print(f"Error: File {self.file_path} not found!")
            return False

        # เช็คขนาดไฟล์
        try:
            file_size = os.path.getsize(self.file_path)
            if file_size == 0:
                print(f"Error: File {self.file_path} is empty!")
                return False
            elif file_size > 100 * 1024 * 1024:  # 100MB
                print(f"Warning: File {self.file_path} is very large ({file_size/(1024*1024):.1f}MB)")
        except:
            pass

        supported_extensions = ['.docx', '.pdf', '.txt', '.doc']
        if not any(self.file_path.lower().endswith(ext) for ext in supported_extensions):
            print(f"Error: Unsupported file type {self.file_path}!")
            return False

        return True

    def extract_keywords(self, text: str, num_keywords: int = 50) -> List[str]:
        """สกัดคำสำคัญจากข้อความ"""
        try:
            # ทำความสะอาดข้อความ
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = word_tokenize(text)
            
            # กรองคำที่ไม่ต้องการ
            try:
                stop_words = set(stopwords.words('english'))
                stop_words.update(['และ', 'หรือ', 'ที่', 'เป็น', 'ใน', 'จาก', 'ของ', 'การ', 'ได้', 'มี', 'ให้'])
            except:
                stop_words = set()
                
            # กรองคำที่มีความยาวเหมาะสม
            filtered_words = [
                word for word in words 
                if len(word) > 2 and word not in stop_words and not word.isdigit()
            ]
            
            # นับความถี่และเลือกคำสำคัญ
            word_freq = Counter(filtered_words)
            keywords = [word for word, freq in word_freq.most_common(num_keywords)]
            
            self.keywords.update(keywords)
            return keywords
            
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

    def segment_text(self, text: str) -> Dict[str, str]:
        """แบ่งข้อความเป็นส่วนๆ ตามหัวข้อ"""
        sections = {}
        current_section = "introduction"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # ตรวจหาหัวข้อ (บรรทัดที่เป็นตัวพิมพ์ใหญ่หรือมีรูปแบบพิเศษ)
            if (len(line) < 100 and 
                (line.isupper() or 
                 re.match(r'^[\d\.\s]*[A-Za-zก-๙]+', line) or
                 line.startswith('บท') or line.startswith('หมวด') or
                 line.startswith('ส่วนที่') or line.startswith('Chapter'))):
                
                # บันทึกส่วนที่แล้ว
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # เริ่มส่วนใหม่
                current_section = line.lower().replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)
        
        # บันทึกส่วนสุดท้าย
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        self.sections = sections
        return sections

    def enhanced_clean_text(self, text: str) -> str:
        """ทำความสะอาดข้อความอย่างละเอียด"""
        if not text:
            return ""
        
        # ลบอักขระพิเศษที่ไม่ต้องการ
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # ปรับปรุงการขึ้นบรรทัดใหม่
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # ลดบรรทัดว่างซ้ำ
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # กรองบรรทัดที่ไม่มีประโยชน์
            if (len(line) < 3 or 
                line.isdigit() or 
                re.match(r'^[\s\-_=]+$', line) or
                line.count('.') > len(line) * 0.3):  # บรรทัดที่มีจุดเยอะเกินไป
                continue
                
            # ปรับปรุงช่องว่าง
            line = re.sub(r'\s+', ' ', line)
            
            # รวมบรรทัดที่ขาดๆ หายๆ
            if (len(cleaned_lines) > 0 and 
                not line.endswith(('.', '!', '?', ':', ';')) and
                len(line) < 80 and
                not line[0].isupper()):
                cleaned_lines[-1] += ' ' + line
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def read_txt_file(self) -> str:
        """อ่านไฟล์ txt with multiple encodings"""
        encodings = ['utf-8', 'utf-8-sig', 'cp874', 'tis-620', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(self.file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    return self.enhanced_clean_text(content)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with encoding {encoding}: {e}")
                continue
                
        return "Error: Cannot read text file with any supported encoding"

    def read_docx_advanced(self) -> str:
        """อ่าน DOCX อย่างละเอียด"""
        try:
            doc = Document(self.file_path)
            content = []
            
            # อ่าน paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    # เช็ค style ของ paragraph
                    style_name = para.style.name if para.style else ""
                    text = para.text.strip()
                    
                    # เพิ่ม marker สำหรับหัวข้อ
                    if 'heading' in style_name.lower() or para.style.font.bold:
                        text = f"\n=== {text} ===\n"
                    
                    content.append(text)
            
            # อ่าน tables อย่างละเอียด
            for i, table in enumerate(doc.tables, 1):
                content.append(f"\n--- ตาราง {i} ---")
                
                for row_idx, row in enumerate(table.rows):
                    row_data = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_data.append(cell_text)
                    
                    if row_data:  # มีข้อมูลในแถว
                        if row_idx == 0:  # แถวแรกอาจเป็นหัวตาราง
                            content.append("หัวตาราง: " + " | ".join(row_data))
                        else:
                            content.append(" | ".join(row_data))
                
                content.append(f"--- จบตาราง {i} ---\n")
            
            # อ่าน headers และ footers
            for section in doc.sections:
                if section.header:
                    header_text = ""
                    for para in section.header.paragraphs:
                        if para.text.strip():
                            header_text += para.text.strip() + " "
                    if header_text:
                        content.insert(0, f"=== HEADER: {header_text.strip()} ===\n")
            
            full_content = '\n'.join(content)
            return self.enhanced_clean_text(full_content)
            
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"

    def read_pdf_advanced(self) -> str:
        """อ่าน PDF อย่างละเอียดและแม่นยำ"""
        try:
            doc = fitz.open(self.file_path)
            full_content = []
            
            # เก็บข้อมูล metadata
            metadata = doc.metadata
            if metadata:
                self.metadata = {
                    'title': metadata.get('title', ''),
                    'author': metadata.get('author', ''),
                    'subject': metadata.get('subject', ''),
                    'pages': len(doc)
                }
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # ลองหลายวิธีในการดึงข้อความ
                methods = [
                    lambda p: p.get_text("text"),
                    lambda p: p.get_text("dict"),
                    lambda p: p.get_text("html"),
                    lambda p: p.get_text("blocks")
                ]
                
                page_content = ""
                
                # วิธีที่ 1: ข้อความทั่วไป
                try:
                    text = page.get_text("text")
                    if text.strip():
                        page_content = self.enhanced_clean_text(text)
                except:
                    pass
                
                # วิธีที่ 2: ดึงข้อมูลแบบ structured
                if not page_content:
                    try:
                        blocks = page.get_text("dict")["blocks"]
                        text_blocks = []
                        
                        for block in blocks:
                            if "lines" in block:  # text block
                                for line in block["lines"]:
                                    line_text = ""
                                    for span in line["spans"]:
                                        line_text += span["text"]
                                    if line_text.strip():
                                        text_blocks.append(line_text.strip())
                        
                        page_content = '\n'.join(text_blocks)
                        page_content = self.enhanced_clean_text(page_content)
                    except:
                        pass
                
                # วิธีที่ 3: ถ้ายังไม่ได้ ลอง OCR (ถ้ามี)
                if not page_content:
                    try:
                        # ถ้าเป็นภาพหรือ scanned document
                        pix = page.get_pixmap()
                        # สำหรับ OCR ต้องติดตั้ง pytesseract
                        page_content = f"[หน้า {page_num + 1}: ไม่สามารถดึงข้อความได้ อาจเป็นภาพหรือ scanned document]"
                    except:
                        page_content = f"[หน้า {page_num + 1}: ไม่สามารถอ่านได้]"
                
                if page_content and page_content.strip():
                    full_content.append(f"\n=== หน้า {page_num + 1} ===")
                    full_content.append(page_content)
                    full_content.append("=" * 50)
            
            doc.close()
            
            final_content = '\n'.join(full_content)
            return final_content.strip() if final_content else "ไม่สามารถดึงข้อความจาก PDF ได้"
            
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def create_content_index(self, content: str) -> Dict[str, List[Tuple[str, int]]]:
        """สร้าง index สำหรับค้นหา"""
        index = {}
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines):
            words = re.findall(r'\b\w+\b', line.lower())
            for word in words:
                if len(word) > 2:  # เฉพาะคำที่มีความยาวมากกว่า 2
                    if word not in index:
                        index[word] = []
                    index[word].append((line.strip(), line_num))
        
        return index

    def smart_search(self, content: str, search_term: str, context_lines: int = 2) -> str:
        """ค้นหาแบบอัจฉริยะ"""
        if not content or not search_term:
            return "ไม่มีข้อมูลสำหรับค้นหา"
        
        search_terms = search_term.lower().split()
        lines = content.split('\n')
        results = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # ค้นหาแบบ exact match
            if search_term.lower() in line_lower:
                context_start = max(0, i - context_lines)
                context_end = min(len(lines), i + context_lines + 1)
                context = lines[context_start:context_end]
                
                # ไฮไลต์คำที่ค้นหา
                highlighted_line = line
                for term in search_terms:
                    highlighted_line = re.sub(
                        f'({re.escape(term)})', 
                        r'**\1**', 
                        highlighted_line, 
                        flags=re.IGNORECASE
                    )
                
                result = {
                    'line_number': i + 1,
                    'matched_line': highlighted_line,
                    'context': context,
                    'relevance_score': len([t for t in search_terms if t in line_lower])
                }
                results.append(result)
            
            # ค้นหาแบบ partial match
            elif any(term in line_lower for term in search_terms):
                relevance = sum(1 for term in search_terms if term in line_lower)
                if relevance >= len(search_terms) * 0.5:  # อย่างน้อยครึ่งหนึ่งของคำค้นหา
                    context_start = max(0, i - context_lines)
                    context_end = min(len(lines), i + context_lines + 1)
                    context = lines[context_start:context_end]
                    
                    result = {
                        'line_number': i + 1,
                        'matched_line': line,
                        'context': context,
                        'relevance_score': relevance
                    }
                    results.append(result)
        
        # เรียงตาม relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        if not results:
            # ลองค้นหาคำที่คล้ายกัน
            similar_results = []
            for i, line in enumerate(lines):
                similarity = 0
                for term in search_terms:
                    if any(term in word for word in line.lower().split()):
                        similarity += 1
                
                if similarity > 0:
                    similar_results.append((i + 1, line, similarity))
            
            if similar_results:
                similar_results.sort(key=lambda x: x[2], reverse=True)
                return f"❓ ไม่พบคำว่า '{search_term}' แต่พบคำที่คล้ายกันในบรรทัดที่ {similar_results[0][0]}: {similar_results[0][1][:100]}..."
            else:
                return f"❌ ไม่พบคำว่า '{search_term}' ในเอกสาร"
        
        # สร้างผลลัพธ์
        output = [f"🔍 พบคำว่า '{search_term}' ในเอกสาร {len(results)} ตำแหน่ง:\n"]
        
        for i, result in enumerate(results[:10]):  # แสดงสูงสุด 10 ผลลัพธ์
            output.append(f"\n=== ผลลัพธ์ที่ {i + 1} (บรรทัด {result['line_number']}) ===")
            output.append(f"📍 {result['matched_line']}")
            
            if result['context']:
                output.append("\n📖 บริบท:")
                for ctx_line in result['context']:
                    if ctx_line.strip():
                        prefix = "➤ " if ctx_line == result['matched_line'] else "  "
                        output.append(f"{prefix}{ctx_line}")
        
        if len(results) > 10:
            output.append(f"\n... และอีก {len(results) - 10} ผลลัพธ์")
        
        return '\n'.join(output)

    def get_comprehensive_summary(self, max_chars: int = 20000) -> str:
        """สร้างสรุปเอกสารอย่างละเอียด"""
        if not self.validate_file():
            return "Error: Could not read the dataset file."

        try:
            # อ่านเอกสารตามประเภท
            if self.file_path.lower().endswith('.txt'):
                content = self.read_txt_file()
            elif self.file_path.lower().endswith('.docx'):
                content = self.read_docx_advanced()
            elif self.file_path.lower().endswith('.pdf'):
                content = self.read_pdf_advanced()
            else:
                return "Error: Unsupported file type"
            
            if not content or content.startswith("Error"):
                return content
            
            # สร้าง sections และ keywords
            sections = self.segment_text(content)
            keywords = self.extract_keywords(content)
            
            # สร้าง index สำหรับค้นหา
            content_index = self.create_content_index(content)
            
            # ตัดเนื้อหาถ้ายาวเกินไป
            original_length = len(content)
            if len(content) > max_chars:
                # พยายามตัดที่จุดที่เหมาะสม
                content = content[:max_chars]
                
                # หาจุดตัดที่ดี
                for cutoff in ['.', '\n\n', '\n', ' ']:
                    last_cutoff = content.rfind(cutoff)
                    if last_cutoff > max_chars * 0.8:
                        content = content[:last_cutoff + (1 if cutoff == '.' else 0)]
                        break
                
                content += f"\n\n[📄 หมายเหตุ: เนื้อหาถูกตัดเพื่อประหยัด token จาก {original_length:,} เป็น {len(content):,} ตัวอักษร]"
            
            # เพิ่มข้อมูล metadata
            summary_parts = []
            
            if self.metadata:
                summary_parts.append("=== ข้อมูลเอกสาร ===")
                for key, value in self.metadata.items():
                    if value:
                        summary_parts.append(f"{key}: {value}")
                summary_parts.append("")
            
            if keywords:
                summary_parts.append("=== คำสำคัญ (Top 20) ===")
                summary_parts.append(", ".join(keywords[:20]))
                summary_parts.append("")
            
            if len(sections) > 1:
                summary_parts.append("=== หัวข้อในเอกสาร ===")
                for section_name in sections.keys():
                    summary_parts.append(f"- {section_name.replace('_', ' ').title()}")
                summary_parts.append("")
            
            summary_parts.append("=== เนื้อหาเอกสาร ===")
            summary_parts.append(content)
            summary_parts.append(f"\n\n[✅ เอกสาร KMUTNB ถูกโหลดและประมวลผลเรียบร้อยแล้ว - รวม {len(content):,} ตัวอักษร]")
            
            final_content = '\n'.join(summary_parts)
            
            # เก็บไว้ใน cache
            self.content_cache['full_content'] = final_content
            self.content_cache['sections'] = sections
            self.content_cache['keywords'] = keywords
            self.content_cache['index'] = content_index
            
            return final_content
                
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def read_document(self) -> Optional[str]:
        """อ่านเอกสารแบบเต็ม (backward compatibility)"""
        return self.get_comprehensive_summary(max_chars=50000)

# ฟังก์ชันหลักที่ app.py จะเรียกใช้
def get_kmutnb_summary(file_path: str) -> str:
    """ฟังก์ชันหลักสำหรับอ่านและสรุปเอกสาร KMUTNB (แบบใหม่)"""
    reader = EnhancedDocumentReader(file_path)
    return reader.get_comprehensive_summary()

def read_kmutnb_dataset(file_path: str) -> str:
    """อ่านเอกสาร KMUTNB แบบเต็ม (แบบใหม่)"""
    reader = EnhancedDocumentReader(file_path)
    content = reader.get_comprehensive_summary(max_chars=100000)
    if content is None or content.startswith("Error"):
        return "Error: Could not read the dataset file."
    return content

def search_in_document(file_path: str, search_term: str) -> str:
    """ค้นหาคำในเอกสาร (แบบใหม่)"""
    reader = EnhancedDocumentReader(file_path)
    content = reader.read_document()
    
    if not content or content.startswith("Error"):
        return "Error: Could not read the document."
    
    return reader.smart_search(content, search_term)

# สำหรับ testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = "test_document.pdf"
    
    if os.path.exists(test_file):
        print("Testing Enhanced Document Reader...")
        print("=" * 50)
        
        reader = EnhancedDocumentReader(test_file)
        summary = reader.get_comprehensive_summary()
        
        print("Document Summary:")
        print(summary[:2000] + "..." if len(summary) > 2000 else summary)
        
        # Test search
        if len(sys.argv) > 2:
            search_term = sys.argv[2]
            print(f"\n\nSearching for '{search_term}':")
            print("=" * 50)
            search_result = reader.smart_search(summary, search_term)
            print(search_result)
    else:
        print(f"Test file '{test_file}' not found")
        print("Usage: python document_reader.py <file_path> [search_term]")