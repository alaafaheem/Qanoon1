# %%
import asyncio
import os
import warnings
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Optional,List
from pinecone import Pinecone
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Event


# %%

warnings.filterwarnings("ignore")
load_dotenv()
Settings.llm = OpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
vector_store = PineconeVectorStore(pinecone_index=pc.Index("qanoon"))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# %%

class SubQuery(BaseModel):
    search_query: str = Field(description="صيغة البحث باللغة العربية الفصحى لهذا الجزء المحدد من السؤال.")
    status_filter: Optional[str] = Field(default=None, description="ساري ونافذ أو ملغى (إن ذُكرت)، وإلا None.")
    year_filter: Optional[str] = Field(default=None, description="سنة الإصدار (إن ذُكرت)، وإلا None.")
    category_filter: Optional[str] = Field(default=None, description="تصنيف القانون (إن ذُكر)، وإلا None.")

class AnalyzerOutput(BaseModel):
    is_question: bool = Field(description="True إذا كان هناك سؤال قانوني، False إذا كانت مجرد تحية.")
    chat_reply: str = Field(description="إذا كانت مجرد تحية، اكتب الرد هنا.")
    queries: List[SubQuery] = Field(description="قائمة بالأسئلة للبحث. إذا كان السؤال يحتوي على أكثر من شق، قم بتفكيكه إلى عدة أسئلة فرعية.")

class ReviewerOutput(BaseModel):
    is_approved: bool = Field(description="True إذا تمت الإجابة على كل أجزاء السؤال بدقة، False إذا تم إهمال جزء أو الإجابة خاطئة.")
    missing_parts: List[str] = Field(default=[], description="الأجزاء التي سأل عنها المستخدم ولم يتم الإجابة عليها (إن وجدت).")
    critique: str = Field(description="ملاحظات توضح سبب الرفض بالتفصيل وما الذي يجب تصحيحه، أو 'ممتاز' إذا قُبلت.")
    error_source: str = Field(description="'none' (مقبولة), 'drafter' (الخطأ في الصياغة أو تجاهل نصوص موجودة), 'researcher' (النصوص المسترجعة لا تغطي كل أجزاء السؤال).")



class RetryResearchEvent(Event):
    question: str
    critique: str
    revision_count: int

class DraftEvent(Event):
    question: str
    retrieved_laws: str
    applied_filters: dict 
    critique: str
    revision_count: int

class ReviewEvent(Event):
    question: str
    retrieved_laws: str
    applied_filters: dict 
    draft_answer: str
    revision_count: int

# %%
class LegalAssistantWorkflow(Workflow):
    
    @step
    async def analyzer_and_researcher(self, ev: StartEvent | RetryResearchEvent) -> DraftEvent | StopEvent:
        if isinstance(ev, StartEvent):
            query = ev.get("question")
            revision_count = 0
            critique = ""
        else:
            query = ev.question
            revision_count = ev.revision_count
            critique = ev.critique
            print(f"\n[RE-ANALYZER] {revision_count + 1} ")
            print(f"{critique}")
        
        prompt_text = f"حلل رسالة المستخدم التالية: {query}\n"
        prompt_text += "ملاحظة هامة: إذا كان سؤال المستخدم يحتوي على عدة أجزاء مختلفة (مثلاً يسأل عن قانونين مختلفين أو موضوعين)، قم بتقسيمها ووضع كل جزء كـ SubQuery منفصل في القائمة.\n"
        
        if critique:
            prompt_text += f"\nتحذير: محاولة البحث السابقة فشلت للأسباب التالية: '{critique}'.\nيرجى تعديل صيغ البحث أو تقليل الفلاتر لضمان جلب نتائج تغطي جميع الأجزاء."
        
        prompt = PromptTemplate(prompt_text)
        analysis = await Settings.llm.astructured_predict(AnalyzerOutput, prompt, query=query)
        
        if not analysis.is_question:
            return StopEvent(result=analysis.chat_reply)
            
        all_nodes = []
        applied_filters_log =[]
        
     
        for sub_q in analysis.queries:
            print(f"[Researcher] '{sub_q.search_query}'")
            filters_list =[]
            applied_filters_dict = {}
            
            if sub_q.status_filter:
                filters_list.append(ExactMatchFilter(key="status", value=sub_q.status_filter))
                applied_filters_dict["status"] = sub_q.status_filter
            if sub_q.year_filter:
                filters_list.append(ExactMatchFilter(key="year", value=str(sub_q.year_filter)))
                applied_filters_dict["year"] = sub_q.year_filter
            if sub_q.category_filter:
                filters_list.append(ExactMatchFilter(key="category", value=sub_q.category_filter))
                applied_filters_dict["category"] = sub_q.category_filter
                
            metadata_filters = MetadataFilters(filters=filters_list) if filters_list else None
            applied_filters_log.append({sub_q.search_query: applied_filters_dict})
            
    
            retriever = index.as_retriever(similarity_top_k=5, filters=metadata_filters)
            nodes = retriever.retrieve(sub_q.search_query)
            all_nodes.extend(nodes)
        
        
        unique_nodes = {n.node_id: n for n in all_nodes}.values()
        
        laws_text = "\n".join([
            f"### النص ###\nالقانون: {n.metadata.get('official_name', 'غير معروف')}\nالحالة: {n.metadata.get('status', 'غير معروف')}\nالرابط: {n.metadata.get('link', 'لا يوجد')}\nالنص: {n.text}\n" 
            for n in unique_nodes
        ])
        
        if not laws_text.strip():
            laws_text = "لا توجد نصوص مطابقة للفلاتر المحددة حالياً."
            
        return DraftEvent(
            question=query, 
            retrieved_laws=laws_text, 
            applied_filters={"queries_filters": applied_filters_log},
            critique="بدون ملاحظات" if not critique else critique, 
            revision_count=revision_count
        )

    @step
    async def drafter(self, ev: DraftEvent) -> ReviewEvent:
        print(f"\n---[Drafter]({ev.revision_count + 1}) ---")
        
        prompt = f"""أنت مستشار قانوني ليبي محترف وبليغ. 
        المهمة الأهم: تأكد من الإجابة على **جميع أجزاء سؤال المستخدم** ولا تهمل أي شق منه بناءً على القوانين المرفقة.
      
     
        شروط الصياغة:
        1. ابدأ بمقدمة طبيعية تذكر فيها رقم القانون وسنته واسمه وحالته (ساري ونافذ، إلخ).
        2. قسّم الإجابة إلى نقاط مرقمة وواضحةو  لعناوين فرعية إذا كان السؤال يحتوي على أكثر من شق.
        3. كل نقطة يجب أن تبدأ بعنوان رئيسي عريض (Bold)، يليه في السطر التالي "وفقاً للمادة رقم كذا" أو "تنص المادة كذا".
        4. اشرح النص القانوني بأسلوب مبسط ومهني دون تغيير في المعنى القانوني .
        5. تأكد من إبراز الشروط الخاصة التي سأل عنها المستخدم في نقطة مستقلة (مثل الإذن الكتابي أو شروط التلبس).
        6. اذكر "رقم المادة" وأرفق الرابط في النهاية.
        7. لا تخترع أي معلومة غير موجودة في النصوص.
        8. إذا كانت القوانين المرفقة لا تحتوي إجابة لأحد أجزاء السؤال، اعتذر للمستخدم وأخبره صراحة أنك لم تجد نصاً يخص هذا الجزء المحدّد.
        
        السؤال: {ev.question}
        
        القوانين المرفقة: 
        {ev.retrieved_laws}
        
        ملاحظات التعديل السابقة: {ev.critique}
        """
        
        response = await Settings.llm.acomplete(prompt)
        
        return ReviewEvent(
            question=ev.question, 
            retrieved_laws=ev.retrieved_laws, 
            applied_filters=ev.applied_filters,
            draft_answer=str(response), 
            revision_count=ev.revision_count
        )

    @step
    async def reviewer(self, ev: ReviewEvent) -> DraftEvent | RetryResearchEvent | StopEvent:
        print(f"\n---[Reviewer]---")
        
        prompt_text = f"""أنت مراجع قانوني ليبي صارم. مهمتك التدقيق في إجابة المصيغ ومقارنتها بسؤال المستخدم والنصوص المسترجعة.
        
        سؤال المستخدم: {ev.question}
        
        النصوص المسترجعة من قاعدة البيانات:
        {ev.retrieved_laws}
        
        إجابة المصيغ:
        {ev.draft_answer}
        
        التعليمات الصارمة لتحديد مصدر الخطأ (error_source):
        1. تأكد أن الإجابة غطت **جميع أجزاء السؤال**. هل أهمل المصيغ جزءاً معيناً؟
        2. إذا تم إهمال جزء من السؤال، ابحث في "النصوص المسترجعة":
           - إذا كانت النصوص المسترجعة **لا تحتوي** على الإجابة لهذا الجزء، فالخطأ من الباحث لأن بحثه كان ناقصاً -> اختر 'researcher'.
           - إذا كانت النصوص المسترجعة **تحتوي** على الإجابة ولكن المصيغ تجاهلها ولم يكتبها -> اختر 'drafter'.
        3. إذا تم اختراع معلومات (هلوسة) غير موجودة في النصوص -> اختر 'drafter'.
        4. إذا كانت كل الأمور ممتازة وتغطي كل أجزاء السؤال -> اختر 'none'.
        """
        
        prompt = PromptTemplate(prompt_text)
        review = await Settings.llm.astructured_predict(ReviewerOutput, prompt)
        
        
        if review.is_approved or review.error_source.lower() == 'none' or ev.revision_count >= 3:
            
            return StopEvent(result=ev.draft_answer)
            
        print(f" The answer was rejected. \n  the reason: {review.critique}\n   error source : {review.error_source}")
        if review.missing_parts:
            print(f" missing parts: {review.missing_parts}")
        
        if review.error_source.lower() == 'researcher':
            return RetryResearchEvent(
                question=ev.question, 
                critique=review.critique, 
                revision_count=ev.revision_count + 1
            )
        else: # drafter
            return DraftEvent(
                question=ev.question, 
                retrieved_laws=ev.retrieved_laws, 
                applied_filters=ev.applied_filters,
                critique=review.critique, 
                revision_count=ev.revision_count + 1
            )

# %%

async def main():

    legal_assistant = LegalAssistantWorkflow(timeout=300, verbose=False)
    
    query = ""
    try:
        final_answer = await legal_assistant.run(question=query)
       
        print("\nFinal Answer:")
        print(final_answer)
    except Exception as e:
        print(f"\n  error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
