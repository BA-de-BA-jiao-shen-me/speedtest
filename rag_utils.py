import os
from typing import List, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader, PDFPlumberLoader, Docx2txtLoader,
    UnstructuredMarkdownLoader, UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader, UnstructuredCSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

class KnowledgeBase:
    def __init__(
        self,
        kb_dir: str,
        embedding_model: str = 'bge-m3',
        chunk_size: int = 500,
        overlap: int = 50,
        build_new: bool = False,
        cache_prefix: str = "kb_index",
        device: str = None
    ):
        self.kb_dir = kb_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cache_prefix = cache_prefix
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device if device else "cpu"}
        )
        self.persist_path = f"{self.cache_prefix}_faiss"
        self.vectorstore = None

        if build_new or not os.path.exists(self.persist_path):
            self.build_index()
        else:
            self.vectorstore = FAISS.load_local(
                self.persist_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )

    def build_index(self):
        docs = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap
        )
        files = os.listdir(self.kb_dir)
        for filename in tqdm(files, desc="知识库构建进度"):
            fpath = os.path.join(self.kb_dir, filename)
            ext = filename.lower().split('.')[-1]
            if ext == "txt":
                loader = TextLoader(fpath, autodetect_encoding=True)
            elif ext == "pdf":
                loader = PDFPlumberLoader(fpath)
            elif ext == "docx":
                loader = Docx2txtLoader(fpath)
            elif ext == "md":
                loader = UnstructuredMarkdownLoader(fpath)
            elif ext == "html":
                loader = UnstructuredHTMLLoader(fpath)
            elif ext == "pptx":
                loader = UnstructuredPowerPointLoader(fpath)
            elif ext == "xlsx":
                loader = UnstructuredExcelLoader(fpath)
            elif ext == "csv":
                loader = UnstructuredCSVLoader(fpath)
            else:
                continue  # 跳过未知格式
            docs.extend(loader.load_and_split(text_splitter=splitter))
        print(f"总共切分出 {len(docs)} 个知识片段，开始建立向量索引...")
        # 用FAISS存储
        self.vectorstore = FAISS.from_documents(
            docs,
            self.embedding_model,
        )
        self.vectorstore.save_local(self.persist_path)

    def query(self, query_text: str, top_k=5) -> List[Tuple[str, float]]:
        # 返回[(文本, 相似度分数)]，兼容原接口
        if not self.vectorstore:
            return []
        results = self.vectorstore.similarity_search_with_score(query_text, k=top_k)
        return [(doc.page_content, score) for doc, score in results]

