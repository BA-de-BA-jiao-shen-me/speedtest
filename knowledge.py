import argparse
import logging
from fastapi.middleware.cors import CORSMiddleware

from ollama_client_lib import load_config
from rag_utils import KnowledgeBase


# --------- 日志配置 ---------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
file_handler = None
main_logger = logging.getLogger(__name__)

config = load_config()

# --------- RAG知识库参数 ---------
RAG_KB_DIR = config.get('rag_kb_dir', 'knowledge_base')


def build_kb_mode():
    print("正在构建知识库并保存...")
    kb = KnowledgeBase(RAG_KB_DIR, build_new=True, device='cuda')
    print("知识库已完成构建和保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    build_kb_mode()
