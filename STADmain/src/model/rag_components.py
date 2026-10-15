from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import copy
import torch
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

def _get_embed_device(model):

    try:
        return model.model.embed_tokens.weight.device
    except Exception:
        return next(model.parameters()).device


def _llama_cag_knowledge_prompt(documents_text: str, answer_instruction: Optional[str] = None) -> str:
    
    if answer_instruction is None:
        answer_instruction = "Answer the question with a super short answer."
    return f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {documents_text}
    ------------------------------------------------
    {answer_instruction}
    Question:
    """


def preprocess_knowledge(
    model,
    tokenizer,
    documents_text: str,
    device: str = "cuda",
) -> Tuple[DynamicCache, int]:
    
    prompt = _llama_cag_knowledge_prompt(documents_text)
    embed_device = _get_embed_device(model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
    past_key_values = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    kv_cache = outputs.past_key_values
    kv_len = kv_cache.key_cache[0].shape[-2]
    return kv_cache, kv_len

def clean_up(kv: DynamicCache, origin_len: int):
    
    for i in range(len(kv.key_cache)):
        kv.key_cache[i]   = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]


def generate_with_kv(
    model,
    tokenizer,
    query: str,
    kv_cache: DynamicCache,
    kv_len: int,
    device: str = "cuda",
    max_new_tokens: int = 100,
) -> str:
   
    clean_up(kv_cache, kv_len)
    embed_device = _get_embed_device(model)

    prompt = f"""
    {query}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)

    origin_ids = input_ids
    output_ids = input_ids.clone()
    next_token = input_ids
    past_key_values = kv_cache

    eos = model.config.eos_token_id
    if eos is None:
        eos_ids = set()
    elif isinstance(eos, int):
        eos_ids = {eos}
    else:
        eos_ids = set(eos)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1).to(embed_device)
            past_key_values = outputs.past_key_values
            output_ids = torch.cat([output_ids, next_token], dim=1)
            if eos_ids and int(next_token.item()) in eos_ids:
                break

    new_tokens = output_ids[:, origin_ids.shape[-1]:]
    answer = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
    return answer



class KnowledgeRetriever:
    

    def __init__(
        self,
        knowledge_base_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        strict: bool = True,
        filter_question_like: bool = False,
    ):
        self.encoder = SentenceTransformer(embedding_model)
        self.top_k = top_k
        self.strict = bool(strict)
        self.filter_question_like = bool(filter_question_like)
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.index           = self._build_faiss_index()
        self.retrieval_count = 0
        print(f"KnowledgeRetriever: {len(self.knowledge_base)} documents indexed")

    @staticmethod
    def _is_question(text: str) -> bool:
        t = text.strip()
        if not t or t.endswith('?'):
            return True
        first_word = t.split()[0].lower()
        question_starters = {
            'who', 'what', 'when', 'where', 'why', 'how', 'which',
            'is', 'are', 'was', 'were', 'does', 'did', 'do',
            'can', 'could', 'would', 'will', 'has', 'have', 'had',
        }
        return first_word in question_starters and len(t.split()) < 25

    def _load_knowledge_base(self, path: str) -> List[Dict]:
     
        if not os.path.exists(path):
            msg = f"Knowledge base not found: {path}"
            if self.strict:
                raise FileNotFoundError(msg)
            print(f"{msg} — using dummy KB because strict=False")
            return self._dummy_kb()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            msg = f"Could not load knowledge base at {path}: {e}"
            if self.strict:
                raise RuntimeError(msg) from e
            print(f"{msg} — using dummy KB because strict=False")
            return self._dummy_kb()

        if not isinstance(data, list) or len(data) == 0:
            msg = f"Knowledge base at {path} must be a non-empty JSON list"
            if self.strict:
                raise ValueError(msg)
            print(f"{msg} — using dummy KB because strict=False")
            return self._dummy_kb()

        clean = []
        for i, d in enumerate(data):
            if not isinstance(d, dict):
                continue
            text = str(d.get("text", "")).strip()
            if not text:
                continue
            if self.filter_question_like and self._is_question(text):
                continue
            doc = d.copy()
            doc.setdefault("id", i)
            doc["text"] = text
            clean.append(doc)

        if not clean:
            msg = (
                f"Knowledge base at {path} contains no usable documents "
                f"after validation/filtering"
            )
            if self.strict:
                raise ValueError(msg)
            print(f"{msg} — using dummy KB because strict=False")
            return self._dummy_kb()

        dropped = len(data) - len(clean)
        print(f"   Loaded {len(clean)} documents from {path}")
        if dropped:
            print(f"   Dropped {dropped} unusable/filtered entries")
        return clean

    def _dummy_kb(self) -> List[Dict]:
        return [
            {"id": 0,  "text": "Sports competitions include football, basketball, tennis, and athletics. Major tournaments such as the World Cup and Olympics attract global audiences. Football is the world's most popular sport with billions of fans. The FIFA World Cup is held every four years and is the most watched sporting event globally.", "category": "sports"},
            {"id": 1,  "text": "Political systems vary from democracies to authoritarian regimes. Elections, legislation, and international relations shape government policy worldwide. Democratic governments hold regular elections where citizens vote for their representatives. Authoritarian regimes concentrate power in a single leader or party.", "category": "politics"},
            {"id": 2,  "text": "Financial markets allow trading of stocks, bonds, and commodities. Central banks regulate interest rates to control inflation and economic growth. The stock market enables companies to raise capital by selling shares to investors. Bond markets allow governments and corporations to borrow money from investors.", "category": "business"},
            {"id": 3,  "text": "Technology companies develop software, hardware, and digital services. Artificial intelligence and machine learning are driving major advances across industries. Machine learning algorithms learn patterns from data to make predictions and decisions. Natural language processing enables computers to understand and generate human language.", "category": "technology"},
            {"id": 4,  "text": "Scientific research advances understanding of biology, chemistry, and physics. Medical breakthroughs include vaccines, gene editing, and cancer treatments. CRISPR gene editing technology allows scientists to modify DNA sequences precisely. mRNA vaccines represent a new approach to immunization that was used for COVID-19.", "category": "science"},
            {"id": 5,  "text": "Global trade involves import and export of goods between countries. Supply chain disruptions affect manufacturing and consumer prices worldwide. International trade agreements reduce tariffs and promote economic cooperation between nations. The World Trade Organization oversees global trade rules and dispute resolution.", "category": "business"},
            {"id": 6,  "text": "Climate change is caused by greenhouse gas emissions from burning fossil fuels. Rising temperatures affect weather patterns, sea levels, and ecosystems. Carbon dioxide and methane are the primary greenhouse gases driving global warming. Renewable energy sources like solar and wind power reduce carbon emissions.", "category": "science"},
            {"id": 7,  "text": "Digital communication platforms enable instant messaging, video calls, and content sharing across the globe. Social media has transformed how people communicate and access information. The internet connects billions of people and devices worldwide through a global network infrastructure.", "category": "technology"},
            {"id": 8,  "text": "Banking services include savings accounts, loans, mortgages, and investment products. Regulatory bodies oversee financial institutions to protect consumers. Central banks like the Federal Reserve control money supply and interest rates to stabilize economies.", "category": "business"},
            {"id": 9,  "text": "International diplomacy involves negotiations between governments to resolve disputes and form alliances. The United Nations was founded in 1945 to promote international peace and cooperation. NATO is a military alliance of North American and European countries formed in 1949.", "category": "politics"},
            {"id": 10, "text": "Healthcare systems deliver medical services through hospitals, clinics, and community health programmes. Modern medicine uses advanced diagnostics including MRI, CT scans, and blood tests. Antibiotics revolutionized treatment of bacterial infections and have saved millions of lives.", "category": "health"},
            {"id": 11, "text": "Space agencies conduct missions to explore the solar system and study Earth from orbit. NASA's Apollo program landed humans on the Moon between 1969 and 1972. Mars rovers have been exploring the Martian surface since the 1990s collecting geological data.", "category": "science"},
            {"id": 12, "text": "Education systems provide structured learning from primary school through university. Universities conduct research and train students in specialized fields of knowledge. Online education platforms have expanded access to learning opportunities globally.", "category": "education"},
            {"id": 13, "text": "Entertainment industries produce films, television shows, music, and games. Streaming platforms like Netflix and Spotify have transformed media consumption globally. The film industry produces thousands of movies annually across many genres and languages.", "category": "entertainment"},
            {"id": 14, "text": "Agricultural production feeds the world population through crops, livestock, and aquaculture. The Green Revolution in the 1960s dramatically increased crop yields through new varieties and fertilizers. Modern precision farming uses GPS and sensors to optimize planting and irrigation.", "category": "science"},
            {"id": 15, "text": "Energy production relies on coal, natural gas, nuclear power, solar panels, and wind turbines. Nuclear power plants generate electricity through controlled fission reactions. Solar photovoltaic panels convert sunlight directly into electricity with no emissions.", "category": "science"},
            {"id": 16, "text": "Transport networks connect cities and countries through roads, railways, airports, and shipping lanes. High-speed rail systems in Japan, China, and Europe travel at over 300 km/h. Electric vehicles use battery-powered motors instead of internal combustion engines.", "category": "technology"},
            {"id": 17, "text": "Law enforcement and justice systems maintain public order and enforce legal codes. Criminal courts determine guilt and impose sentences on those convicted of crimes. Civil courts resolve disputes between individuals, companies, and organizations.", "category": "law"},
            {"id": 18, "text": "Cultural diversity encompasses languages, religions, traditions, and artistic expressions. There are approximately 7000 languages spoken worldwide though many are endangered. Religion influences the values, practices, and social structures of billions of people.", "category": "culture"},
            {"id": 19, "text": "Philosophy explores fundamental questions about existence, knowledge, ethics, and reality. Major philosophical traditions include Western analytic philosophy, continental philosophy, and Eastern traditions. Ethics is the branch of philosophy concerned with moral principles and human conduct.", "category": "philosophy"},
        ]

    def _build_faiss_index(self):
        texts      = [doc["text"] for doc in self.knowledge_base]
        embeddings = self.encoder.encode(
            texts, show_progress_bar=False,
            convert_to_numpy=True, batch_size=32,
        )
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        return index

    def retrieve(
        self,
        query:   str,
        context: Optional[str] = None,
        top_k:   Optional[int] = None,
    ) -> List[Dict]:
        self.retrieval_count += 1
        k           = top_k if top_k is not None else self.top_k
        search_text = f"{context} {query}" if context else query

        q_emb = self.encoder.encode([search_text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(
            q_emb.astype(np.float32),
            min(k * 3, len(self.knowledge_base))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.knowledge_base):
                continue
            doc  = self.knowledge_base[idx].copy()
            if self.filter_question_like and self._is_question(doc.get("text", "")):
                continue
            doc["retrieval_score"] = float(score)
            results.append(doc)
            if len(results) >= k:
                break

        if not results:
            for score, idx in zip(scores[0][:k], indices[0][:k]):
                if idx < len(self.knowledge_base):
                    doc = self.knowledge_base[idx].copy()
                    doc["retrieval_score"] = float(score)
                    results.append(doc)

        return results

    def get_retrieval_stats(self) -> Dict:
        return {
            "total_retrievals":    self.retrieval_count,
            "knowledge_base_size": len(self.knowledge_base),
        }



class StateKnowledgeCache:
   

    def __init__(self, cache_size_mb: int = 500):
        self.cache:      Dict[int, List[Dict]] = {}   # raw docs per state
        self._kv_cache:  Dict[int, DynamicCache] = {} # KV tensors per state
        self._kv_len:    Dict[int, int]          = {} # doc-only KV length
        self.cache_hits:      int = 0
        self.cache_misses:    int = 0
        self.retrieval_calls: int = 0

    
    def get(self, state_id: int) -> Optional[List[Dict]]:
        if state_id in self.cache:
            self.cache_hits += 1
            return self.cache[state_id]
        self.cache_misses += 1
        return None

    def set(self, state_id: int, knowledge: List[Dict]):
        self.cache[state_id] = knowledge
        self.retrieval_calls += 1
        # Invalidate KV cache — pool has changed
        self._kv_cache.pop(state_id, None)
        self._kv_len.pop(state_id, None)

    def accumulate(self, state_id: int, new_docs: List[Dict], max_docs: int = 30):
        
        if state_id not in self.cache:
            self.cache[state_id] = []
        existing_ids = {d.get("id") for d in self.cache[state_id]}
        for doc in new_docs:
            if doc.get("id") not in existing_ids:
                self.cache[state_id].append(doc)
                existing_ids.add(doc.get("id"))
        self.cache[state_id].sort(
            key=lambda x: x.get("retrieval_score", 0), reverse=True
        )
        self.cache[state_id]  = self.cache[state_id][:max_docs]
        self.retrieval_calls += 1
        # Invalidate KV cache — pool has grown
        self._kv_cache.pop(state_id, None)
        self._kv_len.pop(state_id, None)

    
    def get_kv_cache(
        self,
        state_id: int,
        generator: "LlamaKVGenerator",
        top_k: Optional[int] = None,
        max_chars_per_doc: Optional[int] = None,
    ) -> Tuple[Optional[DynamicCache], int]:
      
        if state_id in self._kv_cache:
            self.cache_hits += 1
            return self._kv_cache[state_id], self._kv_len[state_id]

        docs = self.cache.get(state_id, [])
        if not docs:
            self.cache_misses += 1
            return None, 0

        use_docs = docs if top_k is None else docs[:int(top_k)]
        doc_texts = []
        for rank, doc in enumerate(use_docs, 1):
            text = str(doc.get("text", "")).strip()
            title = str(doc.get("title", "")).strip()
            if len(text) < 20:
                continue
            if max_chars_per_doc is not None and max_chars_per_doc > 0:
                text = text[:int(max_chars_per_doc)]
            if title:
                doc_texts.append(f"[Document {rank}]\nTitle: {title}\n{text}")
            else:
                doc_texts.append(f"[Document {rank}]\n{text}")

        if not doc_texts:
            self.cache_misses += 1
            return None, 0

        
        documents_text = "\n\n\n\n\n\n".join(doc_texts)
        kv_cache, kv_len = generator.preprocess_knowledge(documents_text)
        self._kv_cache[state_id] = kv_cache
        self._kv_len[state_id] = kv_len
        self.cache_hits += 1
        return kv_cache, kv_len

  
    def has(self, state_id: int) -> bool:
        return state_id in self.cache

    def clear_state(self, state_id: int):
        self.cache.pop(state_id, None)
        self._kv_cache.pop(state_id, None)
        self._kv_len.pop(state_id, None)

    def clear_all(self):
        self.cache.clear()
        self._kv_cache.clear()
        self._kv_len.clear()
        self.cache_hits      = 0
        self.cache_misses    = 0
        self.retrieval_calls = 0

    def get_state_knowledge_size(self, state_id: int) -> int:
        return len(self.cache.get(state_id, []))

    @staticmethod
    def _canonical_doc_key(doc: Dict) -> Tuple[str, str]:
        
        title = str(doc.get("title", "")).strip().lower()
        if title:
            return ("title", title)

        doc_id = doc.get("id")
        if doc_id is not None:
            return ("id", str(doc_id))

        text = str(doc.get("text", "")).strip().lower()
        return ("text", text[:300])

    def aggregate_knowledge_pool(
        self,
        knowledge_list: List[List[Dict]],
        max_docs: int = 80,
        max_per_title: int = 1,
    ) -> List[Dict]:
        
        all_docs = [doc for docs in knowledge_list for doc in docs]

        best_by_key: Dict[Tuple[str, str], Dict] = {}
        hit_counts: Dict[Tuple[str, str], int] = {}

        for doc in all_docs:
            text = str(doc.get("text", "")).strip()
            if not text:
                continue

            key = self._canonical_doc_key(doc)
            hit_counts[key] = hit_counts.get(key, 0) + 1
            score = float(doc.get("retrieval_score", 0.0))

            if key not in best_by_key or score > float(best_by_key[key].get("retrieval_score", 0.0)):
                chosen = doc.copy()
                chosen["state_cache_retrieval_hits"] = hit_counts[key]
                best_by_key[key] = chosen
            else:
                best_by_key[key]["state_cache_retrieval_hits"] = hit_counts[key]

        unique = list(best_by_key.values())
        unique.sort(
            key=lambda x: (
                int(x.get("state_cache_retrieval_hits", 1)),
                float(x.get("retrieval_score", 0.0)),
            ),
            reverse=True,
        )

      
        selected = []
        title_counts: Dict[str, int] = {}
        for doc in unique:
            title = str(doc.get("title", "")).strip().lower()
            title_key = title or str(doc.get("id", "unknown"))
            title_counts[title_key] = title_counts.get(title_key, 0)
            if title_counts[title_key] >= int(max_per_title):
                continue

            selected.append(doc)
            title_counts[title_key] += 1
            if len(selected) >= int(max_docs):
                break

        return selected

    def get_stats(self) -> Dict:
        total             = self.cache_hits + self.cache_misses
        hit_rate          = self.cache_hits / total if total > 0 else 0.0
        total_docs_cached = sum(len(d) for d in self.cache.values())
        return {
            "cache_size":          len(self.cache),
            "cache_hits":          self.cache_hits,
            "cache_misses":        self.cache_misses,
            "retrieval_calls":     self.retrieval_calls,
            "cache_hit_rate":      hit_rate,
            "retrieval_reduction": 1.0 - (self.retrieval_calls / total) if total > 0 else 0.0,
            "total_docs_cached":   total_docs_cached,
            "unique_states":       len(self.cache),
            "kv_caches_built":     len(self._kv_cache),
        }

    def get_detailed_stats(self) -> Dict:
        stats = self.get_stats()
        per_state = {}
        for state_id, docs in self.cache.items():
            per_state[state_id] = {
                "num_docs":       len(docs),
                "avg_score":      float(np.mean([d.get("retrieval_score", 0) for d in docs])),
                "categories":     list(set(d.get("category", "unknown") for d in docs)),
                "kv_cache_built": state_id in self._kv_cache,
            }
        stats["per_state_breakdown"] = per_state
        return stats




class LlamaKVGenerator:
    

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        max_new_tokens: int = 100,
    ):
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_count = 0

        print(f"Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if str(self.device).startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device requested ({self.device}) but torch.cuda.is_available() is False")
            if ":" in str(self.device):
                cuda_index = int(str(self.device).split(":", 1)[1])
            else:
                cuda_index = 0
            device_map = {"": cuda_index}
            dtype = torch.float16
        else:
            device_map = {"": "cpu"}
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ {model_name} ready on {self.device}")

    def preprocess_knowledge(self, documents_text: str) -> Tuple[DynamicCache, int]:
        
        return preprocess_knowledge(
            self.model,
            self.tokenizer,
            documents_text,
            self.device,
        )

    def generate_with_kv_cache(
        self,
        query: str,
        kv_cache: Optional[DynamicCache],
        kv_len: int,
    ) -> str:
       
        self.generation_count += 1

        if kv_cache is None or kv_len <= 0:
            return self.generate_fresh(query, docs=[])

        return generate_with_kv(
            self.model,
            self.tokenizer,
            query,
            kv_cache,
            kv_len,
            self.device,
            self.max_new_tokens,
        )

    def generate_fresh(self, query: str, docs: List[Dict]) -> str:
        
        self.generation_count += 1

        clean_docs = [
            doc for doc in (docs or [])
            if len(doc.get("text", "").strip()) >= 20
        ]

        if not clean_docs:
            prompt = (
                "<|system|>\n"
                "You are a helpful assistant.<|end|>\n"
                "<|user|>\n"
                f"Answer this question concisely: {query}<|end|>\n"
                "<|assistant|>\n"
            )
        else:
            context_parts = []
            for doc in clean_docs[:5]:
                text = doc.get("text", "").strip()
                title = doc.get("title", "").strip()
                entry = f"{title}\n{text}" if title else text
                context_parts.append(entry[:2000])

            context = "\n\n".join(context_parts)
            prompt = (
                "<|system|>\n"
                "You are a helpful assistant that answers questions "
                "using only the provided context.<|end|>\n"
                "<|user|>\n"
                f"Context:\n{context}\n\n"
                "Answer this question concisely using only the context: "
                f"{query}<|end|>\n"
                "<|assistant|>\n"
            )

        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, input_ids.shape[-1]:]
        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        return answer if answer else f"[No answer for: {query[:60]}]"

    def get_stats(self) -> Dict:
        return {
            "total_generations": self.generation_count,
            "model_name": self.model_name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
        }



class HuggingFaceGenerator(LlamaKVGenerator):
   
    def generate(self, query: str, knowledge=None) -> str:
        
        if isinstance(knowledge, list):
            return self.generate_fresh(query, knowledge)
        elif isinstance(knowledge, str) and knowledge.strip():
            # Plain text context — wrap in doc dict
            return self.generate_fresh(query, [{"text": knowledge, "id": 0}])
        else:
            return self.generate_fresh(query, [])



class SimpleGenerator:
    def __init__(self, max_context_length: int = 150):
        self.max_context_length = max_context_length
        self.generation_count   = 0

    def generate(self, query: str, knowledge: List[Dict]) -> str:
        self.generation_count += 1
        if not knowledge:
            return "No relevant information available."
        docs    = [doc.get("text", "") for doc in knowledge[:3]]
        context = " ".join(docs)
        words   = context.split()
        if len(words) > self.max_context_length:
            context = " ".join(words[:self.max_context_length]) + "..."
        first_sentence = context.split(".")[0] if "." in context else context[:100]
        return f"Based on retrieved information: {first_sentence}."

    def get_stats(self) -> Dict:
        return {
            "total_generations":  self.generation_count,
            "max_context_length": self.max_context_length,
        }