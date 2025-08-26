# RAG System Implementation for MSP Knowledge Base
# Advanced Retrieval-Augmented Generation with Vector Database

from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
from datetime import datetime
import json
import os
from pathlib import Path

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import TextLoader, PDFLoader, WebBaseLoader
import chromadb
import pinecone

class MSPKnowledgeBase:
    """Advanced RAG system specifically designed for MSP operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_api_key = config.get('openai_api_key')
        self.pinecone_api_key = config.get('pinecone_api_key')
        self.pinecone_env = config.get('pinecone_env', 'us-east-1-aws')
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Vector store options
        self.vector_store = None
        self.retrieval_chain = None
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True
        )
        
        # MSP-specific categories
        self.knowledge_categories = {
            "network_infrastructure": {
                "keywords": ["network", "switch", "router", "firewall", "VLAN", "DNS", "DHCP"],
                "priority": 1
            },
            "cybersecurity": {
                "keywords": ["security", "antivirus", "malware", "phishing", "encryption", "backup"],
                "priority": 1
            },
            "cloud_services": {
                "keywords": ["cloud", "AWS", "Azure", "Google Cloud", "migration", "hybrid"],
                "priority": 2
            },
            "compliance": {
                "keywords": ["HIPAA", "SOX", "GDPR", "PCI-DSS", "compliance", "audit"],
                "priority": 1
            },
            "helpdesk_procedures": {
                "keywords": ["ticket", "incident", "problem", "change management", "ITIL"],
                "priority": 2
            },
            "vendor_management": {
                "keywords": ["vendor", "contract", "SLA", "procurement", "licensing"],
                "priority": 3
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize_vector_store(self, documents: List[Dict], store_type: str = "faiss"):
        """Initialize vector store with MSP documents"""
        try:
            # Process and categorize documents
            processed_docs = await self._process_documents(documents)
            
            # Create embeddings
            texts = [doc["content"] for doc in processed_docs]
            metadatas = [doc["metadata"] for doc in processed_docs]
            
            if store_type == "pinecone":
                self.vector_store = await self._init_pinecone_store(texts, metadatas)
            elif store_type == "chroma":
                self.vector_store = await self._init_chroma_store(texts, metadatas)
            else:
                self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            
            # Initialize retrieval chain
            llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                temperature=0.3,
                model="gpt-4"
            )
            
            self.retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 5, "fetch_k": 10}
                ),
                memory=self.memory,
                return_source_documents=True
            )
            
            self.logger.info(f"Vector store initialized with {len(texts)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {str(e)}")
            return False
    
    async def _process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process and enhance MSP documents with metadata"""
        processed_docs = []
        
        for doc in documents:
            # Split document into chunks
            chunks = self.text_splitter.split_text(doc["content"])
            
            for i, chunk in enumerate(chunks):
                # Determine category
                category = self._classify_content(chunk)
                
                # Enhanced metadata
                metadata = {
                    "source": doc.get("source", "unknown"),
                    "title": doc.get("title", "Untitled"),
                    "category": category,
                    "priority": self.knowledge_categories.get(category, {}).get("priority", 3),
                    "chunk_id": f"{doc.get('id', 'doc')}_{i}",
                    "timestamp": datetime.now().isoformat(),
                    "version": doc.get("version", "1.0")
                }
                
                processed_docs.append({
                    "content": chunk,
                    "metadata": metadata
                })
        
        return processed_docs
    
    def _classify_content(self, content: str) -> str:
        """Classify content into MSP categories"""
        content_lower = content.lower()
        
        # Score each category
        category_scores = {}
        for category, info in self.knowledge_categories.items():
            score = sum(1 for keyword in info["keywords"] if keyword.lower() in content_lower)
            if score > 0:
                category_scores[category] = score * info["priority"]
        
        # Return highest scoring category
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    async def _init_pinecone_store(self, texts: List[str], metadatas: List[Dict]):
        """Initialize Pinecone vector store"""
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        
        index_name = "msp-knowledge-base"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine"
            )
        
        return Pinecone.from_texts(
            texts, self.embeddings, metadatas=metadatas, index_name=index_name
        )
    
    async def _init_chroma_store(self, texts: List[str], metadatas: List[Dict]):
        """Initialize ChromaDB vector store"""
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("msp_knowledge")
        
        # Add documents to collection
        ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = await self.embeddings.aembed_documents(texts)
        
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return collection
    
    async def query(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """Enhanced query with MSP-specific intelligence"""
        try:
            # Enhance question with MSP context
            enhanced_question = await self._enhance_question(question, context)
            
            # Query the knowledge base
            response = await self.retrieval_chain.acall({
                "question": enhanced_question
            })
            
            # Post-process response
            processed_response = await self._post_process_response(
                response, question, context
            )
            
            return processed_response
            
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            return {
                "answer": "I apologize, but I'm unable to process your request right now. Please contact our technical team for assistance.",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def _enhance_question(self, question: str, context: Dict = None) -> str:
        """Enhance question with MSP-specific context"""
        context = context or {}
        
        # Add MSP context
        msp_context = """
        You are an expert MSP (Managed Service Provider) technical consultant.
        Provide accurate, actionable advice based on MSP best practices.
        Consider compliance requirements, security implications, and business impact.
        """
        
        # Add client context if available
        if context.get("client_info"):
            client_context = f"""
            Client context: {json.dumps(context['client_info'])}
            """
            msp_context += client_context
        
        enhanced_question = f"{msp_context}\n\nClient Question: {question}"
        return enhanced_question
    
    async def _post_process_response(self, response: Dict, original_question: str, context: Dict) -> Dict[str, Any]:
        """Post-process response with additional MSP insights"""
        
        # Extract answer and sources
        answer = response.get("answer", "")
        source_documents = response.get("source_documents", [])
        
        # Calculate confidence based on source relevance
        confidence = self._calculate_confidence(answer, source_documents, original_question)
        
        # Add MSP-specific recommendations
        recommendations = await self._generate_recommendations(answer, original_question, context)
        
        # Format response
        processed_response = {
            "answer": answer,
            "confidence": confidence,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "relevance_score": self._calculate_source_relevance(doc, original_question)
                }
                for doc in source_documents[:3]
            ],
            "recommendations": recommendations,
            "category": self._classify_content(original_question),
            "timestamp": datetime.now().isoformat()
        }
        
        return processed_response
    
    def _calculate_confidence(self, answer: str, sources: List, question: str) -> float:
        """Calculate confidence score for the response"""
        base_confidence = 0.5
        
        # Boost confidence if answer is detailed
        if len(answer) > 100:
            base_confidence += 0.2
        
        # Boost confidence based on number of relevant sources
        if len(sources) >= 2:
            base_confidence += 0.2
        
        # Boost confidence if answer contains specific technical terms
        technical_terms = ["configure", "implement", "troubleshoot", "monitor", "backup"]
        if any(term in answer.lower() for term in technical_terms):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _calculate_source_relevance(self, document, question: str) -> float:
        """Calculate relevance score for a source document"""
        question_words = set(question.lower().split())
        doc_words = set(document.page_content.lower().split())
        
        # Calculate word overlap
        overlap = len(question_words.intersection(doc_words))
        relevance = overlap / len(question_words) if question_words else 0
        
        return min(relevance, 1.0)
    
    async def _generate_recommendations(self, answer: str, question: str, context: Dict) -> List[str]:
        """Generate MSP-specific recommendations"""
        recommendations = []
        
        # Category-specific recommendations
        category = self._classify_content(question)
        
        category_recommendations = {
            "network_infrastructure": [
                "Consider implementing network monitoring",
                "Review firewall rules and access controls",
                "Plan for redundancy and failover"
            ],
            "cybersecurity": [
                "Implement multi-factor authentication",
                "Regular security awareness training",
                "Schedule penetration testing"
            ],
            "cloud_services": [
                "Evaluate cloud cost optimization",
                "Implement proper backup strategies",
                "Review access management policies"
            ],
            "compliance": [
                "Document all compliance procedures",
                "Schedule regular compliance audits",
                "Implement proper data governance"
            ]
        }
        
        recommendations.extend(category_recommendations.get(category, []))
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    async def add_document(self, document: Dict) -> bool:
        """Add new document to the knowledge base"""
        try:
            # Process the document
            processed_docs = await self._process_documents([document])
            
            # Add to vector store
            texts = [doc["content"] for doc in processed_docs]
            metadatas = [doc["metadata"] for doc in processed_docs]
            
            if hasattr(self.vector_store, 'add_texts'):
                self.vector_store.add_texts(texts, metadatas=metadatas)
            
            self.logger.info(f"Document added: {document.get('title', 'Untitled')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add document: {str(e)}")
            return False
    
    async def update_document(self, document_id: str, updated_content: str) -> bool:
        """Update existing document in the knowledge base"""
        try:
            # This would require implementing document versioning
            # For now, we'll add it as a new document with updated version
            document = {
                "id": document_id,
                "content": updated_content,
                "version": "updated",
                "title": f"Updated Document {document_id}"
            }
            
            return await self.add_document(document)
            
        except Exception as e:
            self.logger.error(f"Failed to update document: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "categories": list(self.knowledge_categories.keys()),
            "embedding_model": "text-embedding-ada-002",
            "chunk_size": self.text_splitter._chunk_size,
            "chunk_overlap": self.text_splitter._chunk_overlap
        }

# Example MSP knowledge base initialization
MSP_KNOWLEDGE_DOCUMENTS = [
    {
        "id": "net_001",
        "title": "Network Security Best Practices",
        "content": """
        Network security is fundamental for MSP clients. Key practices include:
        
        1. Firewall Configuration:
        - Configure firewalls with deny-all default rules
        - Open only necessary ports and services
        - Regular review of firewall rules
        - Implement network segmentation
        
        2. Network Monitoring:
        - Deploy network monitoring tools
        - Set up alerts for unusual traffic patterns
        - Monitor bandwidth utilization
        - Track device connectivity
        
        3. Access Control:
        - Implement VLAN segmentation
        - Use 802.1X authentication where possible
        - Regular access review and cleanup
        - Document all network access points
        """,
        "source": "MSP Best Practices Guide",
        "version": "2.1"
    },
    {
        "id": "sec_001",
        "title": "Cybersecurity Framework Implementation",
        "content": """
        Comprehensive cybersecurity framework for MSP clients:
        
        1. Endpoint Protection:
        - Deploy enterprise antivirus with centralized management
        - Enable real-time protection and behavioral analysis
        - Regular signature updates and system scanning
        - Implement application whitelisting where appropriate
        
        2. Email Security:
        - Configure SPF, DKIM, and DMARC records
        - Deploy advanced threat protection
        - User training on phishing recognition
        - Implement email encryption for sensitive data
        
        3. Backup and Recovery:
        - 3-2-1 backup strategy implementation
        - Regular backup testing and verification
        - Offsite and cloud backup solutions
        - Document recovery procedures and RTO/RPO targets
        
        4. Incident Response:
        - Develop incident response procedures
        - Regular tabletop exercises
        - Maintain incident response team contacts
        - Document lessons learned and improvements
        """,
        "source": "Cybersecurity Playbook",
        "version": "3.0"
    },
    {
        "id": "cloud_001",
        "title": "Cloud Migration Strategy and Best Practices",
        "content": """
        Cloud migration strategy for MSP clients:
        
        1. Assessment Phase:
        - Inventory current infrastructure and applications
        - Assess cloud readiness and dependencies
        - Evaluate security and compliance requirements
        - Calculate cost implications and ROI
        
        2. Migration Planning:
        - Choose appropriate migration strategy (lift-and-shift, re-architecture, etc.)
        - Plan migration phases and timelines
        - Identify pilot applications and users
        - Prepare rollback procedures
        
        3. Cloud Security:
        - Implement identity and access management (IAM)
        - Configure network security groups and firewalls
        - Enable logging and monitoring
        - Implement data encryption in transit and at rest
        
        4. Post-Migration:
        - Monitor performance and costs
        - Optimize resource allocation
        - Train users on new cloud services
        - Regular security and compliance reviews
        """,
        "source": "Cloud Migration Guide",
        "version": "1.5"
    }
]

# Configuration example
rag_config = {
    "openai_api_key": "your-openai-api-key",
    "pinecone_api_key": "your-pinecone-api-key", 
    "pinecone_env": "us-east-1-aws"
}

async def initialize_msp_rag():
    """Initialize MSP RAG system with sample data"""
    rag = MSPKnowledgeBase(rag_config)
    success = await rag.initialize_vector_store(MSP_KNOWLEDGE_DOCUMENTS, "faiss")
    
    if success:
        print("MSP Knowledge Base initialized successfully!")
        return rag
    else:
        print("Failed to initialize knowledge base")
        return None

if __name__ == "__main__":
    async def test_rag():
        rag = await initialize_msp_rag()
        if rag:
            # Test query
            response = await rag.query(
                "How should we configure our firewall for better security?",
                context={"client_info": {"industry": "healthcare", "size": "50 employees"}}
            )
            print(json.dumps(response, indent=2))
    
    asyncio.run(test_rag())