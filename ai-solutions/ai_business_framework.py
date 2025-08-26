# AI-Powered Business Solutions for MSPs
# RAG Systems, AI Calling Assistants, and Chatbots

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import json
import logging
from datetime import datetime
import openai
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import twilio
from twilio.rest import Client as TwilioClient
import speech_recognition as sr
import pyttsx3
import pandas as pd
import requests

class RAGSystem:
    """Retrieval-Augmented Generation System for MSP Knowledge Base"""
    
    def __init__(self, api_key: str, knowledge_base_path: str):
        self.api_key = api_key
        self.knowledge_base_path = knowledge_base_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        self.retrieval_qa = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize_knowledge_base(self, documents: List[str]):
        """Initialize the knowledge base with MSP-specific documentation"""
        try:
            # Split documents into chunks
            text_chunks = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc)
                text_chunks.extend(chunks)
            
            # Create embeddings and vector store
            self.vector_store = FAISS.from_texts(
                text_chunks, 
                self.embeddings
            )
            
            # Create retrieval QA chain
            llm = OpenAI(openai_api_key=self.api_key, temperature=0.7)
            self.retrieval_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
            )
            
            self.logger.info("Knowledge base initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge base: {str(e)}")
            return False
    
    async def query_knowledge_base(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Query the RAG system with context awareness"""
        try:
            if not self.retrieval_qa:
                raise ValueError("Knowledge base not initialized")
            
            # Enhanced query with MSP context
            enhanced_query = f"""
            Context: You are an expert MSP (Managed Service Provider) assistant.
            Client Query: {query}
            
            Please provide a helpful, accurate response based on MSP best practices.
            Include relevant technical details and actionable recommendations.
            """
            
            response = await self.retrieval_qa.arun(enhanced_query)
            
            return {
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "confidence": self._calculate_confidence(query, response),
                "sources": self._extract_sources(query)
            }
            
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            return {
                "query": query,
                "response": "I apologize, but I'm unable to process your request at the moment. Please contact our technical team.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_confidence(self, query: str, response: str) -> float:
        """Calculate confidence score based on response quality"""
        # Simple confidence calculation - can be enhanced with ML models
        if len(response) > 100 and "I don't know" not in response.lower():
            return 0.85
        elif len(response) > 50:
            return 0.65
        else:
            return 0.45
    
    def _extract_sources(self, query: str) -> List[str]:
        """Extract relevant source documents"""
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=3)
            return [doc.page_content[:100] + "..." for doc in docs]
        return []

class AICallingAssistant:
    """AI-powered calling assistant for MSP customer service"""
    
    def __init__(self, twilio_sid: str, twilio_token: str, openai_api_key: str):
        self.twilio_client = TwilioClient(twilio_sid, twilio_token)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.speech_recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.call_log = []
        self.logger = logging.getLogger(__name__)
        
        # Configure TTS settings
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.8)
    
    async def initiate_outbound_call(self, 
                                   phone_number: str, 
                                   campaign_type: str, 
                                   client_data: Dict = None) -> Dict[str, Any]:
        """Initiate AI-powered outbound calling campaign"""
        try:
            # Generate personalized script
            script = await self._generate_call_script(campaign_type, client_data)
            
            # Initiate call via Twilio
            call = self.twilio_client.calls.create(
                to=phone_number,
                from_='+1234567890',  # Your Twilio number
                url='https://your-server.com/ai-call-handler',
                method='POST'
            )
            
            call_record = {
                "call_sid": call.sid,
                "phone_number": phone_number,
                "campaign_type": campaign_type,
                "script": script,
                "timestamp": datetime.now().isoformat(),
                "status": "initiated",
                "client_data": client_data
            }
            
            self.call_log.append(call_record)
            self.logger.info(f"Call initiated: {call.sid}")
            
            return call_record
            
        except Exception as e:
            self.logger.error(f"Failed to initiate call: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def _generate_call_script(self, campaign_type: str, client_data: Dict = None) -> str:
        """Generate personalized call script using AI"""
        
        prompts = {
            "new_prospect": """
            Create a professional, friendly cold calling script for an MSP reaching out to a new prospect.
            Keep it under 30 seconds for the opening.
            Focus on: IT security, cost reduction, productivity improvements.
            """,
            "existing_client": """
            Create a warm calling script for checking in with an existing MSP client.
            Focus on: service satisfaction, additional needs, upcoming projects.
            """,
            "follow_up": """
            Create a follow-up script for a prospect who showed interest.
            Focus on: addressing concerns, scheduling technical assessment.
            """,
            "renewal": """
            Create a renewal discussion script for contract renewals.
            Focus on: value delivered, service improvements, future roadmap.
            """
        }
        
        system_prompt = f"""
        You are an expert MSP sales representative. Generate a natural, conversational script.
        Client data: {json.dumps(client_data) if client_data else 'No specific data'}
        
        {prompts.get(campaign_type, prompts['new_prospect'])}
        """
        
        response = await self.openai_client.chat.completions.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a {campaign_type} calling script"}
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    async def handle_inbound_call(self, call_sid: str, caller_info: Dict) -> Dict[str, Any]:
        """Handle inbound customer service calls with AI"""
        try:
            # Identify caller and retrieve context
            caller_context = await self._get_caller_context(caller_info.get('phone_number'))
            
            # Generate appropriate greeting
            greeting = await self._generate_greeting(caller_context)
            
            call_handler = {
                "call_sid": call_sid,
                "caller_info": caller_info,
                "context": caller_context,
                "greeting": greeting,
                "timestamp": datetime.now().isoformat(),
                "interaction_log": []
            }
            
            self.call_log.append(call_handler)
            return call_handler
            
        except Exception as e:
            self.logger.error(f"Failed to handle inbound call: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def _get_caller_context(self, phone_number: str) -> Dict[str, Any]:
        """Retrieve caller context from CRM/database"""
        # This would integrate with your CRM system
        # Mock implementation
        return {
            "is_client": True,
            "client_name": "ABC Company",
            "account_manager": "John Smith",
            "last_ticket": "Network connectivity issue - resolved",
            "contract_status": "active",
            "priority_level": "standard"
        }
    
    async def _generate_greeting(self, caller_context: Dict) -> str:
        """Generate personalized greeting based on caller context"""
        if caller_context.get('is_client'):
            return f"Hello! Thank you for calling [MSP Name]. I see you're calling from {caller_context.get('client_name')}. How can I assist you today?"
        else:
            return "Hello! Thank you for calling [MSP Name]. We're here to help with all your IT needs. How can I assist you today?"

class IntelligentChatbot:
    """Advanced chatbot for MSP customer service and lead generation"""
    
    def __init__(self, openai_api_key: str, rag_system: RAGSystem):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.rag_system = rag_system
        self.conversation_history = {}
        self.lead_qualification_rules = {
            "company_size": ["How many employees", "size of your business"],
            "current_it_setup": ["current IT setup", "who manages your IT"],
            "pain_points": ["IT challenges", "problems with", "frustrated with"],
            "budget": ["budget", "investment", "spend on IT"],
            "timeline": ["when", "urgently", "timeline"]
        }
        self.logger = logging.getLogger(__name__)
    
    async def process_message(self, 
                            user_id: str, 
                            message: str, 
                            channel: str = "web") -> Dict[str, Any]:
        """Process incoming chat message with AI intelligence"""
        try:
            # Initialize conversation history
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = {
                    "messages": [],
                    "lead_score": 0,
                    "qualified_data": {},
                    "intent": "unknown",
                    "channel": channel
                }
            
            # Add user message to history
            self.conversation_history[user_id]["messages"].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Analyze intent and generate response
            intent_analysis = await self._analyze_intent(message)
            response = await self._generate_response(user_id, message, intent_analysis)
            
            # Update lead qualification
            await self._update_lead_qualification(user_id, message)
            
            # Add bot response to history
            self.conversation_history[user_id]["messages"].append({
                "role": "assistant",
                "content": response["content"],
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "response": response["content"],
                "intent": intent_analysis,
                "lead_score": self.conversation_history[user_id]["lead_score"],
                "qualified_data": self.conversation_history[user_id]["qualified_data"],
                "suggested_actions": response.get("suggested_actions", [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process message: {str(e)}")
            return {
                "response": "I apologize for the inconvenience. Let me connect you with a human agent who can better assist you.",
                "error": str(e)
            }
    
    async def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user intent using AI"""
        system_prompt = """
        Analyze the user's message and identify their intent. Possible intents:
        - support_request: User needs technical help
        - sales_inquiry: User is interested in services
        - information_seeking: User wants to learn about services
        - complaint: User has an issue to report
        - compliment: User is providing positive feedback
        - pricing: User wants pricing information
        - demo_request: User wants a demo or trial
        
        Return JSON with intent, confidence, and key entities.
        """
        
        response = await self.openai_client.chat.completions.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=200
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"intent": "unknown", "confidence": 0.5, "entities": []}
    
    async def _generate_response(self, user_id: str, message: str, intent: Dict) -> Dict[str, Any]:
        """Generate intelligent response based on intent and context"""
        
        # Get conversation context
        context = self.conversation_history[user_id]
        recent_messages = context["messages"][-5:]  # Last 5 messages
        
        # Use RAG for technical questions
        if intent.get("intent") in ["support_request", "information_seeking"]:
            rag_response = await self.rag_system.query_knowledge_base(message)
            base_response = rag_response["response"]
        else:
            base_response = await self._generate_contextual_response(message, intent, context)
        
        # Add personalization and MSP-specific enhancements
        enhanced_response = await self._enhance_response(base_response, intent, context)
        
        return {
            "content": enhanced_response,
            "suggested_actions": self._get_suggested_actions(intent, context)
        }
    
    async def _generate_contextual_response(self, message: str, intent: Dict, context: Dict) -> str:
        """Generate contextual response using AI"""
        system_prompt = f"""
        You are an expert MSP customer service representative. Respond professionally and helpfully.
        
        User intent: {intent.get('intent', 'unknown')}
        Conversation context: User has {len(context['messages'])} previous messages.
        Lead score: {context['lead_score']}/100
        
        Provide a helpful, professional response that:
        1. Addresses the user's specific need
        2. Demonstrates MSP expertise
        3. Guides toward appropriate next steps
        4. Maintains a friendly, professional tone
        """
        
        response = await self.openai_client.chat.completions.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    async def _enhance_response(self, base_response: str, intent: Dict, context: Dict) -> str:
        """Enhance response with MSP-specific value propositions"""
        
        enhancements = {
            "sales_inquiry": "\n\nWould you like to schedule a free IT assessment to see how we can help optimize your technology infrastructure?",
            "pricing": "\n\nOur pricing is customized based on your specific needs. I'd be happy to schedule a quick consultation to provide accurate pricing.",
            "support_request": "\n\nIf you need immediate assistance, I can connect you with our technical team right away.",
            "demo_request": "\n\nI can schedule a personalized demo that shows exactly how our services would benefit your business."
        }
        
        enhancement = enhancements.get(intent.get("intent"), "")
        return base_response + enhancement
    
    async def _update_lead_qualification(self, user_id: str, message: str):
        """Update lead qualification score based on message content"""
        context = self.conversation_history[user_id]
        
        # Check for qualification indicators
        for category, keywords in self.lead_qualification_rules.items():
            for keyword in keywords:
                if keyword.lower() in message.lower():
                    context["qualified_data"][category] = True
                    context["lead_score"] += 10
        
        # Cap lead score at 100
        context["lead_score"] = min(context["lead_score"], 100)
    
    def _get_suggested_actions(self, intent: Dict, context: Dict) -> List[str]:
        """Get suggested follow-up actions"""
        actions = []
        
        if intent.get("intent") == "sales_inquiry" and context["lead_score"] > 50:
            actions.append("schedule_consultation")
            actions.append("send_case_studies")
        
        if intent.get("intent") == "support_request":
            actions.append("escalate_to_tech")
            actions.append("create_ticket")
        
        if context["lead_score"] > 70:
            actions.append("assign_account_manager")
        
        return actions

class BusinessIntelligenceAnalyzer:
    """Analyze conversations and calls for business intelligence"""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.analytics_data = {
            "call_analytics": [],
            "chat_analytics": [],
            "lead_insights": [],
            "service_requests": []
        }
    
    async def analyze_conversation_trends(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Analyze conversation trends for business insights"""
        
        # Common pain points analysis
        pain_points = await self._extract_pain_points(conversations)
        
        # Lead quality analysis
        lead_quality = await self._analyze_lead_quality(conversations)
        
        # Service demand analysis
        service_demand = await self._analyze_service_demand(conversations)
        
        return {
            "pain_points": pain_points,
            "lead_quality": lead_quality,
            "service_demand": service_demand,
            "recommendations": await self._generate_business_recommendations(pain_points, service_demand)
        }
    
    async def _extract_pain_points(self, conversations: List[Dict]) -> Dict[str, int]:
        """Extract and count common pain points"""
        pain_point_keywords = {
            "security_concerns": ["hack", "breach", "security", "virus", "malware"],
            "downtime_issues": ["down", "outage", "offline", "not working"],
            "slow_performance": ["slow", "sluggish", "performance", "speed"],
            "cost_concerns": ["expensive", "cost", "budget", "save money"],
            "compliance_issues": ["compliance", "audit", "regulations", "GDPR", "HIPAA"]
        }
        
        pain_points = {key: 0 for key in pain_point_keywords.keys()}
        
        for conv in conversations:
            for message in conv.get("messages", []):
                content = message.get("content", "").lower()
                for category, keywords in pain_point_keywords.items():
                    for keyword in keywords:
                        if keyword in content:
                            pain_points[category] += 1
        
        return pain_points
    
    async def _analyze_lead_quality(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Analyze lead quality metrics"""
        total_leads = len(conversations)
        qualified_leads = len([c for c in conversations if c.get("lead_score", 0) > 50])
        high_intent_leads = len([c for c in conversations if c.get("lead_score", 0) > 75])
        
        return {
            "total_leads": total_leads,
            "qualified_leads": qualified_leads,
            "high_intent_leads": high_intent_leads,
            "qualification_rate": qualified_leads / total_leads if total_leads > 0 else 0,
            "high_intent_rate": high_intent_leads / total_leads if total_leads > 0 else 0
        }
    
    async def _analyze_service_demand(self, conversations: List[Dict]) -> Dict[str, int]:
        """Analyze demand for specific MSP services"""
        service_keywords = {
            "cloud_migration": ["cloud", "migration", "AWS", "Azure", "Google Cloud"],
            "cybersecurity": ["security", "firewall", "antivirus", "backup"],
            "managed_services": ["monitoring", "maintenance", "support"],
            "consulting": ["strategy", "planning", "assessment", "consultation"],
            "infrastructure": ["network", "server", "hardware", "infrastructure"]
        }
        
        service_demand = {key: 0 for key in service_keywords.keys()}
        
        for conv in conversations:
            for message in conv.get("messages", []):
                content = message.get("content", "").lower()
                for service, keywords in service_keywords.items():
                    for keyword in keywords:
                        if keyword in content:
                            service_demand[service] += 1
        
        return service_demand
    
    async def _generate_business_recommendations(self, pain_points: Dict, service_demand: Dict) -> List[str]:
        """Generate business recommendations based on analysis"""
        recommendations = []
        
        # Top pain points
        top_pain_point = max(pain_points.items(), key=lambda x: x[1])
        if top_pain_point[1] > 0:
            recommendations.append(f"Focus marketing on {top_pain_point[0].replace('_', ' ')} solutions")
        
        # Service demand insights
        top_service = max(service_demand.items(), key=lambda x: x[1])
        if top_service[1] > 0:
            recommendations.append(f"Expand {top_service[0].replace('_', ' ')} service offerings")
        
        return recommendations

# Example usage and integration
if __name__ == "__main__":
    async def main():
        # Initialize systems
        rag_system = RAGSystem("your-openai-key", "/path/to/knowledge/base")
        ai_caller = AICallingAssistant("twilio-sid", "twilio-token", "openai-key")
        chatbot = IntelligentChatbot("openai-key", rag_system)
        analytics = BusinessIntelligenceAnalyzer("openai-key")
        
        # Sample MSP knowledge base
        msp_documents = [
            "Network monitoring best practices for small businesses...",
            "Cybersecurity frameworks for compliance requirements...",
            "Cloud migration strategies and implementation...",
            "Disaster recovery planning and testing procedures..."
        ]
        
        await rag_system.initialize_knowledge_base(msp_documents)
        
        # Example chatbot interaction
        response = await chatbot.process_message(
            user_id="client_001",
            message="Our network has been running slowly and we're concerned about security.",
            channel="web"
        )
        
        print("Chatbot Response:", response)
        
        # Example outbound calling
        call_result = await ai_caller.initiate_outbound_call(
            phone_number="+1234567890",
            campaign_type="new_prospect",
            client_data={"company": "ABC Corp", "industry": "Healthcare"}
        )
        
        print("Call Result:", call_result)

    # Run the example
    # asyncio.run(main())