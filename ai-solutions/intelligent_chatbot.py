# Intelligent Chatbot for MSP Customer Support and Sales
# Multi-channel AI chatbot with CRM integration and lead qualification

from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import json
import os
import re
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import websockets
from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
import openai
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import uuid

class ConversationState(Enum):
    GREETING = "greeting"
    QUALIFICATION = "qualification"
    TECHNICAL_SUPPORT = "technical_support"
    SALES_INQUIRY = "sales_inquiry"
    ESCALATION = "escalation"
    CLOSING = "closing"
    COMPLETED = "completed"

class LeadQuality(Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    NOT_QUALIFIED = "not_qualified"

@dataclass
class ChatSession:
    """Structure for chat session management"""
    session_id: str
    client_id: Optional[str]
    client_name: Optional[str]
    client_email: Optional[str]
    client_company: Optional[str]
    channel: str  # web, teams, slack, sms, etc.
    state: ConversationState
    lead_quality: LeadQuality
    conversation_history: List[Dict[str, str]]
    context_data: Dict[str, Any]
    start_time: datetime
    last_activity: datetime
    escalation_requested: bool
    satisfaction_score: Optional[float]
    qualified_services: List[str]
    budget_range: Optional[str]
    decision_timeframe: Optional[str]
    pain_points: List[str]
    next_steps: List[str]

@dataclass
class ChatMetrics:
    """Structure for chat analytics and metrics"""
    session_id: str
    total_messages: int
    avg_response_time: float
    resolution_time: Optional[float]
    escalation_count: int
    satisfaction_score: Optional[float]
    lead_converted: bool
    services_discussed: List[str]
    intent_classification: Dict[str, float]

class MSPIntelligentChatbot:
    """Advanced AI chatbot for MSP customer support and sales"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_client = OpenAI(api_key=config.get('openai_api_key'))
        
        # Session management
        self.active_sessions: Dict[str, ChatSession] = {}
        self.session_metrics: Dict[str, ChatMetrics] = {}
        
        # Knowledge base and models
        self.intent_classifier = None
        self.vectorizer = None
        self._load_models()
        
        # MSP-specific knowledge base
        self.msp_services = {
            "managed_it": {
                "description": "Complete IT infrastructure management and monitoring",
                "benefits": ["24/7 monitoring", "proactive maintenance", "reduced downtime"],
                "price_range": "$150-500 per user/month",
                "qualification_questions": [
                    "How many employees do you have?",
                    "What's your current IT setup?",
                    "Have you experienced IT downtime issues?"
                ]
            },
            "cybersecurity": {
                "description": "Comprehensive cybersecurity protection and monitoring",
                "benefits": ["threat detection", "compliance management", "employee training"],
                "price_range": "$100-300 per user/month",
                "qualification_questions": [
                    "Have you experienced any security incidents?",
                    "What industry are you in?",
                    "Do you handle sensitive customer data?"
                ]
            },
            "cloud_migration": {
                "description": "Safe and efficient migration to cloud platforms",
                "benefits": ["scalability", "cost savings", "remote access"],
                "price_range": "$5,000-50,000 project cost",
                "qualification_questions": [
                    "What systems are you looking to migrate?",
                    "What's your migration timeline?",
                    "What's your budget for this project?"
                ]
            },
            "backup_disaster_recovery": {
                "description": "Comprehensive data backup and disaster recovery solutions",
                "benefits": ["data protection", "business continuity", "compliance"],
                "price_range": "$50-200 per user/month",
                "qualification_questions": [
                    "How critical is your data?",
                    "What's your acceptable downtime?",
                    "Do you have current backup systems?"
                ]
            },
            "help_desk": {
                "description": "24/7 IT support and help desk services",
                "benefits": ["rapid response", "expert technicians", "user training"],
                "price_range": "$25-100 per user/month",
                "qualification_questions": [
                    "How many support tickets do you typically have?",
                    "What are your support hours requirements?",
                    "What types of issues are most common?"
                ]
            }
        }
        
        # Common MSP pain points and solutions
        self.pain_point_solutions = {
            "downtime": {
                "services": ["managed_it", "backup_disaster_recovery"],
                "message": "I understand downtime can be costly. Our managed IT services include 24/7 monitoring to prevent issues before they impact your business."
            },
            "security": {
                "services": ["cybersecurity", "managed_it"],
                "message": "Security is crucial today. Our cybersecurity solutions provide comprehensive protection with continuous monitoring and threat detection."
            },
            "costs": {
                "services": ["managed_it", "cloud_migration"],
                "message": "Cost control is important. Our managed services often reduce IT costs by 20-40% through proactive maintenance and cloud optimization."
            },
            "scalability": {
                "services": ["cloud_migration", "managed_it"],
                "message": "Growing businesses need scalable solutions. Our cloud migration services ensure your IT grows with your business seamlessly."
            },
            "compliance": {
                "services": ["cybersecurity", "backup_disaster_recovery"],
                "message": "Compliance can be complex. Our solutions ensure you meet industry regulations with automated reporting and documentation."
            }
        }
        
        # Conversation templates
        self.response_templates = {
            "greeting": [
                "Hello! Welcome to {company_name}. I'm here to help with your IT needs. How can I assist you today?",
                "Hi there! I'm your AI assistant from {company_name}. What can I help you with regarding your IT infrastructure?",
                "Welcome! I'm here to help you explore how {company_name} can improve your IT operations. What's your biggest IT challenge?"
            ],
            "qualification": [
                "To better understand your needs, could you tell me about your current IT setup?",
                "What industry is your business in? This helps me recommend the most relevant solutions.",
                "How many employees do you have? This helps me suggest appropriately scaled solutions."
            ],
            "technical_support": [
                "I'd be happy to help with your technical issue. Can you describe what's happening?",
                "Let me gather some information to help resolve this quickly. What specific problem are you experiencing?",
                "I'm here to help troubleshoot. What system or application is causing difficulties?"
            ],
            "sales_inquiry": [
                "I'd love to discuss our services. What specific IT challenges is your business facing?",
                "Great question about our services. What's most important to your business - security, reliability, or cost savings?",
                "I can help you find the right solution. What prompted you to look into managed IT services?"
            ]
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _load_models(self):
        """Load or initialize machine learning models"""
        try:
            # Try to load existing models
            with open('intent_classifier.pkl', 'rb') as f:
                self.intent_classifier = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.logger.info("Loaded existing ML models")
        except FileNotFoundError:
            # Initialize and train new models
            self._train_intent_classifier()
            self.logger.info("Initialized new ML models")
    
    def _train_intent_classifier(self):
        """Train intent classification model with MSP-specific data"""
        training_data = [
            ("I need help with my computer", "technical_support"),
            ("My email isn't working", "technical_support"),
            ("The server is down", "technical_support"),
            ("Can't access the network", "technical_support"),
            ("Password reset needed", "technical_support"),
            
            ("What services do you offer", "sales_inquiry"),
            ("How much does managed IT cost", "sales_inquiry"),
            ("Tell me about cybersecurity", "sales_inquiry"),
            ("Do you provide cloud services", "sales_inquiry"),
            ("I'm interested in your backup solutions", "sales_inquiry"),
            
            ("Hello", "greeting"),
            ("Hi there", "greeting"),
            ("Good morning", "greeting"),
            ("Hey", "greeting"),
            
            ("Thank you", "closing"),
            ("That's all I needed", "closing"),
            ("Goodbye", "closing"),
            ("Have a great day", "closing"),
            
            ("I need to speak to a human", "escalation"),
            ("Transfer me to someone", "escalation"),
            ("This isn't helping", "escalation"),
            ("I want to talk to a manager", "escalation")
        ]
        
        texts, labels = zip(*training_data)
        
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = self.vectorizer.fit_transform(texts)
        
        self.intent_classifier = MultinomialNB()
        self.intent_classifier.fit(X, labels)
        
        # Save models
        with open('intent_classifier.pkl', 'wb') as f:
            pickle.dump(self.intent_classifier, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    async def create_session(self, channel: str, client_info: Dict = None) -> str:
        """Create new chat session"""
        session_id = str(uuid.uuid4())
        
        session = ChatSession(
            session_id=session_id,
            client_id=client_info.get('client_id') if client_info else None,
            client_name=client_info.get('name') if client_info else None,
            client_email=client_info.get('email') if client_info else None,
            client_company=client_info.get('company') if client_info else None,
            channel=channel,
            state=ConversationState.GREETING,
            lead_quality=LeadQuality.NOT_QUALIFIED,
            conversation_history=[],
            context_data={},
            start_time=datetime.now(),
            last_activity=datetime.now(),
            escalation_requested=False,
            satisfaction_score=None,
            qualified_services=[],
            budget_range=None,
            decision_timeframe=None,
            pain_points=[],
            next_steps=[]
        )
        
        self.active_sessions[session_id] = session
        
        # Initialize metrics
        self.session_metrics[session_id] = ChatMetrics(
            session_id=session_id,
            total_messages=0,
            avg_response_time=0.0,
            resolution_time=None,
            escalation_count=0,
            satisfaction_score=None,
            lead_converted=False,
            services_discussed=[],
            intent_classification={}
        )
        
        self.logger.info(f"Created chat session: {session_id}")
        return session_id
    
    async def process_message(self, session_id: str, message: str, message_type: str = "user") -> Dict[str, Any]:
        """Process incoming message and generate response"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.last_activity = datetime.now()
        
        # Add message to history
        session.conversation_history.append({
            "type": message_type,
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update metrics
        metrics = self.session_metrics[session_id]
        metrics.total_messages += 1
        
        if message_type == "user":
            # Classify intent
            intent = await self._classify_intent(message)
            metrics.intent_classification[intent] = metrics.intent_classification.get(intent, 0) + 1
            
            # Update conversation state
            session.state = await self._update_conversation_state(session, message, intent)
            
            # Extract information
            await self._extract_client_information(session, message)
            
            # Generate response
            response = await self._generate_response(session, message, intent)
            
            # Add response to history
            session.conversation_history.append({
                "type": "bot",
                "content": response["message"],
                "timestamp": datetime.now().isoformat()
            })
            
            return response
        
        return {"message": "Message recorded", "action": "none"}
    
    async def _classify_intent(self, message: str) -> str:
        """Classify user intent using trained model"""
        if not self.intent_classifier or not self.vectorizer:
            return "unknown"
        
        try:
            message_vector = self.vectorizer.transform([message.lower()])
            intent = self.intent_classifier.predict(message_vector)[0]
            confidence = max(self.intent_classifier.predict_proba(message_vector)[0])
            
            # If confidence is low, use OpenAI for better classification
            if confidence < 0.6:
                intent = await self._classify_intent_with_ai(message)
            
            return intent
            
        except Exception as e:
            self.logger.error(f"Intent classification failed: {str(e)}")
            return "unknown"
    
    async def _classify_intent_with_ai(self, message: str) -> str:
        """Use OpenAI for advanced intent classification"""
        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Classify the user's intent from these categories:
                        - greeting: general greetings and hellos
                        - technical_support: technical issues, problems, troubleshooting
                        - sales_inquiry: questions about services, pricing, capabilities  
                        - qualification: providing business information
                        - escalation: wanting human support
                        - closing: ending conversation, thanks
                        - unknown: unclear intent
                        
                        Return only the category name."""
                    },
                    {
                        "role": "user", 
                        "content": message
                    }
                ],
                temperature=0.1,
                max_tokens=20
            )
            
            return response.choices[0].message.content.strip().lower()
            
        except Exception as e:
            self.logger.error(f"AI intent classification failed: {str(e)}")
            return "unknown"
    
    async def _update_conversation_state(self, session: ChatSession, message: str, intent: str) -> ConversationState:
        """Update conversation state based on intent and context"""
        
        # State transition logic
        if intent == "greeting" and session.state == ConversationState.GREETING:
            return ConversationState.QUALIFICATION
        
        elif intent == "technical_support":
            return ConversationState.TECHNICAL_SUPPORT
        
        elif intent == "sales_inquiry":
            return ConversationState.SALES_INQUIRY
        
        elif intent == "escalation":
            session.escalation_requested = True
            return ConversationState.ESCALATION
        
        elif intent == "closing":
            return ConversationState.CLOSING
        
        # Continue in current state for qualification and ongoing conversations
        return session.state
    
    async def _extract_client_information(self, session: ChatSession, message: str):
        """Extract client information from message"""
        message_lower = message.lower()
        
        # Extract company size
        numbers = re.findall(r'\b(\d+)\s*(?:employees?|people|staff|users?)\b', message_lower)
        if numbers:
            session.context_data['company_size'] = int(numbers[0])
        
        # Extract industry keywords
        industries = {
            'healthcare': ['healthcare', 'medical', 'hospital', 'clinic', 'doctor'],
            'finance': ['bank', 'financial', 'insurance', 'credit', 'investment'],
            'legal': ['law', 'legal', 'attorney', 'lawyer', 'court'],
            'manufacturing': ['manufacturing', 'factory', 'production', 'industrial'],
            'retail': ['retail', 'store', 'shop', 'sales', 'customer'],
            'education': ['school', 'education', 'university', 'college', 'student']
        }
        
        for industry, keywords in industries.items():
            if any(keyword in message_lower for keyword in keywords):
                session.context_data['industry'] = industry
                break
        
        # Extract pain points
        pain_keywords = {
            'downtime': ['down', 'outage', 'offline', 'not working', 'broken'],
            'security': ['security', 'breach', 'hack', 'virus', 'malware', 'attack'],
            'costs': ['expensive', 'cost', 'budget', 'money', 'price', 'cheaper'],
            'scalability': ['grow', 'scale', 'expand', 'bigger', 'more users'],
            'compliance': ['compliance', 'regulation', 'audit', 'requirement']
        }
        
        for pain_point, keywords in pain_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                if pain_point not in session.pain_points:
                    session.pain_points.append(pain_point)
        
        # Extract budget information
        budget_patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*)\s*dollars?',
            r'budget.*?(\d+(?:,\d{3})*)'
        ]
        
        for pattern in budget_patterns:
            matches = re.findall(pattern, message_lower)
            if matches:
                session.budget_range = matches[0]
                break
    
    async def _generate_response(self, session: ChatSession, message: str, intent: str) -> Dict[str, Any]:
        """Generate intelligent response based on context"""
        
        response_data = {
            "message": "",
            "action": "continue",
            "suggestions": [],
            "lead_quality": session.lead_quality.value,
            "next_steps": []
        }
        
        try:
            if session.state == ConversationState.GREETING:
                response_data["message"] = await self._generate_greeting(session)
                response_data["suggestions"] = [
                    "I'm having technical issues",
                    "Tell me about your services", 
                    "I'm interested in cybersecurity"
                ]
            
            elif session.state == ConversationState.QUALIFICATION:
                response_data = await self._handle_qualification(session, message)
            
            elif session.state == ConversationState.TECHNICAL_SUPPORT:
                response_data = await self._handle_technical_support(session, message)
            
            elif session.state == ConversationState.SALES_INQUIRY:
                response_data = await self._handle_sales_inquiry(session, message)
            
            elif session.state == ConversationState.ESCALATION:
                response_data = await self._handle_escalation(session, message)
            
            elif session.state == ConversationState.CLOSING:
                response_data = await self._handle_closing(session, message)
            
            # Update lead quality
            session.lead_quality = await self._assess_lead_quality(session)
            response_data["lead_quality"] = session.lead_quality.value
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            return {
                "message": "I apologize, but I'm having trouble processing your request. Let me connect you with a human representative.",
                "action": "escalate",
                "suggestions": [],
                "lead_quality": session.lead_quality.value
            }
    
    async def _generate_greeting(self, session: ChatSession) -> str:
        """Generate personalized greeting"""
        company_name = self.config.get('company_name', 'our IT services company')
        
        if session.client_name:
            return f"Hello {session.client_name}! Welcome back to {company_name}. How can I help you today?"
        
        # Random greeting from templates
        import random
        template = random.choice(self.response_templates["greeting"])
        return template.format(company_name=company_name)
    
    async def _handle_qualification(self, session: ChatSession, message: str) -> Dict[str, Any]:
        """Handle lead qualification conversation"""
        
        # Check if we have basic info
        has_company_size = 'company_size' in session.context_data
        has_industry = 'industry' in session.context_data
        has_pain_points = len(session.pain_points) > 0
        
        if not has_company_size:
            return {
                "message": "To better understand your needs, how many employees does your company have?",
                "action": "continue",
                "suggestions": ["1-10 employees", "11-50 employees", "51-200 employees", "200+ employees"]
            }
        
        elif not has_industry:
            return {
                "message": "What industry is your business in? This helps me recommend the most relevant IT solutions.",
                "action": "continue", 
                "suggestions": ["Healthcare", "Finance", "Legal", "Manufacturing", "Retail", "Other"]
            }
        
        elif not has_pain_points:
            return {
                "message": "What's your biggest IT challenge right now? Understanding this helps me suggest the best solutions.",
                "action": "continue",
                "suggestions": ["Frequent downtime", "Security concerns", "High IT costs", "Need to scale", "Compliance requirements"]
            }
        
        else:
            # We have enough info, transition to recommendations
            recommendations = await self._generate_service_recommendations(session)
            session.state = ConversationState.SALES_INQUIRY
            
            return {
                "message": f"Based on what you've told me, I can see how we might help. {recommendations}",
                "action": "continue",
                "suggestions": ["Tell me more about pricing", "How quickly can you help?", "Can I speak to someone?"]
            }
    
    async def _handle_technical_support(self, session: ChatSession, message: str) -> Dict[str, Any]:
        """Handle technical support conversation"""
        
        # Use OpenAI to provide intelligent technical guidance
        context = f"""
        Client message: {message}
        Company size: {session.context_data.get('company_size', 'unknown')}
        Industry: {session.context_data.get('industry', 'unknown')}
        Previous conversation: {session.conversation_history[-3:] if len(session.conversation_history) > 3 else []}
        """
        
        response = await self.openai_client.chat.completions.acreate(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful IT support specialist for an MSP. 
                    Provide helpful troubleshooting guidance while being professional and empathetic.
                    If the issue seems complex, suggest escalation to human support.
                    Keep responses under 150 words."""
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        ai_response = response.choices[0].message.content
        
        # Determine if escalation is needed
        escalation_keywords = ['complex', 'server', 'network', 'critical', 'urgent', 'security']
        needs_escalation = any(keyword in message.lower() for keyword in escalation_keywords)
        
        if needs_escalation:
            ai_response += "\n\nThis seems like it might need hands-on assistance. Would you like me to connect you with one of our technical specialists?"
            
        return {
            "message": ai_response,
            "action": "escalate" if needs_escalation else "continue",
            "suggestions": ["Yes, connect me with someone", "Let me try that first", "What else can you help with?"]
        }
    
    async def _handle_sales_inquiry(self, session: ChatSession, message: str) -> Dict[str, Any]:
        """Handle sales and service inquiries"""
        
        # Identify mentioned services
        mentioned_services = []
        for service_key, service_info in self.msp_services.items():
            if any(keyword in message.lower() for keyword in service_key.split('_')):
                mentioned_services.append(service_key)
                if service_key not in session.qualified_services:
                    session.qualified_services.append(service_key)
        
        if mentioned_services:
            service = mentioned_services[0]
            service_info = self.msp_services[service]
            
            response_message = f"""Great question about our {service.replace('_', ' ')} services! 

{service_info['description']}

Key benefits include:
{chr(10).join(f'• {benefit}' for benefit in service_info['benefits'])}

Typical investment range: {service_info['price_range']}

Would you like to discuss how this could specifically help your business?"""
            
            return {
                "message": response_message,
                "action": "continue",
                "suggestions": ["Yes, let's discuss specifics", "What about pricing?", "Can I see a demo?"]
            }
        
        else:
            # General sales inquiry
            recommendations = await self._generate_service_recommendations(session)
            return {
                "message": f"I'd be happy to help you explore our services. {recommendations}",
                "action": "continue", 
                "suggestions": ["Tell me about managed IT", "What about cybersecurity?", "Cloud migration options"]
            }
    
    async def _generate_service_recommendations(self, session: ChatSession) -> str:
        """Generate personalized service recommendations"""
        
        company_size = session.context_data.get('company_size', 0)
        industry = session.context_data.get('industry', 'general')
        pain_points = session.pain_points
        
        recommendations = []
        
        # Size-based recommendations
        if company_size < 20:
            recommendations.append("For a company your size, our Essential Managed IT package provides great value with 24/7 monitoring and support.")
        elif company_size < 100:
            recommendations.append("With your company size, our Professional Managed IT services would include dedicated support and advanced security.")
        else:
            recommendations.append("For an enterprise your size, our Premium Managed IT services include custom solutions and dedicated account management.")
        
        # Pain point-based recommendations
        for pain_point in pain_points[:2]:  # Top 2 pain points
            if pain_point in self.pain_point_solutions:
                solution = self.pain_point_solutions[pain_point]
                recommendations.append(solution["message"])
        
        return " ".join(recommendations[:2])  # Limit to 2 recommendations
    
    async def _handle_escalation(self, session: ChatSession, message: str) -> Dict[str, Any]:
        """Handle escalation to human support"""
        
        session.escalation_requested = True
        self.session_metrics[session.session_id].escalation_count += 1
        
        return {
            "message": """I understand you'd like to speak with someone from our team. I'm connecting you now with a technical specialist who can provide personalized assistance.

While you wait, here's your conversation summary that I'll share with them:
• Your main concerns: {pain_points}
• Services discussed: {services}
• Company size: {size} employees

Expected wait time: 2-3 minutes. Is there anything else I can help clarify before the transfer?""".format(
                pain_points=", ".join(session.pain_points) if session.pain_points else "General inquiry",
                services=", ".join(session.qualified_services) if session.qualified_services else "None yet",
                size=session.context_data.get('company_size', 'Unknown')
            ),
            "action": "escalate",
            "next_steps": ["Human agent will connect shortly"]
        }
    
    async def _handle_closing(self, session: ChatSession, message: str) -> Dict[str, Any]:
        """Handle conversation closing"""
        
        # Ask for satisfaction rating
        if session.satisfaction_score is None:
            return {
                "message": "Before you go, how would you rate your experience today? (1-5 stars)",
                "action": "feedback",
                "suggestions": ["⭐⭐⭐⭐⭐ Excellent", "⭐⭐⭐⭐ Good", "⭐⭐⭐ Average"]
            }
        
        # Final closing message
        closing_message = f"""Thank you for chatting with {self.config.get('company_name', 'us')} today! """
        
        if session.lead_quality in [LeadQuality.HOT, LeadQuality.WARM]:
            closing_message += "One of our specialists will follow up with you within 24 hours to discuss next steps."
        
        closing_message += " Have a great day!"
        
        session.state = ConversationState.COMPLETED
        
        return {
            "message": closing_message,
            "action": "complete",
            "next_steps": session.next_steps
        }
    
    async def _assess_lead_quality(self, session: ChatSession) -> LeadQuality:
        """Assess lead quality based on conversation data"""
        
        score = 0
        
        # Company size scoring
        company_size = session.context_data.get('company_size', 0)
        if company_size >= 50:
            score += 3
        elif company_size >= 20:
            score += 2
        elif company_size >= 5:
            score += 1
        
        # Industry scoring (some industries are better fits)
        industry = session.context_data.get('industry', '')
        if industry in ['healthcare', 'finance', 'legal']:
            score += 2
        elif industry in ['manufacturing', 'retail']:
            score += 1
        
        # Pain points scoring
        score += len(session.pain_points)
        
        # Services interest scoring
        score += len(session.qualified_services)
        
        # Budget indication scoring
        if session.budget_range:
            score += 2
        
        # Engagement scoring
        if len(session.conversation_history) >= 6:
            score += 1
        
        # Determine quality
        if score >= 8:
            return LeadQuality.HOT
        elif score >= 5:
            return LeadQuality.WARM
        elif score >= 2:
            return LeadQuality.COLD
        else:
            return LeadQuality.NOT_QUALIFIED
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        metrics = self.session_metrics[session_id]
        
        # Calculate session duration
        if session.state == ConversationState.COMPLETED:
            duration = (session.last_activity - session.start_time).total_seconds()
            metrics.resolution_time = duration
        
        return {
            "session_id": session_id,
            "client_info": {
                "name": session.client_name,
                "email": session.client_email,
                "company": session.client_company,
                "company_size": session.context_data.get('company_size'),
                "industry": session.context_data.get('industry')
            },
            "conversation_summary": {
                "state": session.state.value,
                "lead_quality": session.lead_quality.value,
                "pain_points": session.pain_points,
                "qualified_services": session.qualified_services,
                "budget_range": session.budget_range,
                "escalation_requested": session.escalation_requested,
                "satisfaction_score": session.satisfaction_score
            },
            "metrics": asdict(metrics),
            "next_steps": session.next_steps,
            "conversation_history": session.conversation_history
        }
    
    def get_analytics(self, date_range: int = 7) -> Dict[str, Any]:
        """Get chatbot analytics for specified date range"""
        cutoff_date = datetime.now() - timedelta(days=date_range)
        
        # Filter recent sessions
        recent_sessions = [
            session for session in self.active_sessions.values()
            if session.start_time >= cutoff_date
        ]
        
        if not recent_sessions:
            return {"error": "No sessions found in date range"}
        
        # Calculate metrics
        total_sessions = len(recent_sessions)
        completed_sessions = len([s for s in recent_sessions if s.state == ConversationState.COMPLETED])
        escalated_sessions = len([s for s in recent_sessions if s.escalation_requested])
        
        # Lead quality distribution
        lead_quality_counts = {}
        for quality in LeadQuality:
            lead_quality_counts[quality.value] = len([s for s in recent_sessions if s.lead_quality == quality])
        
        # Service interest analysis
        service_mentions = {}
        for session in recent_sessions:
            for service in session.qualified_services:
                service_mentions[service] = service_mentions.get(service, 0) + 1
        
        # Pain point analysis
        pain_point_counts = {}
        for session in recent_sessions:
            for pain_point in session.pain_points:
                pain_point_counts[pain_point] = pain_point_counts.get(pain_point, 0) + 1
        
        return {
            "date_range": date_range,
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "completion_rate": completed_sessions / total_sessions if total_sessions > 0 else 0,
            "escalation_rate": escalated_sessions / total_sessions if total_sessions > 0 else 0,
            "lead_quality_distribution": lead_quality_counts,
            "top_service_interests": dict(sorted(service_mentions.items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_pain_points": dict(sorted(pain_point_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "average_messages_per_session": np.mean([len(s.conversation_history) for s in recent_sessions]),
            "channels": list(set([s.channel for s in recent_sessions]))
        }

# Example configuration
chatbot_config = {
    "openai_api_key": "your-openai-api-key",
    "company_name": "Your MSP Company",
    "webhook_url": "https://your-domain.com/webhook"
}

async def test_chatbot():
    """Test chatbot functionality"""
    chatbot = MSPIntelligentChatbot(chatbot_config)
    
    # Create session
    session_id = await chatbot.create_session("web", {"name": "John Doe", "email": "john@example.com"})
    
    # Test conversation flow
    messages = [
        "Hello",
        "I have 25 employees and we're in healthcare",
        "We're having security concerns and some downtime issues",
        "Tell me about your managed IT services",
        "What's the pricing like?",
        "Can I speak to someone?"
    ]
    
    for message in messages:
        response = await chatbot.process_message(session_id, message)
        print(f"User: {message}")
        print(f"Bot: {response['message']}\n")
    
    # Get session summary
    summary = await chatbot.get_session_summary(session_id)
    print("Session Summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    asyncio.run(test_chatbot())