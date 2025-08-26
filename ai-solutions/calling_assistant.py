# AI Calling Assistant for MSP Client Outreach and Support
# Automated voice campaigns and intelligent call handling

from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass, asdict
import aiohttp
from twilio.rest import Client as TwilioClient
from twilio.twiml import VoiceResponse
import speech_recognition as sr
import pyttsx3
from openai import OpenAI
import pyaudio
import wave
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class CallCampaign:
    """Structure for call campaign configuration"""
    id: str
    name: str
    script: str
    target_audience: str
    objective: str
    max_calls_per_day: int
    call_window_start: str  # HH:MM format
    call_window_end: str    # HH:MM format
    retry_attempts: int
    callback_url: str
    voice_settings: Dict[str, Any]
    analytics_config: Dict[str, Any]

@dataclass
class CallRecord:
    """Structure for call tracking and analytics"""
    call_id: str
    campaign_id: str
    client_phone: str
    client_name: str
    call_status: str  # scheduled, in_progress, completed, failed, no_answer
    call_duration: int
    call_start: datetime
    call_end: Optional[datetime]
    transcript: str
    sentiment_score: float
    intent_detected: str
    follow_up_required: bool
    notes: str
    callback_requested: bool

class MSPCallingAssistant:
    """Advanced AI-powered calling assistant for MSP operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize services
        self.twilio_client = TwilioClient(
            config.get('twilio_account_sid'),
            config.get('twilio_auth_token')
        )
        self.openai_client = OpenAI(api_key=config.get('openai_api_key'))
        
        # Speech recognition and synthesis
        self.speech_recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self._configure_tts()
        
        # Campaign management
        self.active_campaigns: Dict[str, CallCampaign] = {}
        self.call_records: List[CallRecord] = []
        self.client_database: Dict[str, Dict] = {}
        
        # AI models for conversation intelligence
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.conversation_contexts: Dict[str, Dict] = {}
        
        # MSP-specific intents and responses
        self.msp_intents = {
            "technical_support": {
                "keywords": ["problem", "issue", "not working", "error", "down", "slow"],
                "response_template": "I understand you're experiencing technical difficulties. Let me gather some information to help resolve this quickly."
            },
            "service_inquiry": {
                "keywords": ["service", "pricing", "plan", "package", "offering"],
                "response_template": "I'd be happy to discuss our managed IT services and find the right solution for your business needs."
            },
            "security_concern": {
                "keywords": ["security", "breach", "virus", "malware", "hacked", "suspicious"],
                "response_template": "Security is our top priority. Let me immediately connect you with our cybersecurity team."
            },
            "billing_inquiry": {
                "keywords": ["bill", "invoice", "payment", "charge", "cost"],
                "response_template": "I can help you with billing questions. Let me review your account details."
            },
            "appointment_scheduling": {
                "keywords": ["schedule", "appointment", "meeting", "visit", "time"],
                "response_template": "I'll be happy to schedule a consultation at your convenience."
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _configure_tts(self):
        """Configure text-to-speech engine for natural voice"""
        voices = self.tts_engine.getProperty('voices')
        # Select a professional voice
        for voice in voices:
            if 'english' in voice.name.lower() and 'female' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        self.tts_engine.setProperty('rate', 160)  # Moderate speaking speed
        self.tts_engine.setProperty('volume', 0.8)
    
    async def create_campaign(self, campaign_data: Dict) -> CallCampaign:
        """Create a new calling campaign"""
        try:
            campaign = CallCampaign(**campaign_data)
            self.active_campaigns[campaign.id] = campaign
            
            self.logger.info(f"Campaign created: {campaign.name}")
            return campaign
            
        except Exception as e:
            self.logger.error(f"Failed to create campaign: {str(e)}")
            raise
    
    async def start_campaign(self, campaign_id: str, client_list: List[Dict]) -> Dict[str, Any]:
        """Start an automated calling campaign"""
        if campaign_id not in self.active_campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        campaign = self.active_campaigns[campaign_id]
        results = {
            "campaign_id": campaign_id,
            "total_calls": len(client_list),
            "scheduled_calls": 0,
            "errors": []
        }
        
        try:
            for client in client_list:
                # Schedule call within campaign window
                call_time = self._calculate_call_time(campaign)
                
                call_record = CallRecord(
                    call_id=f"{campaign_id}_{client['phone']}_{int(datetime.now().timestamp())}",
                    campaign_id=campaign_id,
                    client_phone=client['phone'],
                    client_name=client.get('name', 'Unknown'),
                    call_status='scheduled',
                    call_duration=0,
                    call_start=call_time,
                    call_end=None,
                    transcript="",
                    sentiment_score=0.0,
                    intent_detected="",
                    follow_up_required=False,
                    notes="",
                    callback_requested=False
                )
                
                # Schedule the call
                success = await self._schedule_call(call_record, campaign)
                if success:
                    self.call_records.append(call_record)
                    results["scheduled_calls"] += 1
                else:
                    results["errors"].append(f"Failed to schedule call for {client['phone']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Campaign start failed: {str(e)}")
            results["errors"].append(str(e))
            return results
    
    def _calculate_call_time(self, campaign: CallCampaign) -> datetime:
        """Calculate optimal call time within campaign window"""
        now = datetime.now()
        
        # Parse time window
        start_time = datetime.strptime(campaign.call_window_start, "%H:%M").time()
        end_time = datetime.strptime(campaign.call_window_end, "%H:%M").time()
        
        # If current time is within window, schedule for now + 1 minute
        current_time = now.time()
        if start_time <= current_time <= end_time:
            return now + timedelta(minutes=1)
        
        # Otherwise, schedule for next available window
        if current_time < start_time:
            # Schedule for today during window
            call_date = now.date()
        else:
            # Schedule for tomorrow during window
            call_date = (now + timedelta(days=1)).date()
        
        call_datetime = datetime.combine(call_date, start_time)
        return call_datetime + timedelta(minutes=np.random.randint(0, 30))  # Randomize within first 30 min
    
    async def _schedule_call(self, call_record: CallRecord, campaign: CallCampaign) -> bool:
        """Schedule a call using Twilio"""
        try:
            # Create TwiML for the call
            response = VoiceResponse()
            
            # Add initial greeting
            greeting = self._generate_greeting(call_record.client_name, campaign)
            response.say(greeting, voice='Polly.Joanna')
            
            # Add gather for user response
            gather = response.gather(
                input='speech',
                action=f"{campaign.callback_url}/handle_response/{call_record.call_id}",
                speech_timeout=5,
                language='en-US'
            )
            gather.say("Please let me know how I can assist you today.", voice='Polly.Joanna')
            
            # Fallback if no response
            response.say("I didn't hear a response. I'll try calling back later. Have a great day!")
            
            # Create the call
            call = self.twilio_client.calls.create(
                to=call_record.client_phone,
                from_=self.config.get('twilio_phone_number'),
                twiml=str(response),
                status_callback=f"{campaign.callback_url}/call_status/{call_record.call_id}",
                status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
                record=True  # Record the call for analysis
            )
            
            call_record.call_id = call.sid
            self.logger.info(f"Call scheduled: {call.sid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to schedule call: {str(e)}")
            return False
    
    def _generate_greeting(self, client_name: str, campaign: CallCampaign) -> str:
        """Generate personalized greeting for the call"""
        base_greeting = f"Hello {client_name}, this is calling from {self.config.get('company_name', 'your managed IT service provider')}."
        
        # Add campaign-specific message
        if "security" in campaign.objective.lower():
            return f"{base_greeting} I'm calling to discuss important cybersecurity updates for your business."
        elif "service" in campaign.objective.lower():
            return f"{base_greeting} I wanted to reach out about new IT services that could benefit your organization."
        elif "support" in campaign.objective.lower():
            return f"{base_greeting} I'm following up on your recent support request."
        else:
            return f"{base_greeting} I hope you're doing well today."
    
    async def handle_call_response(self, call_id: str, speech_result: str) -> str:
        """Process speech input during call and generate intelligent response"""
        try:
            # Find call record
            call_record = next((call for call in self.call_records if call.call_id == call_id), None)
            if not call_record:
                return "I'm sorry, I couldn't find your call information."
            
            # Update transcript
            call_record.transcript += f"Client: {speech_result}\n"
            
            # Analyze intent and sentiment
            intent = await self._analyze_intent(speech_result)
            sentiment = await self._analyze_sentiment(speech_result)
            
            call_record.intent_detected = intent
            call_record.sentiment_score = sentiment
            
            # Generate intelligent response
            response = await self._generate_response(speech_result, intent, call_record)
            call_record.transcript += f"Assistant: {response}\n"
            
            # Determine if follow-up is needed
            call_record.follow_up_required = self._requires_follow_up(intent, sentiment)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to handle call response: {str(e)}")
            return "I'm sorry, I'm having trouble processing your request. Let me transfer you to a human representative."
    
    async def _analyze_intent(self, speech_text: str) -> str:
        """Analyze user intent from speech"""
        speech_lower = speech_text.lower()
        
        # Check for MSP-specific intents
        for intent, config in self.msp_intents.items():
            if any(keyword in speech_lower for keyword in config["keywords"]):
                return intent
        
        # Use OpenAI for more complex intent analysis
        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing customer intent in IT services calls. Classify the intent from the following categories: technical_support, service_inquiry, security_concern, billing_inquiry, appointment_scheduling, general_question, complaint, compliment, other."
                    },
                    {
                        "role": "user",
                        "content": f"Classify the intent of this customer statement: '{speech_text}'"
                    }
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            return response.choices[0].message.content.strip().lower()
            
        except Exception as e:
            self.logger.error(f"Intent analysis failed: {str(e)}")
            return "other"
    
    async def _analyze_sentiment(self, speech_text: str) -> float:
        """Analyze sentiment of customer speech"""
        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze the sentiment of the customer's statement and return a score between -1 (very negative) and 1 (very positive). Return only the numeric score."
                    },
                    {
                        "role": "user",
                        "content": speech_text
                    }
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            return float(response.choices[0].message.content.strip())
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}")
            return 0.0
    
    async def _generate_response(self, speech_text: str, intent: str, call_record: CallRecord) -> str:
        """Generate intelligent response based on context"""
        
        # Get base response template
        base_response = self.msp_intents.get(intent, {}).get("response_template", "I understand your concern.")
        
        # Use OpenAI to generate contextual response
        try:
            context = f"""
            Client: {call_record.client_name}
            Intent: {intent}
            Previous conversation: {call_record.transcript[-500:]}  # Last 500 chars
            Current statement: {speech_text}
            """
            
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a professional MSP customer service representative. 
                        Generate a helpful, concise response (under 100 words) that addresses the customer's needs.
                        Be professional, empathetic, and solution-focused. Always offer next steps or assistance."""
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            return base_response
    
    def _requires_follow_up(self, intent: str, sentiment: float) -> bool:
        """Determine if call requires follow-up action"""
        # Always follow up on technical issues and complaints
        if intent in ["technical_support", "security_concern"] or sentiment < -0.3:
            return True
        
        # Follow up on service inquiries with neutral/positive sentiment
        if intent == "service_inquiry" and sentiment > -0.1:
            return True
        
        return False
    
    async def handle_inbound_call(self, caller_number: str) -> str:
        """Handle incoming calls with intelligent routing"""
        try:
            # Look up client in database
            client_info = self.client_database.get(caller_number, {})
            client_name = client_info.get('name', 'Valued Client')
            
            # Create call record
            call_record = CallRecord(
                call_id=f"inbound_{caller_number}_{int(datetime.now().timestamp())}",
                campaign_id="inbound",
                client_phone=caller_number,
                client_name=client_name,
                call_status='in_progress',
                call_duration=0,
                call_start=datetime.now(),
                call_end=None,
                transcript="",
                sentiment_score=0.0,
                intent_detected="",
                follow_up_required=False,
                notes="Inbound call",
                callback_requested=False
            )
            
            self.call_records.append(call_record)
            
            # Generate TwiML for intelligent call handling
            response = VoiceResponse()
            
            # Personalized greeting
            greeting = f"Hello {client_name}, thank you for calling {self.config.get('company_name', 'our IT support team')}."
            response.say(greeting, voice='Polly.Joanna')
            
            # Gather customer input
            gather = response.gather(
                input='speech',
                action=f"{self.config.get('callback_url')}/handle_inbound_response/{call_record.call_id}",
                speech_timeout=7,
                language='en-US'
            )
            gather.say("How can I help you today? Please describe your question or concern.", voice='Polly.Joanna')
            
            # Menu fallback
            response.say("If you'd prefer to use our menu, press 1 for technical support, 2 for billing, or 3 for sales.")
            
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Inbound call handling failed: {str(e)}")
            response = VoiceResponse()
            response.say("I apologize, but we're experiencing technical difficulties. Please hold while I transfer you to a representative.")
            response.dial(self.config.get('fallback_number', '+1234567890'))
            return str(response)
    
    async def generate_call_summary(self, call_id: str) -> Dict[str, Any]:
        """Generate comprehensive call summary and insights"""
        call_record = next((call for call in self.call_records if call.call_id == call_id), None)
        if not call_record:
            return {"error": "Call record not found"}
        
        try:
            # Use OpenAI to generate summary
            summary_prompt = f"""
            Generate a comprehensive summary of this customer service call:
            
            Call Details:
            - Client: {call_record.client_name}
            - Duration: {call_record.call_duration} seconds
            - Intent: {call_record.intent_detected}
            - Sentiment: {call_record.sentiment_score}
            
            Full Transcript:
            {call_record.transcript}
            
            Please provide:
            1. Call summary (2-3 sentences)
            2. Key issues discussed
            3. Resolution status
            4. Recommended follow-up actions
            5. Client satisfaction assessment
            """
            
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            ai_summary = response.choices[0].message.content
            
            return {
                "call_id": call_id,
                "client_name": call_record.client_name,
                "call_status": call_record.call_status,
                "duration": call_record.call_duration,
                "intent": call_record.intent_detected,
                "sentiment_score": call_record.sentiment_score,
                "follow_up_required": call_record.follow_up_required,
                "ai_summary": ai_summary,
                "transcript": call_record.transcript,
                "timestamp": call_record.call_start.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate call summary: {str(e)}")
            return {"error": str(e)}
    
    async def get_campaign_analytics(self, campaign_id: str) -> Dict[str, Any]:
        """Get detailed analytics for a campaign"""
        campaign_calls = [call for call in self.call_records if call.campaign_id == campaign_id]
        
        if not campaign_calls:
            return {"error": "No calls found for campaign"}
        
        # Calculate metrics
        total_calls = len(campaign_calls)
        completed_calls = len([call for call in campaign_calls if call.call_status == 'completed'])
        avg_duration = np.mean([call.call_duration for call in campaign_calls if call.call_duration > 0])
        avg_sentiment = np.mean([call.sentiment_score for call in campaign_calls])
        
        # Intent distribution
        intents = [call.intent_detected for call in campaign_calls if call.intent_detected]
        intent_counts = {intent: intents.count(intent) for intent in set(intents)}
        
        # Follow-up requirements
        follow_ups_needed = len([call for call in campaign_calls if call.follow_up_required])
        
        return {
            "campaign_id": campaign_id,
            "total_calls": total_calls,
            "completed_calls": completed_calls,
            "completion_rate": completed_calls / total_calls if total_calls > 0 else 0,
            "average_duration": round(avg_duration, 2) if avg_duration else 0,
            "average_sentiment": round(avg_sentiment, 3),
            "intent_distribution": intent_counts,
            "follow_ups_required": follow_ups_needed,
            "success_metrics": {
                "positive_sentiment_calls": len([call for call in campaign_calls if call.sentiment_score > 0.2]),
                "appointments_scheduled": len([call for call in campaign_calls if "appointment" in call.intent_detected]),
                "technical_issues_resolved": len([call for call in campaign_calls if call.intent_detected == "technical_support" and not call.follow_up_required])
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health metrics"""
        return {
            "active_campaigns": len(self.active_campaigns),
            "total_call_records": len(self.call_records),
            "twilio_status": "connected" if self.twilio_client else "disconnected",
            "openai_status": "connected" if self.openai_client else "disconnected",
            "recent_call_volume": len([call for call in self.call_records if (datetime.now() - call.call_start).days < 1]),
            "system_health": "operational"
        }

# Example configuration
calling_config = {
    "twilio_account_sid": "your-twilio-account-sid",
    "twilio_auth_token": "your-twilio-auth-token",
    "twilio_phone_number": "+1234567890",
    "openai_api_key": "your-openai-api-key",
    "company_name": "Your MSP Company",
    "callback_url": "https://your-domain.com/webhook",
    "fallback_number": "+1234567890"
}

# Example campaign data
sample_campaign = {
    "id": "msp_security_outreach_2024",
    "name": "Cybersecurity Awareness Campaign",
    "script": "Proactive outreach to discuss cybersecurity improvements",
    "target_audience": "Small to medium businesses",
    "objective": "security awareness and service upselling",
    "max_calls_per_day": 50,
    "call_window_start": "09:00",
    "call_window_end": "17:00",
    "retry_attempts": 2,
    "callback_url": "https://your-domain.com/webhook/campaign",
    "voice_settings": {"voice": "Polly.Joanna", "speed": "medium"},
    "analytics_config": {"track_sentiment": True, "record_calls": True}
}

# Example client list
sample_clients = [
    {"name": "John Smith", "phone": "+1234567891", "company": "ABC Corp"},
    {"name": "Jane Doe", "phone": "+1234567892", "company": "XYZ Inc"},
    {"name": "Mike Johnson", "phone": "+1234567893", "company": "Tech Startup LLC"}
]

async def test_calling_assistant():
    """Test the calling assistant functionality"""
    assistant = MSPCallingAssistant(calling_config)
    
    # Create campaign
    campaign = await assistant.create_campaign(sample_campaign)
    print(f"Created campaign: {campaign.name}")
    
    # Start campaign (in production, this would make actual calls)
    # results = await assistant.start_campaign(campaign.id, sample_clients)
    # print(f"Campaign results: {results}")
    
    # Get analytics
    analytics = await assistant.get_campaign_analytics(campaign.id)
    print(f"Campaign analytics: {analytics}")

if __name__ == "__main__":
    asyncio.run(test_calling_assistant())