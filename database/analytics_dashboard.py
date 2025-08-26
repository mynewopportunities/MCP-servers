# MSP Analytics and Business Intelligence Dashboard
# Real-time analytics, KPIs, and price-to-value metrics for MSP operations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import psycopg2
import redis
from datetime import datetime, timedelta, date
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import streamlit.components.v1 as components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration for MSP analytics"""
    host: str = "localhost"
    port: int = 5431  # HAProxy load balancer
    database: str = "msp_enterprise"
    username: str = "msp_admin"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 30

@dataclass
class KPIMetric:
    """Structure for KPI metric representation"""
    name: str
    current_value: float
    target_value: float
    previous_period_value: float
    unit: str
    trend: str  # 'up', 'down', 'stable'
    status: str  # 'green', 'yellow', 'red'
    variance_percentage: float

class MSPAnalyticsDashboard:
    """Comprehensive MSP analytics and business intelligence dashboard"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_engine = None
        self.redis_client = None
        self._initialize_connections()
        
        # Dashboard configuration
        st.set_page_config(
            page_title="MSP Enterprise Analytics",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        self._apply_custom_styles()
        
        # Cache settings
        self.cache_ttl = 300  # 5 minutes
        
    def _initialize_connections(self):
        """Initialize database and cache connections"""
        try:
            # PostgreSQL connection
            connection_string = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            self.db_engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Redis connection for caching
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.redis_client = None
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            st.error("Failed to connect to database. Please check configuration.")
            
    def _apply_custom_styles(self):
        """Apply custom CSS styles for better dashboard appearance"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .kpi-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .kpi-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: white;
            text-align: center;
        }
        
        .kpi-label {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            margin-top: 0.5rem;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        }
        
        .status-green { border-left-color: #28a745 !important; }
        .status-yellow { border-left-color: #ffc107 !important; }
        .status-red { border-left-color: #dc3545 !important; }
        
        .sidebar-content {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get cached data if available and valid"""
        if not self.redis_client:
            return None
            
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None
        
    def _set_cached_data(self, cache_key: str, data: Any, ttl: int = None):
        """Cache data with TTL"""
        if not self.redis_client:
            return
            
        try:
            ttl = ttl or self.cache_ttl
            self.redis_client.setex(cache_key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            
    @st.cache_data(ttl=300)
    def _execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute SQL query with caching"""
        try:
            with self.db_engine.connect() as connection:
                result = pd.read_sql(text(query), connection, params=params or {})
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return pd.DataFrame()
            
    def get_financial_kpis(self, date_range: Tuple[date, date]) -> List[KPIMetric]:
        """Get key financial KPIs"""
        start_date, end_date = date_range
        
        query = """
        WITH current_period AS (
            SELECT 
                SUM(total_amount) as total_revenue,
                COUNT(DISTINCT organization_id) as active_clients,
                AVG(total_amount) as avg_invoice_amount,
                COUNT(*) as invoice_count
            FROM invoices 
            WHERE invoice_date >= :start_date AND invoice_date <= :end_date
            AND status = 'paid'
        ),
        previous_period AS (
            SELECT 
                SUM(total_amount) as prev_revenue,
                COUNT(DISTINCT organization_id) as prev_active_clients
            FROM invoices 
            WHERE invoice_date >= :prev_start AND invoice_date < :start_date
            AND status = 'paid'
        ),
        recurring_revenue AS (
            SELECT SUM(cs.custom_price * cs.quantity) as mrr
            FROM client_services cs
            JOIN services s ON cs.service_id = s.id
            WHERE cs.status = 'active' 
            AND s.service_type = 'recurring'
        )
        SELECT 
            cp.total_revenue,
            cp.active_clients,
            cp.avg_invoice_amount,
            pp.prev_revenue,
            pp.prev_active_clients,
            rr.mrr
        FROM current_period cp, previous_period pp, recurring_revenue rr
        """
        
        # Calculate previous period dates
        period_length = (end_date - start_date).days
        prev_start = start_date - timedelta(days=period_length)
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'prev_start': prev_start
        }
        
        result = self._execute_query(query, params)
        
        if result.empty:
            return []
            
        row = result.iloc[0]
        
        kpis = []
        
        # Total Revenue
        revenue_variance = ((row['total_revenue'] - row['prev_revenue']) / row['prev_revenue'] * 100) if row['prev_revenue'] > 0 else 0
        kpis.append(KPIMetric(
            name="Total Revenue",
            current_value=row['total_revenue'] or 0,
            target_value=100000,  # Example target
            previous_period_value=row['prev_revenue'] or 0,
            unit="USD",
            trend="up" if revenue_variance > 0 else "down" if revenue_variance < 0 else "stable",
            status="green" if revenue_variance >= 5 else "yellow" if revenue_variance >= 0 else "red",
            variance_percentage=revenue_variance
        ))
        
        # Monthly Recurring Revenue
        kpis.append(KPIMetric(
            name="Monthly Recurring Revenue",
            current_value=row['mrr'] or 0,
            target_value=75000,
            previous_period_value=0,  # Would need historical MRR data
            unit="USD",
            trend="stable",
            status="green" if row['mrr'] >= 75000 else "yellow" if row['mrr'] >= 50000 else "red",
            variance_percentage=0
        ))
        
        # Active Clients
        client_variance = ((row['active_clients'] - row['prev_active_clients']) / row['prev_active_clients'] * 100) if row['prev_active_clients'] > 0 else 0
        kpis.append(KPIMetric(
            name="Active Clients",
            current_value=row['active_clients'] or 0,
            target_value=50,
            previous_period_value=row['prev_active_clients'] or 0,
            unit="count",
            trend="up" if client_variance > 0 else "down" if client_variance < 0 else "stable",
            status="green" if client_variance >= 0 else "red",
            variance_percentage=client_variance
        ))
        
        return kpis
        
    def get_operational_kpis(self, date_range: Tuple[date, date]) -> List[KPIMetric]:
        """Get key operational KPIs"""
        start_date, end_date = date_range
        
        query = """
        WITH ticket_metrics AS (
            SELECT 
                COUNT(*) as total_tickets,
                COUNT(*) FILTER (WHERE sla_breach = false) as sla_met_tickets,
                AVG(EXTRACT(EPOCH FROM (first_response_at - created_at))/60)::numeric as avg_response_minutes,
                AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))/3600)::numeric as avg_resolution_hours,
                COUNT(*) FILTER (WHERE priority = 'critical') as critical_tickets
            FROM service_tickets
            WHERE created_at >= :start_date AND created_at <= :end_date
            AND resolved_at IS NOT NULL
        ),
        satisfaction_metrics AS (
            SELECT AVG(overall_satisfaction)::numeric as avg_satisfaction
            FROM client_satisfaction
            WHERE survey_date >= :start_date AND survey_date <= :end_date
        )
        SELECT 
            tm.*,
            sm.avg_satisfaction
        FROM ticket_metrics tm, satisfaction_metrics sm
        """
        
        params = {'start_date': start_date, 'end_date': end_date}
        result = self._execute_query(query, params)
        
        if result.empty:
            return []
            
        row = result.iloc[0]
        kpis = []
        
        # SLA Compliance Rate
        sla_rate = (row['sla_met_tickets'] / row['total_tickets'] * 100) if row['total_tickets'] > 0 else 0
        kpis.append(KPIMetric(
            name="SLA Compliance Rate",
            current_value=sla_rate,
            target_value=95.0,
            previous_period_value=0,
            unit="%",
            trend="stable",
            status="green" if sla_rate >= 95 else "yellow" if sla_rate >= 90 else "red",
            variance_percentage=0
        ))
        
        # Average Response Time
        kpis.append(KPIMetric(
            name="Avg Response Time",
            current_value=row['avg_response_minutes'] or 0,
            target_value=15.0,
            previous_period_value=0,
            unit="minutes",
            trend="stable",
            status="green" if (row['avg_response_minutes'] or 0) <= 15 else "yellow" if (row['avg_response_minutes'] or 0) <= 30 else "red",
            variance_percentage=0
        ))
        
        # Client Satisfaction
        kpis.append(KPIMetric(
            name="Client Satisfaction",
            current_value=row['avg_satisfaction'] or 0,
            target_value=4.5,
            previous_period_value=0,
            unit="/5.0",
            trend="stable",
            status="green" if (row['avg_satisfaction'] or 0) >= 4.5 else "yellow" if (row['avg_satisfaction'] or 0) >= 4.0 else "red",
            variance_percentage=0
        ))
        
        return kpis
        
    def render_kpi_cards(self, kpis: List[KPIMetric]):
        """Render KPI cards with styling"""
        cols = st.columns(len(kpis))
        
        for i, kpi in enumerate(kpis):
            with cols[i]:
                # Determine status class
                status_class = f"status-{kpi.status}"
                
                # Trend indicator
                trend_icon = "📈" if kpi.trend == "up" else "📉" if kpi.trend == "down" else "➡️"
                
                # Format value
                if kpi.unit == "USD":
                    formatted_value = f"${kpi.current_value:,.0f}"
                elif kpi.unit == "%":
                    formatted_value = f"{kpi.current_value:.1f}%"
                elif kpi.unit == "count":
                    formatted_value = f"{kpi.current_value:.0f}"
                else:
                    formatted_value = f"{kpi.current_value:.1f} {kpi.unit}"
                
                st.markdown(f"""
                <div class="metric-card {status_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin: 0; color: #333;">{kpi.name}</h3>
                            <div style="font-size: 1.8rem; font-weight: bold; color: #1f77b4; margin: 0.5rem 0;">
                                {formatted_value}
                            </div>
                            <div style="font-size: 0.9rem; color: #666;">
                                Target: {kpi.target_value:,.0f} {kpi.unit}
                            </div>
                        </div>
                        <div style="font-size: 2rem;">
                            {trend_icon}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    def render_client_profitability_analysis(self, date_range: Tuple[date, date]):
        """Render client profitability analysis charts"""
        start_date, end_date = date_range
        
        query = """
        SELECT 
            o.name as client_name,
            o.industry,
            o.company_size,
            SUM(ii.line_total) as total_revenue,
            COUNT(DISTINCT i.id) as invoice_count,
            AVG(cs.satisfaction_score) as avg_satisfaction,
            COUNT(DISTINCT st.id) as ticket_count,
            (SUM(ii.line_total) / NULLIF(COUNT(DISTINCT st.id), 0)) as revenue_per_ticket
        FROM organizations o
        LEFT JOIN invoices i ON o.id = i.organization_id
        LEFT JOIN invoice_items ii ON i.id = ii.invoice_id
        LEFT JOIN client_services cs ON o.id = cs.organization_id
        LEFT JOIN service_tickets st ON o.id = st.organization_id
        WHERE i.invoice_date >= :start_date AND i.invoice_date <= :end_date
        GROUP BY o.id, o.name, o.industry, o.company_size
        HAVING SUM(ii.line_total) > 0
        ORDER BY total_revenue DESC
        LIMIT 20
        """
        
        params = {'start_date': start_date, 'end_date': end_date}
        df = self._execute_query(query, params)
        
        if df.empty:
            st.warning("No client profitability data available for the selected period.")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by client
            fig_revenue = px.bar(
                df.head(10), 
                x='total_revenue', 
                y='client_name',
                orientation='h',
                title='Top 10 Clients by Revenue',
                labels={'total_revenue': 'Total Revenue ($)', 'client_name': 'Client'},
                color='total_revenue',
                color_continuous_scale='Blues'
            )
            fig_revenue.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_revenue, use_container_width=True)
            
        with col2:
            # Revenue per ticket analysis
            df_filtered = df[df['revenue_per_ticket'].notna()]
            if not df_filtered.empty:
                fig_efficiency = px.scatter(
                    df_filtered,
                    x='ticket_count',
                    y='revenue_per_ticket',
                    size='total_revenue',
                    color='avg_satisfaction',
                    hover_name='client_name',
                    title='Client Efficiency Analysis',
                    labels={
                        'ticket_count': 'Number of Tickets',
                        'revenue_per_ticket': 'Revenue per Ticket ($)',
                        'avg_satisfaction': 'Satisfaction Score'
                    },
                    color_continuous_scale='RdYlGn'
                )
                fig_efficiency.update_layout(height=500)
                st.plotly_chart(fig_efficiency, use_container_width=True)
                
    def render_service_performance_dashboard(self, date_range: Tuple[date, date]):
        """Render service performance analysis"""
        start_date, end_date = date_range
        
        query = """
        SELECT 
            s.name as service_name,
            s.category,
            COUNT(DISTINCT cs.organization_id) as client_count,
            SUM(cs.quantity) as total_units,
            AVG(cs.custom_price) as avg_price,
            SUM(ii.line_total) as total_revenue,
            AVG(cs.satisfaction_score) as avg_satisfaction,
            AVG(cs.utilization_percentage) as avg_utilization
        FROM services s
        LEFT JOIN client_services cs ON s.id = cs.service_id
        LEFT JOIN invoice_items ii ON cs.id = ii.client_service_id
        LEFT JOIN invoices i ON ii.invoice_id = i.id
        WHERE cs.start_date >= :start_date AND cs.start_date <= :end_date
        AND cs.status = 'active'
        GROUP BY s.id, s.name, s.category
        HAVING COUNT(DISTINCT cs.organization_id) > 0
        ORDER BY total_revenue DESC
        """
        
        params = {'start_date': start_date, 'end_date': end_date}
        df = self._execute_query(query, params)
        
        if df.empty:
            st.warning("No service performance data available.")
            return
            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Revenue by service category
            category_revenue = df.groupby('category')['total_revenue'].sum().reset_index()
            fig_category = px.pie(
                category_revenue,
                values='total_revenue',
                names='category',
                title='Revenue Distribution by Service Category'
            )
            st.plotly_chart(fig_category, use_container_width=True)
            
        with col2:
            # Service adoption vs satisfaction
            fig_adoption = px.scatter(
                df,
                x='client_count',
                y='avg_satisfaction',
                size='total_revenue',
                color='category',
                hover_name='service_name',
                title='Service Adoption vs Satisfaction',
                labels={
                    'client_count': 'Number of Clients',
                    'avg_satisfaction': 'Average Satisfaction Score'
                }
            )
            st.plotly_chart(fig_adoption, use_container_width=True)
            
        with col3:
            # Service utilization
            df_util = df[df['avg_utilization'].notna()].head(10)
            fig_util = px.bar(
                df_util,
                x='avg_utilization',
                y='service_name',
                orientation='h',
                title='Service Utilization Rates',
                labels={'avg_utilization': 'Utilization %', 'service_name': 'Service'}
            )
            fig_util.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_util, use_container_width=True)
            
    def render_compliance_status_dashboard(self):
        """Render compliance status overview"""
        query = """
        SELECT 
            o.name as client_name,
            cf.name as framework_name,
            ccs.overall_status,
            ccs.compliance_percentage,
            ccs.open_findings,
            ccs.critical_findings,
            ccs.next_assessment_due - CURRENT_DATE as days_to_assessment
        FROM organizations o
        JOIN client_compliance_status ccs ON o.id = ccs.organization_id
        JOIN compliance_frameworks cf ON ccs.framework_id = cf.id
        WHERE ccs.last_assessment_date >= CURRENT_DATE - INTERVAL '6 months'
        ORDER BY ccs.compliance_percentage ASC
        """
        
        df = self._execute_query(query)
        
        if df.empty:
            st.warning("No compliance data available.")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            # Compliance status distribution
            status_counts = df['overall_status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Overall Compliance Status Distribution',
                color_discrete_map={
                    'compliant': '#28a745',
                    'partial': '#ffc107',
                    'non_compliant': '#dc3545',
                    'not_assessed': '#6c757d'
                }
            )
            st.plotly_chart(fig_status, use_container_width=True)
            
        with col2:
            # Framework compliance rates
            framework_avg = df.groupby('framework_name')['compliance_percentage'].mean().reset_index()
            fig_framework = px.bar(
                framework_avg,
                x='framework_name',
                y='compliance_percentage',
                title='Average Compliance Rate by Framework',
                labels={'compliance_percentage': 'Compliance %', 'framework_name': 'Framework'}
            )
            fig_framework.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_framework, use_container_width=True)
            
        # Detailed compliance table
        st.subheader("Detailed Compliance Status")
        
        # Color code compliance percentages
        def color_compliance(val):
            if val >= 90:
                return 'background-color: #d4edda'
            elif val >= 70:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
                
        styled_df = df.style.applymap(color_compliance, subset=['compliance_percentage'])
        st.dataframe(styled_df, use_container_width=True)
        
    def render_price_to_value_analysis(self, date_range: Tuple[date, date]):
        """Render price-to-value metric analysis"""
        start_date, end_date = date_range
        
        query = """
        WITH client_metrics AS (
            SELECT 
                o.id,
                o.name as client_name,
                o.industry,
                o.company_size,
                SUM(ii.line_total) as total_revenue,
                AVG(cs.satisfaction_score) as avg_satisfaction,
                COUNT(DISTINCT st.id) as ticket_count,
                AVG(CASE WHEN st.resolved_at IS NOT NULL 
                    THEN EXTRACT(EPOCH FROM (st.resolved_at - st.created_at))/3600 
                    END) as avg_resolution_hours,
                COUNT(*) FILTER (WHERE st.sla_breach = false)::float / 
                NULLIF(COUNT(*), 0) * 100 as sla_compliance_rate
            FROM organizations o
            LEFT JOIN invoices i ON o.id = i.organization_id
            LEFT JOIN invoice_items ii ON i.id = ii.invoice_id
            LEFT JOIN client_services cs ON o.id = cs.organization_id
            LEFT JOIN service_tickets st ON o.id = st.organization_id
            WHERE i.invoice_date >= :start_date AND i.invoice_date <= :end_date
            GROUP BY o.id, o.name, o.industry, o.company_size
            HAVING SUM(ii.line_total) > 0
        )
        SELECT 
            *,
            -- Price-to-Value Score Calculation
            (avg_satisfaction * 0.4 + 
             LEAST(sla_compliance_rate, 100) * 0.01 * 0.3 + 
             CASE WHEN avg_resolution_hours <= 4 THEN 5.0 
                  WHEN avg_resolution_hours <= 8 THEN 4.0
                  WHEN avg_resolution_hours <= 24 THEN 3.0
                  ELSE 2.0 END * 0.3) as value_score,
            (total_revenue / 1000) as price_score, -- Normalized price score
            -- Value per dollar calculation
            (avg_satisfaction * 0.4 + 
             LEAST(sla_compliance_rate, 100) * 0.01 * 0.3 + 
             CASE WHEN avg_resolution_hours <= 4 THEN 5.0 
                  WHEN avg_resolution_hours <= 8 THEN 4.0
                  WHEN avg_resolution_hours <= 24 THEN 3.0
                  ELSE 2.0 END * 0.3) / NULLIF((total_revenue / 1000), 0) as value_per_dollar
        FROM client_metrics
        ORDER BY value_per_dollar DESC
        """
        
        params = {'start_date': start_date, 'end_date': end_date}
        df = self._execute_query(query, params)
        
        if df.empty:
            st.warning("No price-to-value data available.")
            return
            
        # Price-to-Value Matrix
        fig_matrix = px.scatter(
            df,
            x='price_score',
            y='value_score',
            size='total_revenue',
            color='value_per_dollar',
            hover_name='client_name',
            hover_data=['avg_satisfaction', 'sla_compliance_rate', 'avg_resolution_hours'],
            title='Client Price-to-Value Analysis Matrix',
            labels={
                'price_score': 'Price Score (Revenue/1000)',
                'value_score': 'Value Score (Composite)',
                'value_per_dollar': 'Value per Dollar'
            },
            color_continuous_scale='RdYlGn'
        )
        
        # Add quadrant lines
        fig_matrix.add_hline(y=df['value_score'].median(), line_dash="dash", line_color="gray")
        fig_matrix.add_vline(x=df['price_score'].median(), line_dash="dash", line_color="gray")
        
        # Add quadrant annotations
        fig_matrix.add_annotation(x=df['price_score'].max()*0.8, y=df['value_score'].max()*0.9,
                                text="High Price<br>High Value", showarrow=False, bgcolor="rgba(0,255,0,0.1)")
        fig_matrix.add_annotation(x=df['price_score'].min()*1.2, y=df['value_score'].max()*0.9,
                                text="Low Price<br>High Value", showarrow=False, bgcolor="rgba(255,255,0,0.1)")
        fig_matrix.add_annotation(x=df['price_score'].max()*0.8, y=df['value_score'].min()*1.2,
                                text="High Price<br>Low Value", showarrow=False, bgcolor="rgba(255,0,0,0.1)")
        fig_matrix.add_annotation(x=df['price_score'].min()*1.2, y=df['value_score'].min()*1.2,
                                text="Low Price<br>Low Value", showarrow=False, bgcolor="rgba(128,128,128,0.1)")
        
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Top value performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Value Performers")
            top_value = df.nlargest(10, 'value_per_dollar')[['client_name', 'value_per_dollar', 'avg_satisfaction', 'total_revenue']]
            st.dataframe(top_value, use_container_width=True)
            
        with col2:
            st.subheader("Value Improvement Opportunities")
            improvement_opps = df.nsmallest(10, 'value_per_dollar')[['client_name', 'value_per_dollar', 'avg_satisfaction', 'sla_compliance_rate']]
            st.dataframe(improvement_opps, use_container_width=True)
            
    def render_predictive_analytics(self, date_range: Tuple[date, date]):
        """Render predictive analytics for business trends"""
        start_date, end_date = date_range
        
        # Revenue trend analysis
        query = """
        SELECT 
            DATE_TRUNC('month', invoice_date) as month,
            SUM(total_amount) as monthly_revenue,
            COUNT(DISTINCT organization_id) as active_clients,
            AVG(total_amount) as avg_invoice_amount
        FROM invoices
        WHERE invoice_date >= :start_date - INTERVAL '12 months'
        AND invoice_date <= :end_date
        AND status = 'paid'
        GROUP BY DATE_TRUNC('month', invoice_date)
        ORDER BY month
        """
        
        params = {'start_date': start_date, 'end_date': end_date}
        df = self._execute_query(query, params)
        
        if df.empty:
            st.warning("No historical data available for predictive analysis.")
            return
            
        # Create trend plots with predictions
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trend', 'Client Growth', 'Average Invoice Value', 'Growth Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Revenue trend
        fig.add_trace(
            go.Scatter(x=df['month'], y=df['monthly_revenue'], 
                      mode='lines+markers', name='Monthly Revenue'),
            row=1, col=1
        )
        
        # Client growth
        fig.add_trace(
            go.Scatter(x=df['month'], y=df['active_clients'], 
                      mode='lines+markers', name='Active Clients'),
            row=1, col=2
        )
        
        # Average invoice value
        fig.add_trace(
            go.Scatter(x=df['month'], y=df['avg_invoice_amount'], 
                      mode='lines+markers', name='Avg Invoice'),
            row=2, col=1
        )
        
        # Growth rate calculation
        df['revenue_growth'] = df['monthly_revenue'].pct_change() * 100
        fig.add_trace(
            go.Scatter(x=df['month'], y=df['revenue_growth'], 
                      mode='lines+markers', name='Revenue Growth %'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Business Trend Analysis & Predictions")
        st.plotly_chart(fig, use_container_width=True)
        
        # Simple forecasting (linear regression)
        if len(df) >= 6:  # Need minimum data points
            from sklearn.linear_model import LinearRegression
            import warnings
            warnings.filterwarnings('ignore')
            
            # Prepare data for forecasting
            df['month_numeric'] = pd.to_datetime(df['month']).astype(np.int64) // 10**9
            X = df['month_numeric'].values.reshape(-1, 1)
            y_revenue = df['monthly_revenue'].values
            y_clients = df['active_clients'].values
            
            # Fit models
            revenue_model = LinearRegression().fit(X, y_revenue)
            clients_model = LinearRegression().fit(X, y_clients)
            
            # Generate predictions for next 6 months
            last_date = pd.to_datetime(df['month'].max())
            future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='M')
            future_numeric = future_months.astype(np.int64) // 10**9
            
            predicted_revenue = revenue_model.predict(future_numeric.values.reshape(-1, 1))
            predicted_clients = clients_model.predict(future_numeric.values.reshape(-1, 1))
            
            # Display predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Revenue Forecast (Next 6 Months)")
                forecast_df = pd.DataFrame({
                    'Month': future_months.strftime('%Y-%m'),
                    'Predicted Revenue': predicted_revenue.astype(int)
                })
                st.dataframe(forecast_df, use_container_width=True)
                
            with col2:
                st.subheader("Client Growth Forecast")
                client_forecast_df = pd.DataFrame({
                    'Month': future_months.strftime('%Y-%m'),
                    'Predicted Clients': predicted_clients.astype(int)
                })
                st.dataframe(client_forecast_df, use_container_width=True)
                
    def render_dashboard(self):
        """Main dashboard rendering function"""
        # Header
        st.markdown('<h1 class="main-header">MSP Enterprise Analytics Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar controls
        with st.sidebar:
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            st.header("Dashboard Controls")
            
            # Date range selector
            date_range = st.date_input(
                "Select Date Range",
                value=(date.today() - timedelta(days=30), date.today()),
                max_value=date.today()
            )
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = date.today() - timedelta(days=30)
                end_date = date.today()
                
            # Dashboard sections
            sections = st.multiselect(
                "Select Dashboard Sections",
                ["KPIs", "Client Analysis", "Service Performance", "Compliance Status", 
                 "Price-to-Value", "Predictive Analytics"],
                default=["KPIs", "Client Analysis", "Service Performance"]
            )
            
            # Refresh button
            if st.button("Refresh Data", type="primary"):
                st.cache_data.clear()
                st.rerun()
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Main dashboard content
        if "KPIs" in sections:
            st.header("📊 Key Performance Indicators")
            
            # Get KPIs
            financial_kpis = self.get_financial_kpis((start_date, end_date))
            operational_kpis = self.get_operational_kpis((start_date, end_date))
            
            if financial_kpis or operational_kpis:
                st.subheader("Financial Metrics")
                self.render_kpi_cards(financial_kpis)
                
                st.subheader("Operational Metrics") 
                self.render_kpi_cards(operational_kpis)
            else:
                st.warning("No KPI data available for the selected period.")
                
        if "Client Analysis" in sections:
            st.header("👥 Client Profitability Analysis")
            self.render_client_profitability_analysis((start_date, end_date))
            
        if "Service Performance" in sections:
            st.header("🔧 Service Performance Dashboard")
            self.render_service_performance_dashboard((start_date, end_date))
            
        if "Compliance Status" in sections:
            st.header("🛡️ Compliance Status Overview")
            self.render_compliance_status_dashboard()
            
        if "Price-to-Value" in sections:
            st.header("💰 Price-to-Value Analysis")
            self.render_price_to_value_analysis((start_date, end_date))
            
        if "Predictive Analytics" in sections:
            st.header("🔮 Predictive Analytics")
            self.render_predictive_analytics((start_date, end_date))

def main():
    """Main application entry point"""
    # Database configuration
    config = DatabaseConfig(
        host=st.secrets.get("DB_HOST", "localhost"),
        port=st.secrets.get("DB_PORT", 5431),
        database=st.secrets.get("DB_NAME", "msp_enterprise"),
        username=st.secrets.get("DB_USER", "msp_admin"),
        password=st.secrets.get("DB_PASSWORD", "")
    )
    
    # Initialize and render dashboard
    dashboard = MSPAnalyticsDashboard(config)
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()