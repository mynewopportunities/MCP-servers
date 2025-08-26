# MSP Enterprise Database Schema
# Comprehensive database design for MSP operations, compliance, and analytics

-- =====================================================
-- MSP ENTERPRISE DATABASE SCHEMA v2.0
-- High Availability, Compliance, and Analytics Platform
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "ltree";

-- =====================================================
-- CORE BUSINESS ENTITIES
-- =====================================================

-- Company/Organization management
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    legal_name VARCHAR(255),
    industry VARCHAR(100),
    company_size VARCHAR(50), -- 'small', 'medium', 'large', 'enterprise'
    tax_id VARCHAR(50),
    website VARCHAR(255),
    phone VARCHAR(50),
    email VARCHAR(255),
    
    -- Address information
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    postal_code VARCHAR(20),
    country VARCHAR(100),
    
    -- Business details
    business_hours JSONB,
    timezone VARCHAR(50),
    fiscal_year_end DATE,
    
    -- Compliance and risk
    compliance_requirements TEXT[],
    risk_level VARCHAR(20) DEFAULT 'medium',
    security_clearance VARCHAR(50),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    is_active BOOLEAN DEFAULT true,
    
    -- Search optimization
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(name, '') || ' ' || coalesce(legal_name, '') || ' ' || coalesce(industry, ''))
    ) STORED
);

-- Client contracts and agreements
CREATE TABLE contracts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    contract_number VARCHAR(100) UNIQUE NOT NULL,
    contract_type VARCHAR(50) NOT NULL, -- 'managed_services', 'project', 'support', 'consulting'
    
    -- Contract terms
    start_date DATE NOT NULL,
    end_date DATE,
    auto_renewal BOOLEAN DEFAULT false,
    renewal_notice_days INTEGER DEFAULT 30,
    
    -- Financial terms
    total_contract_value DECIMAL(12,2),
    billing_frequency VARCHAR(20), -- 'monthly', 'quarterly', 'annually'
    payment_terms VARCHAR(50),
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Service levels
    sla_uptime DECIMAL(5,2), -- 99.9%
    response_time_critical INTEGER, -- minutes
    response_time_high INTEGER,
    response_time_medium INTEGER,
    response_time_low INTEGER,
    
    -- Contract status
    status VARCHAR(20) DEFAULT 'active',
    
    -- Legal and compliance
    governing_law VARCHAR(100),
    dispute_resolution TEXT,
    confidentiality_level VARCHAR(20),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    
    -- Performance tracking
    contract_health_score DECIMAL(3,2), -- 0.00 to 1.00
    last_review_date TIMESTAMP,
    next_review_date TIMESTAMP
);

-- =====================================================
-- SERVICE CATALOG AND PRICING
-- =====================================================

-- Master service catalog
CREATE TABLE services (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100), -- 'managed_it', 'cybersecurity', 'cloud', 'consulting'
    subcategory VARCHAR(100),
    
    -- Service details
    service_type VARCHAR(50), -- 'recurring', 'one_time', 'project_based'
    unit_of_measure VARCHAR(50), -- 'per_user', 'per_device', 'per_server', 'flat_rate'
    
    -- Pricing structure
    base_price DECIMAL(10,2),
    minimum_units INTEGER DEFAULT 1,
    maximum_units INTEGER,
    volume_discounts JSONB, -- Tiered pricing structure
    
    -- Service level specifications
    availability_sla DECIMAL(5,2),
    response_time_sla INTEGER,
    resolution_time_sla INTEGER,
    
    -- Resource requirements
    required_skills TEXT[],
    estimated_hours_per_unit DECIMAL(5,2),
    
    -- Compliance and certification requirements
    compliance_frameworks TEXT[],
    certifications_required TEXT[],
    
    -- Service status
    is_active BOOLEAN DEFAULT true,
    requires_approval BOOLEAN DEFAULT false,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1
);

-- Client-specific service pricing and customizations
CREATE TABLE client_services (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    service_id UUID NOT NULL REFERENCES services(id),
    contract_id UUID REFERENCES contracts(id),
    
    -- Custom pricing
    custom_price DECIMAL(10,2),
    discount_percentage DECIMAL(5,2),
    discount_reason TEXT,
    
    -- Service configuration
    quantity INTEGER NOT NULL DEFAULT 1,
    custom_sla JSONB,
    service_hours JSONB,
    
    -- Billing configuration
    billing_frequency VARCHAR(20),
    proration_rules TEXT,
    
    -- Service dates
    start_date DATE NOT NULL,
    end_date DATE,
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'active',
    provisioning_status VARCHAR(20) DEFAULT 'pending',
    
    -- Performance metrics
    utilization_percentage DECIMAL(5,2),
    satisfaction_score DECIMAL(3,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(organization_id, service_id, start_date)
);

-- =====================================================
-- FINANCIAL MANAGEMENT
-- =====================================================

-- Invoice management
CREATE TABLE invoices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    contract_id UUID REFERENCES contracts(id),
    
    -- Invoice details
    invoice_number VARCHAR(100) UNIQUE NOT NULL,
    invoice_date DATE NOT NULL,
    due_date DATE NOT NULL,
    billing_period_start DATE,
    billing_period_end DATE,
    
    -- Financial amounts
    subtotal DECIMAL(12,2) NOT NULL,
    tax_amount DECIMAL(12,2) DEFAULT 0,
    discount_amount DECIMAL(12,2) DEFAULT 0,
    total_amount DECIMAL(12,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Payment tracking
    status VARCHAR(20) DEFAULT 'draft', -- 'draft', 'sent', 'paid', 'overdue', 'cancelled'
    payment_date TIMESTAMP,
    payment_method VARCHAR(50),
    payment_reference VARCHAR(100),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID
);

-- Invoice line items
CREATE TABLE invoice_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    invoice_id UUID NOT NULL REFERENCES invoices(id),
    client_service_id UUID REFERENCES client_services(id),
    service_id UUID REFERENCES services(id),
    
    -- Line item details
    description TEXT NOT NULL,
    quantity DECIMAL(10,2) NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    line_total DECIMAL(12,2) NOT NULL,
    
    -- Service period
    service_period_start DATE,
    service_period_end DATE,
    
    -- Billing codes
    billing_code VARCHAR(50),
    gl_account VARCHAR(50),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Revenue recognition and analytics
CREATE TABLE revenue_recognition (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    invoice_item_id UUID NOT NULL REFERENCES invoice_items(id),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    
    -- Recognition details
    recognition_date DATE NOT NULL,
    recognized_amount DECIMAL(12,2) NOT NULL,
    recognition_method VARCHAR(50), -- 'straight_line', 'usage_based', 'milestone'
    
    -- Performance obligations
    performance_obligation TEXT,
    completion_percentage DECIMAL(5,2),
    
    -- Accounting period
    accounting_period VARCHAR(10), -- YYYY-MM
    fiscal_quarter INTEGER,
    fiscal_year INTEGER,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- COMPLIANCE AND RISK MANAGEMENT
-- =====================================================

-- Compliance frameworks and requirements
CREATE TABLE compliance_frameworks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20),
    description TEXT,
    
    -- Framework details
    industry_focus VARCHAR(100),
    regulatory_body VARCHAR(100),
    mandatory BOOLEAN DEFAULT false,
    
    -- Implementation details
    controls_count INTEGER,
    implementation_timeline INTEGER, -- days
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual compliance controls
CREATE TABLE compliance_controls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    framework_id UUID NOT NULL REFERENCES compliance_frameworks(id),
    control_id VARCHAR(50) NOT NULL,
    control_name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Control classification
    control_type VARCHAR(50), -- 'preventive', 'detective', 'corrective'
    control_category VARCHAR(100),
    risk_level VARCHAR(20),
    
    -- Implementation requirements
    implementation_guidance TEXT,
    testing_procedures TEXT,
    frequency VARCHAR(50), -- 'continuous', 'monthly', 'quarterly', 'annually'
    
    -- Evidence requirements
    evidence_types TEXT[],
    retention_period INTEGER, -- months
    
    -- Status
    is_mandatory BOOLEAN DEFAULT true,
    is_automated BOOLEAN DEFAULT false,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(framework_id, control_id)
);

-- Client compliance status tracking
CREATE TABLE client_compliance_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    framework_id UUID NOT NULL REFERENCES compliance_frameworks(id),
    
    -- Compliance status
    overall_status VARCHAR(20) DEFAULT 'not_assessed', -- 'compliant', 'non_compliant', 'partial', 'not_assessed'
    compliance_percentage DECIMAL(5,2),
    
    -- Assessment details
    last_assessment_date TIMESTAMP,
    next_assessment_due TIMESTAMP,
    assessment_method VARCHAR(50),
    
    -- Remediation tracking
    open_findings INTEGER DEFAULT 0,
    critical_findings INTEGER DEFAULT 0,
    high_findings INTEGER DEFAULT 0,
    
    -- Certification status
    certification_status VARCHAR(20),
    certification_date TIMESTAMP,
    certification_expiry TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(organization_id, framework_id)
);

-- Individual control assessments
CREATE TABLE control_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    control_id UUID NOT NULL REFERENCES compliance_controls(id),
    
    -- Assessment results
    assessment_date TIMESTAMP NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'pass', 'fail', 'partial', 'not_tested'
    effectiveness_rating VARCHAR(20), -- 'effective', 'needs_improvement', 'ineffective'
    
    -- Testing details
    testing_method VARCHAR(50),
    tester_name VARCHAR(100),
    sample_size INTEGER,
    
    -- Findings and recommendations
    findings TEXT,
    recommendations TEXT,
    management_response TEXT,
    
    -- Remediation tracking
    remediation_plan TEXT,
    remediation_due_date DATE,
    remediation_status VARCHAR(20),
    
    -- Evidence
    evidence_references TEXT[],
    evidence_location TEXT,
    
    -- Risk assessment
    inherent_risk VARCHAR(20),
    residual_risk VARCHAR(20),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID
);

-- =====================================================
-- PERFORMANCE METRICS AND ANALYTICS
-- =====================================================

-- Key Performance Indicators (KPIs)
CREATE TABLE kpi_definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(50), -- 'financial', 'operational', 'client_satisfaction', 'compliance'
    
    -- Calculation details
    calculation_method TEXT,
    data_sources TEXT[],
    calculation_frequency VARCHAR(20), -- 'real_time', 'daily', 'weekly', 'monthly'
    
    -- Targets and benchmarks
    target_value DECIMAL(12,4),
    benchmark_value DECIMAL(12,4),
    unit_of_measure VARCHAR(50),
    
    -- Thresholds
    red_threshold DECIMAL(12,4),
    yellow_threshold DECIMAL(12,4),
    green_threshold DECIMAL(12,4),
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Historical KPI values
CREATE TABLE kpi_values (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kpi_id UUID NOT NULL REFERENCES kpi_definitions(id),
    organization_id UUID REFERENCES organizations(id), -- NULL for company-wide KPIs
    
    -- Value details
    measurement_date TIMESTAMP NOT NULL,
    period_type VARCHAR(20), -- 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    period_start TIMESTAMP,
    period_end TIMESTAMP,
    
    -- Values
    actual_value DECIMAL(12,4),
    target_value DECIMAL(12,4),
    variance DECIMAL(12,4),
    variance_percentage DECIMAL(5,2),
    
    -- Status indicators
    status VARCHAR(20), -- 'green', 'yellow', 'red'
    trend VARCHAR(20), -- 'improving', 'stable', 'declining'
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(kpi_id, organization_id, measurement_date, period_type)
);

-- Client satisfaction tracking
CREATE TABLE client_satisfaction (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    
    -- Survey details
    survey_date TIMESTAMP NOT NULL,
    survey_type VARCHAR(50), -- 'nps', 'csat', 'quarterly_review', 'project_completion'
    response_method VARCHAR(50),
    
    -- Satisfaction scores
    overall_satisfaction DECIMAL(3,2), -- 1.00 to 5.00
    nps_score INTEGER, -- -100 to 100
    likelihood_to_renew DECIMAL(3,2),
    likelihood_to_recommend DECIMAL(3,2),
    
    -- Detailed ratings
    service_quality_rating DECIMAL(3,2),
    response_time_rating DECIMAL(3,2),
    technical_expertise_rating DECIMAL(3,2),
    communication_rating DECIMAL(3,2),
    value_for_money_rating DECIMAL(3,2),
    
    -- Qualitative feedback
    feedback_text TEXT,
    improvement_suggestions TEXT,
    positive_comments TEXT,
    
    -- Follow-up actions
    follow_up_required BOOLEAN DEFAULT false,
    follow_up_actions TEXT,
    follow_up_due_date DATE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    survey_sent_by UUID,
    
    -- Response tracking
    response_time_hours INTEGER,
    survey_completion_rate DECIMAL(5,2)
);

-- =====================================================
-- INCIDENT AND SERVICE MANAGEMENT
-- =====================================================

-- Service incidents and requests
CREATE TABLE service_tickets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticket_number VARCHAR(100) UNIQUE NOT NULL,
    organization_id UUID NOT NULL REFERENCES organizations(id),
    
    -- Ticket classification
    ticket_type VARCHAR(50) NOT NULL, -- 'incident', 'service_request', 'change_request', 'problem'
    category VARCHAR(100),
    subcategory VARCHAR(100),
    
    -- Content
    title VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Priority and impact
    priority VARCHAR(20) NOT NULL, -- 'critical', 'high', 'medium', 'low'
    impact VARCHAR(20) NOT NULL, -- 'critical', 'high', 'medium', 'low'
    urgency VARCHAR(20) NOT NULL,
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'open',
    resolution_status VARCHAR(50),
    
    -- Assignment
    assigned_to UUID,
    assigned_team VARCHAR(100),
    
    -- Time tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    first_response_at TIMESTAMP,
    resolved_at TIMESTAMP,
    closed_at TIMESTAMP,
    
    -- SLA tracking
    response_sla_minutes INTEGER,
    resolution_sla_minutes INTEGER,
    sla_breach BOOLEAN DEFAULT false,
    
    -- Resolution details
    resolution_summary TEXT,
    root_cause TEXT,
    preventive_measures TEXT,
    
    -- Customer communication
    client_notification_sent BOOLEAN DEFAULT false,
    client_satisfaction_rating DECIMAL(3,2),
    
    -- Metadata
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID
);

-- =====================================================
-- ASSET AND INVENTORY MANAGEMENT
-- =====================================================

-- IT Assets tracking
CREATE TABLE assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    
    -- Asset identification
    asset_tag VARCHAR(100) UNIQUE,
    serial_number VARCHAR(100),
    model_number VARCHAR(100),
    
    -- Asset details
    asset_type VARCHAR(50) NOT NULL, -- 'server', 'workstation', 'network_device', 'software'
    category VARCHAR(100),
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    
    -- Location and assignment
    location VARCHAR(255),
    assigned_to_user VARCHAR(100),
    assigned_to_department VARCHAR(100),
    
    -- Financial information
    purchase_date DATE,
    purchase_price DECIMAL(10,2),
    depreciation_method VARCHAR(50),
    current_value DECIMAL(10,2),
    
    -- Lifecycle management
    warranty_start DATE,
    warranty_end DATE,
    maintenance_schedule VARCHAR(100),
    eol_date DATE,
    
    -- Status
    status VARCHAR(50) DEFAULT 'active', -- 'active', 'inactive', 'retired', 'disposed'
    condition VARCHAR(50), -- 'excellent', 'good', 'fair', 'poor'
    
    -- Compliance and security
    compliance_status VARCHAR(50),
    security_classification VARCHAR(50),
    encryption_status BOOLEAN,
    
    -- Configuration
    specifications JSONB,
    installed_software JSONB,
    network_configuration JSONB,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_audit_date TIMESTAMP
);

-- =====================================================
-- MONITORING AND ALERTING
-- =====================================================

-- System monitoring metrics
CREATE TABLE monitoring_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    asset_id UUID REFERENCES assets(id),
    
    -- Metric identification
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50), -- 'performance', 'availability', 'security', 'capacity'
    data_source VARCHAR(100),
    
    -- Timestamp and value
    timestamp TIMESTAMP NOT NULL,
    value DECIMAL(15,6),
    unit VARCHAR(20),
    
    -- Contextual information
    tags JSONB,
    metadata JSONB,
    
    -- Aggregation period
    aggregation_period VARCHAR(20), -- 'raw', '1min', '5min', '1hour', '1day'
    
    -- Status
    status VARCHAR(20), -- 'normal', 'warning', 'critical'
    
    -- Indexing for time-series queries
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create hypertable for time-series data (if using TimescaleDB)
-- SELECT create_hypertable('monitoring_metrics', 'timestamp');

-- Alert definitions and rules
CREATE TABLE alert_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id),
    
    -- Rule identification
    rule_name VARCHAR(100) NOT NULL,
    description TEXT,
    severity VARCHAR(20), -- 'info', 'warning', 'critical'
    
    -- Condition definition
    metric_name VARCHAR(100),
    condition_operator VARCHAR(10), -- '>', '<', '>=', '<=', '==', '!='
    threshold_value DECIMAL(15,6),
    
    -- Time-based conditions
    evaluation_window INTEGER, -- minutes
    trigger_duration INTEGER, -- minutes
    
    -- Actions
    notification_channels TEXT[], -- 'email', 'sms', 'webhook', 'slack'
    escalation_rules JSONB,
    auto_remediation_script TEXT,
    
    -- Status
    is_enabled BOOLEAN DEFAULT true,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID
);

-- Alert instances
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_id UUID NOT NULL REFERENCES alert_rules(id),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    
    -- Alert details
    alert_time TIMESTAMP NOT NULL,
    severity VARCHAR(20),
    title VARCHAR(255),
    message TEXT,
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'acknowledged', 'resolved', 'closed'
    acknowledged_at TIMESTAMP,
    acknowledged_by UUID,
    resolved_at TIMESTAMP,
    resolved_by UUID,
    
    -- Correlation
    correlation_id UUID, -- For grouping related alerts
    parent_alert_id UUID REFERENCES alerts(id),
    
    -- Impact assessment
    affected_services TEXT[],
    business_impact VARCHAR(20),
    
    -- Resolution tracking
    resolution_notes TEXT,
    time_to_acknowledge INTEGER, -- seconds
    time_to_resolve INTEGER, -- seconds
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- REPORTING AND ANALYTICS VIEWS
-- =====================================================

-- Client profitability analysis
CREATE VIEW client_profitability AS
SELECT 
    o.id as organization_id,
    o.name as client_name,
    o.industry,
    o.company_size,
    
    -- Revenue metrics
    SUM(ii.line_total) as total_revenue,
    AVG(ii.unit_price) as average_unit_price,
    COUNT(DISTINCT i.id) as invoice_count,
    
    -- Service metrics
    COUNT(DISTINCT cs.service_id) as services_count,
    AVG(cs.satisfaction_score) as avg_satisfaction,
    
    -- Support metrics
    COUNT(DISTINCT st.id) as ticket_count,
    AVG(EXTRACT(EPOCH FROM (st.resolved_at - st.created_at))/3600) as avg_resolution_hours,
    
    -- Profitability calculations
    (SUM(ii.line_total) / COUNT(DISTINCT st.id)) as revenue_per_ticket,
    
    -- Contract health
    AVG(c.contract_health_score) as avg_contract_health,
    
    -- Time period
    DATE_TRUNC('month', i.invoice_date) as month
    
FROM organizations o
LEFT JOIN invoices i ON o.id = i.organization_id
LEFT JOIN invoice_items ii ON i.id = ii.invoice_id  
LEFT JOIN client_services cs ON o.id = cs.organization_id
LEFT JOIN service_tickets st ON o.id = st.organization_id
LEFT JOIN contracts c ON o.id = c.organization_id

WHERE i.invoice_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY o.id, o.name, o.industry, o.company_size, DATE_TRUNC('month', i.invoice_date);

-- Service performance dashboard
CREATE VIEW service_performance_dashboard AS
SELECT 
    s.id as service_id,
    s.name as service_name,
    s.category,
    
    -- Adoption metrics
    COUNT(DISTINCT cs.organization_id) as client_count,
    SUM(cs.quantity) as total_units,
    AVG(cs.custom_price) as avg_price,
    
    -- Revenue metrics
    SUM(ii.line_total) as total_revenue,
    AVG(ii.line_total / cs.quantity) as revenue_per_unit,
    
    -- Satisfaction metrics
    AVG(cs.satisfaction_score) as avg_satisfaction,
    COUNT(CASE WHEN cs.satisfaction_score >= 4.0 THEN 1 END) as highly_satisfied_clients,
    
    -- Utilization metrics
    AVG(cs.utilization_percentage) as avg_utilization,
    
    -- Support metrics
    COUNT(st.id) as related_tickets,
    AVG(CASE WHEN st.resolved_at IS NOT NULL 
        THEN EXTRACT(EPOCH FROM (st.resolved_at - st.created_at))/3600 
        END) as avg_resolution_hours,
    
    -- Time period
    DATE_TRUNC('quarter', cs.start_date) as quarter

FROM services s
LEFT JOIN client_services cs ON s.id = cs.service_id
LEFT JOIN invoice_items ii ON cs.id = ii.client_service_id
LEFT JOIN service_tickets st ON cs.organization_id = st.organization_id

WHERE cs.start_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY s.id, s.name, s.category, DATE_TRUNC('quarter', cs.start_date);

-- Compliance status summary
CREATE VIEW compliance_summary AS
SELECT 
    o.id as organization_id,
    o.name as client_name,
    cf.name as framework_name,
    ccs.overall_status,
    ccs.compliance_percentage,
    ccs.open_findings,
    ccs.critical_findings,
    
    -- Risk indicators
    CASE 
        WHEN ccs.critical_findings > 0 THEN 'high'
        WHEN ccs.open_findings > 5 THEN 'medium'
        ELSE 'low'
    END as risk_level,
    
    -- Time to next assessment
    ccs.next_assessment_due - CURRENT_DATE as days_to_assessment,
    
    -- Recent assessment activity
    COUNT(ca.id) FILTER (WHERE ca.assessment_date >= CURRENT_DATE - INTERVAL '30 days') as recent_assessments,
    
    ccs.last_assessment_date,
    ccs.certification_expiry

FROM organizations o
JOIN client_compliance_status ccs ON o.id = ccs.organization_id
JOIN compliance_frameworks cf ON ccs.framework_id = cf.id
LEFT JOIN control_assessments ca ON o.id = ca.organization_id

GROUP BY o.id, o.name, cf.name, ccs.overall_status, ccs.compliance_percentage, 
         ccs.open_findings, ccs.critical_findings, ccs.next_assessment_due, 
         ccs.last_assessment_date, ccs.certification_expiry;

-- =====================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =====================================================

-- Organizations
CREATE INDEX idx_organizations_search ON organizations USING GIN(search_vector);
CREATE INDEX idx_organizations_industry ON organizations(industry);
CREATE INDEX idx_organizations_size ON organizations(company_size);
CREATE INDEX idx_organizations_active ON organizations(is_active);

-- Contracts
CREATE INDEX idx_contracts_org_status ON contracts(organization_id, status);
CREATE INDEX idx_contracts_dates ON contracts(start_date, end_date);
CREATE INDEX idx_contracts_renewal ON contracts(next_review_date) WHERE auto_renewal = true;

-- Services and pricing
CREATE INDEX idx_client_services_org_date ON client_services(organization_id, start_date);
CREATE INDEX idx_client_services_active ON client_services(status) WHERE status = 'active';

-- Financial
CREATE INDEX idx_invoices_org_date ON invoices(organization_id, invoice_date);
CREATE INDEX idx_invoices_status ON invoices(status);
CREATE INDEX idx_invoices_due ON invoices(due_date) WHERE status IN ('sent', 'overdue');

-- Compliance
CREATE INDEX idx_compliance_status_org ON client_compliance_status(organization_id);
CREATE INDEX idx_control_assessments_org_date ON control_assessments(organization_id, assessment_date);
CREATE INDEX idx_control_assessments_status ON control_assessments(status);

-- Performance metrics
CREATE INDEX idx_kpi_values_kpi_date ON kpi_values(kpi_id, measurement_date);
CREATE INDEX idx_kpi_values_org_date ON kpi_values(organization_id, measurement_date) WHERE organization_id IS NOT NULL;

-- Monitoring (time-series optimization)
CREATE INDEX idx_monitoring_metrics_time ON monitoring_metrics(timestamp DESC);
CREATE INDEX idx_monitoring_metrics_org_time ON monitoring_metrics(organization_id, timestamp DESC);
CREATE INDEX idx_monitoring_metrics_asset_time ON monitoring_metrics(asset_id, timestamp DESC) WHERE asset_id IS NOT NULL;

-- Alerts
CREATE INDEX idx_alerts_org_status ON alerts(organization_id, status);
CREATE INDEX idx_alerts_time_severity ON alerts(alert_time, severity);

-- Service tickets
CREATE INDEX idx_tickets_org_status ON service_tickets(organization_id, status);
CREATE INDEX idx_tickets_priority_created ON service_tickets(priority, created_at);
CREATE INDEX idx_tickets_sla_breach ON service_tickets(sla_breach) WHERE sla_breach = true;

-- =====================================================
-- TRIGGERS FOR AUTOMATION
-- =====================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers to relevant tables
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_contracts_updated_at BEFORE UPDATE ON contracts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_services_updated_at BEFORE UPDATE ON services FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_client_services_updated_at BEFORE UPDATE ON client_services FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_invoices_updated_at BEFORE UPDATE ON invoices FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Contract health score calculation trigger
CREATE OR REPLACE FUNCTION calculate_contract_health_score()
RETURNS TRIGGER AS $$
DECLARE
    satisfaction_score DECIMAL(3,2);
    sla_performance DECIMAL(3,2);
    ticket_volume INTEGER;
    health_score DECIMAL(3,2);
BEGIN
    -- Get average satisfaction for the organization
    SELECT AVG(overall_satisfaction) INTO satisfaction_score
    FROM client_satisfaction 
    WHERE organization_id = NEW.organization_id 
    AND survey_date >= CURRENT_DATE - INTERVAL '6 months';
    
    -- Calculate SLA performance (simplified)
    SELECT (COUNT(*) FILTER (WHERE sla_breach = false)::DECIMAL / COUNT(*))
    INTO sla_performance
    FROM service_tickets 
    WHERE organization_id = NEW.organization_id
    AND created_at >= CURRENT_DATE - INTERVAL '3 months';
    
    -- Get ticket volume trend
    SELECT COUNT(*) INTO ticket_volume
    FROM service_tickets
    WHERE organization_id = NEW.organization_id
    AND created_at >= CURRENT_DATE - INTERVAL '1 month';
    
    -- Calculate composite health score
    health_score := COALESCE(
        (COALESCE(satisfaction_score, 3.0) / 5.0 * 0.4) +
        (COALESCE(sla_performance, 0.9) * 0.4) +
        (CASE WHEN ticket_volume < 10 THEN 0.2 ELSE 0.1 END),
        0.5
    );
    
    NEW.contract_health_score := LEAST(1.0, GREATEST(0.0, health_score));
    NEW.last_review_date := CURRENT_TIMESTAMP;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_contract_health BEFORE UPDATE ON contracts FOR EACH ROW EXECUTE FUNCTION calculate_contract_health_score();

-- =====================================================
-- DATA RETENTION POLICIES
-- =====================================================

-- Monitoring data retention (keep 13 months, aggregate older data)
CREATE OR REPLACE FUNCTION cleanup_monitoring_data()
RETURNS void AS $$
BEGIN
    -- Delete raw monitoring data older than 13 months
    DELETE FROM monitoring_metrics 
    WHERE timestamp < CURRENT_DATE - INTERVAL '13 months'
    AND aggregation_period = 'raw';
    
    -- Delete aggregated data older than 3 years
    DELETE FROM monitoring_metrics 
    WHERE timestamp < CURRENT_DATE - INTERVAL '3 years'
    AND aggregation_period IN ('1hour', '1day');
    
    -- Archive completed service tickets older than 2 years
    -- (In production, move to archive table instead of delete)
    DELETE FROM service_tickets
    WHERE status = 'closed' 
    AND closed_at < CURRENT_DATE - INTERVAL '2 years';
    
END;
$$ LANGUAGE plpgsql;

-- Schedule cleanup job (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-monitoring-data', '0 2 1 * *', 'SELECT cleanup_monitoring_data();');

-- =====================================================
-- INITIAL DATA AND CONFIGURATION
-- =====================================================

-- Insert default compliance frameworks
INSERT INTO compliance_frameworks (name, version, description, industry_focus, regulatory_body, mandatory, controls_count) VALUES
('SOC 2 Type II', '2017', 'Service Organization Control 2 Trust Services Criteria', 'Technology Services', 'AICPA', true, 64),
('ISO 27001', '2013', 'Information Security Management System', 'All Industries', 'ISO', false, 114),
('NIST Cybersecurity Framework', '1.1', 'Framework for Improving Critical Infrastructure Cybersecurity', 'Critical Infrastructure', 'NIST', false, 108),
('HIPAA', '2013', 'Health Insurance Portability and Accountability Act', 'Healthcare', 'HHS', true, 42),
('PCI DSS', '4.0', 'Payment Card Industry Data Security Standard', 'Payment Processing', 'PCI SSC', true, 12),
('GDPR', '2018', 'General Data Protection Regulation', 'Data Processing', 'EU', true, 99);

-- Insert default service categories
INSERT INTO services (service_code, name, description, category, service_type, unit_of_measure, base_price) VALUES
('MGIT-001', 'Managed IT Services - Essential', '24/7 monitoring and basic support', 'managed_it', 'recurring', 'per_user', 150.00),
('MGIT-002', 'Managed IT Services - Professional', 'Comprehensive IT management with proactive support', 'managed_it', 'recurring', 'per_user', 250.00),
('MGIT-003', 'Managed IT Services - Enterprise', 'Full-service IT management with dedicated resources', 'managed_it', 'recurring', 'per_user', 400.00),
('CYBR-001', 'Cybersecurity Monitoring', '24/7 security monitoring and threat detection', 'cybersecurity', 'recurring', 'per_user', 100.00),
('CYBR-002', 'Vulnerability Assessment', 'Monthly vulnerability scans and reporting', 'cybersecurity', 'recurring', 'per_organization', 500.00),
('CYBR-003', 'Security Awareness Training', 'Monthly security training for employees', 'cybersecurity', 'recurring', 'per_user', 25.00),
('CLOUD-001', 'Cloud Migration Services', 'Complete cloud migration and setup', 'cloud_services', 'project_based', 'flat_rate', 15000.00),
('CLOUD-002', 'Cloud Management', 'Ongoing cloud infrastructure management', 'cloud_services', 'recurring', 'per_server', 200.00),
('BACKUP-001', 'Data Backup and Recovery', 'Automated backup with disaster recovery', 'backup_recovery', 'recurring', 'per_gb', 2.00),
('SUPPORT-001', '24/7 Help Desk', 'Round-the-clock technical support', 'support', 'recurring', 'per_user', 50.00);

-- Insert default KPI definitions
INSERT INTO kpi_definitions (name, description, category, calculation_method, target_value, unit_of_measure, red_threshold, yellow_threshold, green_threshold) VALUES
('Client Satisfaction Score', 'Average client satisfaction rating', 'client_satisfaction', 'Average of satisfaction ratings', 4.5, 'rating', 3.0, 4.0, 4.5),
('Monthly Recurring Revenue', 'Total monthly recurring revenue', 'financial', 'Sum of active monthly services', 100000.00, 'USD', 80000.00, 90000.00, 100000.00),
('Average Response Time', 'Average time to first response on tickets', 'operational', 'Average response time in minutes', 15.0, 'minutes', 30.0, 20.0, 15.0),
('SLA Compliance Rate', 'Percentage of SLAs met', 'operational', 'Percentage of tickets meeting SLA', 95.0, 'percentage', 85.0, 90.0, 95.0),
('Client Retention Rate', 'Percentage of clients retained annually', 'financial', 'Retention rate calculation', 95.0, 'percentage', 85.0, 90.0, 95.0),
('Security Incident Rate', 'Number of security incidents per month', 'compliance', 'Count of security incidents', 0.0, 'count', 3.0, 1.0, 0.0);

-- Create database roles and permissions
CREATE ROLE msp_admin;
CREATE ROLE msp_analyst; 
CREATE ROLE msp_technician;
CREATE ROLE msp_client_readonly;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO msp_admin;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO msp_analyst;
GRANT SELECT, INSERT, UPDATE ON service_tickets, assets, monitoring_metrics, alerts TO msp_technician;
GRANT SELECT ON organizations, client_services, invoices, client_satisfaction TO msp_client_readonly;

-- =====================================================
-- PERFORMANCE OPTIMIZATION SETTINGS
-- =====================================================

-- Optimize for analytics workloads
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET track_io_timing = on;
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log slow queries

-- Memory settings for better performance
ALTER SYSTEM SET effective_cache_size = '4GB';
ALTER SYSTEM SET shared_buffers = '1GB';  
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';

-- Connection and worker settings
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET max_worker_processes = 16;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Autovacuum settings for better maintenance
ALTER SYSTEM SET autovacuum_naptime = '30s';
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.1;
ALTER SYSTEM SET autovacuum_analyze_scale_factor = 0.05;

-- Note: These settings require a PostgreSQL restart to take effect

-- =====================================================
-- ANALYTICS AND REPORTING FUNCTIONS
-- =====================================================

-- Function to calculate client lifetime value
CREATE OR REPLACE FUNCTION calculate_client_ltv(client_id UUID)
RETURNS DECIMAL(12,2) AS $$
DECLARE
    avg_monthly_revenue DECIMAL(12,2);
    retention_rate DECIMAL(5,4);
    ltv DECIMAL(12,2);
BEGIN
    -- Calculate average monthly revenue
    SELECT AVG(monthly_revenue) INTO avg_monthly_revenue
    FROM (
        SELECT 
            DATE_TRUNC('month', i.invoice_date) as month,
            SUM(i.total_amount) as monthly_revenue
        FROM invoices i
        WHERE i.organization_id = client_id
        AND i.invoice_date >= CURRENT_DATE - INTERVAL '12 months'
        GROUP BY DATE_TRUNC('month', i.invoice_date)
    ) monthly_rev;
    
    -- Estimate retention rate (simplified)
    SELECT 0.95 INTO retention_rate; -- Default 95% monthly retention
    
    -- Calculate LTV using formula: ARPU / Churn Rate
    ltv := COALESCE(avg_monthly_revenue, 0) / (1 - retention_rate);
    
    RETURN ltv;
END;
$$ LANGUAGE plpgsql;

-- Function to generate compliance report
CREATE OR REPLACE FUNCTION generate_compliance_report(client_id UUID, framework_name TEXT)
RETURNS TABLE (
    control_id VARCHAR(50),
    control_name VARCHAR(255),
    status VARCHAR(20),
    last_assessment TIMESTAMP,
    findings_count INTEGER,
    risk_level VARCHAR(20)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cc.control_id,
        cc.control_name,
        COALESCE(ca.status, 'not_assessed'::VARCHAR(20)) as status,
        ca.assessment_date as last_assessment,
        COUNT(ca.id)::INTEGER as findings_count,
        cc.risk_level
    FROM compliance_controls cc
    JOIN compliance_frameworks cf ON cc.framework_id = cf.id
    LEFT JOIN control_assessments ca ON cc.id = ca.control_id AND ca.organization_id = client_id
    WHERE cf.name = framework_name
    GROUP BY cc.control_id, cc.control_name, ca.status, ca.assessment_date, cc.risk_level
    ORDER BY cc.control_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON DATABASE TEMPLATE IS 'MSP Enterprise Database - Comprehensive platform for managed service provider operations, compliance management, and business analytics';