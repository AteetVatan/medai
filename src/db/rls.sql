-- Row Level Security (RLS) Policies for medAI MVP
-- Ensures data isolation between organizations and proper access control

-- Enable RLS on all tables
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE referrals ENABLE ROW LEVEL SECURITY;
ALTER TABLE encounters ENABLE ROW LEVEL SECURITY;
ALTER TABLE audio_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE medical_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE clinical_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_config ENABLE ROW LEVEL SECURITY;

-- Helper function to get current user's organization
CREATE OR REPLACE FUNCTION get_current_user_org_id()
RETURNS UUID AS $$
BEGIN
    -- In Supabase, this would typically come from JWT claims
    -- For now, we'll use a session variable
    RETURN COALESCE(
        current_setting('app.current_organization_id', true)::UUID,
        NULL
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Helper function to get current user ID
CREATE OR REPLACE FUNCTION get_current_user_id()
RETURNS UUID AS $$
BEGIN
    -- In Supabase, this would come from auth.uid()
    RETURN COALESCE(
        current_setting('app.current_user_id', true)::UUID,
        NULL
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Organizations policies
CREATE POLICY "Users can view their own organization" ON organizations
    FOR SELECT USING (id = get_current_user_org_id());

CREATE POLICY "Super admins can view all organizations" ON organizations
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM users 
            WHERE id = get_current_user_id() 
            AND role = 'super_admin'
        )
    );

-- Users policies
CREATE POLICY "Users can view users in their organization" ON users
    FOR SELECT USING (organization_id = get_current_user_org_id());

CREATE POLICY "Users can update their own profile" ON users
    FOR UPDATE USING (id = get_current_user_id());

CREATE POLICY "Admins can manage users in their organization" ON users
    FOR ALL USING (
        organization_id = get_current_user_org_id() AND
        EXISTS (
            SELECT 1 FROM users 
            WHERE id = get_current_user_id() 
            AND role IN ('admin', 'super_admin')
        )
    );

-- Patients policies
CREATE POLICY "Users can view patients in their organization" ON patients
    FOR SELECT USING (organization_id = get_current_user_org_id());

CREATE POLICY "Therapists can manage patients in their organization" ON patients
    FOR ALL USING (
        organization_id = get_current_user_org_id() AND
        EXISTS (
            SELECT 1 FROM users 
            WHERE id = get_current_user_id() 
            AND role IN ('therapist', 'admin', 'super_admin')
        )
    );

-- Referrals policies
CREATE POLICY "Users can view referrals in their organization" ON referrals
    FOR SELECT USING (organization_id = get_current_user_org_id());

CREATE POLICY "Therapists can manage referrals in their organization" ON referrals
    FOR ALL USING (
        organization_id = get_current_user_org_id() AND
        EXISTS (
            SELECT 1 FROM users 
            WHERE id = get_current_user_id() 
            AND role IN ('therapist', 'admin', 'super_admin')
        )
    );

-- Encounters policies
CREATE POLICY "Users can view encounters in their organization" ON encounters
    FOR SELECT USING (organization_id = get_current_user_org_id());

CREATE POLICY "Therapists can view their own encounters" ON encounters
    FOR SELECT USING (therapist_id = get_current_user_id());

CREATE POLICY "Therapists can manage encounters they're assigned to" ON encounters
    FOR ALL USING (
        organization_id = get_current_user_org_id() AND
        (therapist_id = get_current_user_id() OR
         EXISTS (
             SELECT 1 FROM users 
             WHERE id = get_current_user_id() 
             AND role IN ('admin', 'super_admin')
         ))
    );

-- Audio records policies
CREATE POLICY "Users can view audio records in their organization" ON audio_records
    FOR SELECT USING (organization_id = get_current_user_org_id());

CREATE POLICY "Therapists can view audio records from their encounters" ON audio_records
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM encounters 
            WHERE encounters.id = audio_records.encounter_id 
            AND encounters.therapist_id = get_current_user_id()
        )
    );

CREATE POLICY "Therapists can manage audio records from their encounters" ON audio_records
    FOR ALL USING (
        organization_id = get_current_user_org_id() AND
        EXISTS (
            SELECT 1 FROM encounters 
            WHERE encounters.id = audio_records.encounter_id 
            AND encounters.therapist_id = get_current_user_id()
        )
    );

-- Medical entities policies
CREATE POLICY "Users can view medical entities in their organization" ON medical_entities
    FOR SELECT USING (organization_id = get_current_user_org_id());

CREATE POLICY "Therapists can view medical entities from their encounters" ON medical_entities
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM encounters 
            WHERE encounters.id = medical_entities.encounter_id 
            AND encounters.therapist_id = get_current_user_id()
        )
    );

CREATE POLICY "System can insert medical entities" ON medical_entities
    FOR INSERT WITH CHECK (organization_id = get_current_user_org_id());

-- Clinical notes policies
CREATE POLICY "Users can view clinical notes in their organization" ON clinical_notes
    FOR SELECT USING (organization_id = get_current_user_org_id());

CREATE POLICY "Therapists can view clinical notes from their encounters" ON clinical_notes
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM encounters 
            WHERE encounters.id = clinical_notes.encounter_id 
            AND encounters.therapist_id = get_current_user_id()
        )
    );

CREATE POLICY "Therapists can manage clinical notes from their encounters" ON clinical_notes
    FOR ALL USING (
        organization_id = get_current_user_org_id() AND
        EXISTS (
            SELECT 1 FROM encounters 
            WHERE encounters.id = clinical_notes.encounter_id 
            AND encounters.therapist_id = get_current_user_id()
        )
    );

-- Audit log policies
CREATE POLICY "Users can view audit logs in their organization" ON audit_log
    FOR SELECT USING (organization_id = get_current_user_org_id());

CREATE POLICY "System can insert audit logs" ON audit_log
    FOR INSERT WITH CHECK (organization_id = get_current_user_org_id());

CREATE POLICY "Admins can view all audit logs in their organization" ON audit_log
    FOR SELECT USING (
        organization_id = get_current_user_org_id() AND
        EXISTS (
            SELECT 1 FROM users 
            WHERE id = get_current_user_id() 
            AND role IN ('admin', 'super_admin')
        )
    );

-- System config policies
CREATE POLICY "Admins can view system config" ON system_config
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM users 
            WHERE id = get_current_user_id() 
            AND role IN ('admin', 'super_admin')
        )
    );

CREATE POLICY "Super admins can manage system config" ON system_config
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM users 
            WHERE id = get_current_user_id() 
            AND role = 'super_admin'
        )
    );

-- Functions for setting session variables (to be called from application)
CREATE OR REPLACE FUNCTION set_session_context(user_id UUID, organization_id UUID)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_user_id', user_id::text, true);
    PERFORM set_config('app.current_organization_id', organization_id::text, true);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to clear session context
CREATE OR REPLACE FUNCTION clear_session_context()
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_user_id', NULL, true);
    PERFORM set_config('app.current_organization_id', NULL, true);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Views for common queries with RLS applied
CREATE VIEW user_encounters AS
SELECT 
    e.*,
    p.first_name as patient_first_name,
    p.last_name as patient_last_name,
    p.patient_number,
    u.first_name as therapist_first_name,
    u.last_name as therapist_last_name
FROM encounters e
JOIN patients p ON e.patient_id = p.id
JOIN users u ON e.therapist_id = u.id
WHERE e.organization_id = get_current_user_org_id();

CREATE VIEW encounter_audio_summary AS
SELECT 
    e.id as encounter_id,
    e.patient_id,
    e.therapist_id,
    COUNT(ar.id) as audio_count,
    SUM(ar.duration_seconds) as total_duration,
    MAX(ar.created_at) as last_audio_at
FROM encounters e
LEFT JOIN audio_records ar ON e.id = ar.encounter_id
WHERE e.organization_id = get_current_user_org_id()
GROUP BY e.id, e.patient_id, e.therapist_id;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;
