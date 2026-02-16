-- Migration 007: Authentication - Users and Sessions
-- Creates tables for user authentication and authorization

-- ==================== User Status Enum ====================

DO $$ BEGIN
    CREATE TYPE user_status_enum AS ENUM (
        'active',
        'inactive',
        'suspended',
        'pending_verification'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;


-- ==================== Users Table ====================

CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    roles TEXT[] NOT NULL DEFAULT ARRAY['user']::TEXT[],
    status user_status_enum NOT NULL DEFAULT 'active',
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for users table
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);


-- ==================== User Sessions Table (Optional - for token revocation) ====================

CREATE TABLE IF NOT EXISTS user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    refresh_token_hash VARCHAR(255) NOT NULL,
    device_info JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Indexes for user_sessions table
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_refresh_token_hash ON user_sessions(refresh_token_hash);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_sessions_is_active ON user_sessions(is_active);


-- ==================== Audit Log Table ====================

CREATE TABLE IF NOT EXISTS auth_audit_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL, -- login, logout, register, password_change, etc.
    success BOOLEAN NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for auth_audit_log table
CREATE INDEX IF NOT EXISTS idx_auth_audit_log_user_id ON auth_audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_audit_log_action ON auth_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_auth_audit_log_created_at ON auth_audit_log(created_at);


-- ==================== Functions ====================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for users table
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    UPDATE user_sessions
    SET is_active = FALSE,
        revoked_at = NOW()
    WHERE expires_at < NOW()
      AND is_active = TRUE;
END;
$$ LANGUAGE plpgsql;


-- ==================== Initial Data ====================

-- ⚠️ SECURITY WARNING: Default credentials for development/testing only!
-- These accounts have weak passwords and MUST be changed or disabled in production.
-- Consider using environment variables or secure configuration management instead.

-- Create default admin user (password: 'admin123' - CHANGE IN PRODUCTION!)
-- Password hash for 'admin123' using bcrypt
-- TODO: Remove or disable this account before production deployment
INSERT INTO users (user_id, email, password_hash, full_name, roles, status, email_verified)
VALUES (
    'a0000000-0000-0000-0000-000000000001'::uuid,
    'admin@cognitionos.ai',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyDxD7KXMdLa', -- 'admin123'
    'System Administrator',
    ARRAY['user', 'admin'],
    'active',
    true
)
ON CONFLICT (email) DO NOTHING;


-- Create test user (password: 'testuser123' - FOR TESTING ONLY!)
-- TODO: Remove this account before production deployment
INSERT INTO users (user_id, email, password_hash, full_name, roles, status, email_verified)
VALUES (
    'b0000000-0000-0000-0000-000000000001'::uuid,
    'test@cognitionos.ai',
    '$2b$12$GZGKZKZqUFH5KqP7Qm9OKO7zGxJ5X4bPYNlPnL8JxJ9KqL7Qm9OK', -- 'testuser123'
    'Test User',
    ARRAY['user'],
    'active',
    true
)
ON CONFLICT (email) DO NOTHING;


-- ==================== Comments ====================

COMMENT ON TABLE users IS 'User accounts for authentication and authorization';
COMMENT ON TABLE user_sessions IS 'Active user sessions for token management and revocation';
COMMENT ON TABLE auth_audit_log IS 'Audit log for authentication events';

COMMENT ON COLUMN users.user_id IS 'Unique user identifier';
COMMENT ON COLUMN users.email IS 'User email address (unique)';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt hashed password';
COMMENT ON COLUMN users.roles IS 'User roles for authorization';
COMMENT ON COLUMN users.status IS 'Account status (active, inactive, suspended, pending_verification)';
COMMENT ON COLUMN users.failed_login_attempts IS 'Count of failed login attempts (resets on successful login)';
COMMENT ON COLUMN users.locked_until IS 'Account locked until this timestamp (null if not locked)';
