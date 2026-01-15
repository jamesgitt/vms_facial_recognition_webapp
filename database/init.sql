-- Initialize database schema for face recognition visitor storage
-- This script runs automatically when the database container is first created
-- Schema matches the Visitor entity diagram

-- Create visitors table
CREATE TABLE IF NOT EXISTS visitors (
    id VARCHAR(255) PRIMARY KEY,
    "firstName" VARCHAR(255),
    "lastName" VARCHAR(255),
    "fullName" VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(50),
    "imageUrl" VARCHAR(500),
    "base64Image" TEXT,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on createdAt for faster queries
CREATE INDEX IF NOT EXISTS idx_visitors_created_at ON visitors("createdAt");
CREATE INDEX IF NOT EXISTS idx_visitors_email ON visitors(email);
CREATE INDEX IF NOT EXISTS idx_visitors_fullname ON visitors("fullName");

-- Create function to update updatedAt timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW."updatedAt" = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updatedAt
CREATE TRIGGER update_visitors_updated_at 
    BEFORE UPDATE ON visitors 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data (optional - for testing)
-- Uncomment and modify as needed
/*
INSERT INTO visitors (id, "firstName", "lastName", "fullName", email, phone, "base64Image") VALUES
('visitor_001', 'John', 'Doe', 'John Doe', 'john@example.com', '+1234567890', 'base64_encoded_image_here'),
('visitor_002', 'Jane', 'Smith', 'Jane Smith', 'jane@example.com', '+1234567891', 'base64_encoded_image_here');
*/

-- Grant permissions (if using different user)
-- GRANT ALL PRIVILEGES ON TABLE visitors TO your_user;

COMMENT ON TABLE visitors IS 'Stores visitor information and base64 encoded face images for recognition';
COMMENT ON COLUMN visitors.id IS 'Unique identifier for each visitor';
COMMENT ON COLUMN visitors."base64Image" IS 'Base64 encoded image string for face recognition';
COMMENT ON COLUMN visitors."imageUrl" IS 'URL reference to visitor image (optional)';
COMMENT ON COLUMN visitors."fullName" IS 'Full name of the visitor (firstName + lastName)';
