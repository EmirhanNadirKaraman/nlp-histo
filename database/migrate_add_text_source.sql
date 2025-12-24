-- Migration: Add text_source column to documents table
-- This preserves existing data

-- Add the text_source column
ALTER TABLE documents ADD COLUMN IF NOT EXISTS text_source VARCHAR(20) DEFAULT 'xml';

-- Update existing records to 'xml'
UPDATE documents SET text_source = 'xml' WHERE text_source IS NULL;

-- Verify the change
SELECT COUNT(*) as total_documents, text_source FROM documents GROUP BY text_source;

COMMIT;
