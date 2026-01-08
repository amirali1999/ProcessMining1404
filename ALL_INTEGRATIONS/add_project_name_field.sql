-- SQL Script to manually add project_name field to CSVImportSession
-- Run this if migration fails

ALTER TABLE discovery_csvimportsession 
ADD COLUMN project_name VARCHAR(255) NULL;

-- Verify the field was added
SELECT sql FROM sqlite_master 
WHERE type='table' AND name='discovery_csvimportsession';
