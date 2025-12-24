-- Database schema for NLP Histopathology project
-- This schema stores hierarchical text structure from XML files

-- Drop existing tables if they exist
DROP TABLE IF EXISTS figures CASCADE;
DROP TABLE IF EXISTS text_elements CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- Documents table: stores metadata about each XML file
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    pmcid VARCHAR(50) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    title TEXT,
    journal VARCHAR(255),
    publication_year INTEGER,
    text_source VARCHAR(20) DEFAULT 'xml',  -- 'xml' or 'pdf'
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Text elements table: stores each paragraph/text with its hierarchical path
CREATE TABLE text_elements (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Unique path for this text element
    unique_path TEXT UNIQUE NOT NULL,

    -- Hierarchical information
    path_list TEXT[] NOT NULL,  -- Array of section names: ['Methods', '2.1 Staining']
    path_string TEXT NOT NULL,  -- String format: "Methods > 2.1 Staining"
    depth INTEGER NOT NULL,     -- Nesting level (1, 2, 3, ...)
    position_in_section INTEGER NOT NULL,  -- Position within the current section

    -- The actual text content
    text_content TEXT NOT NULL,

    -- Metadata
    word_count INTEGER,
    char_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure uniqueness per document
    UNIQUE(document_id, path_string, position_in_section)
);

-- Figures table: stores figure metadata and image information
CREATE TABLE figures (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Figure identification
    figure_id VARCHAR(100),  -- e.g., "fig1", "fig2a"
    figure_label VARCHAR(50),  -- e.g., "Figure 1", "Fig 2A"

    -- Caption and context
    caption_text TEXT,

    -- Image file information
    graphic_ref VARCHAR(255),  -- Original filename from XML (e.g., "image1.jpg")
    image_filename VARCHAR(255),  -- Stored filename
    image_path TEXT,  -- Full path to stored image file
    image_format VARCHAR(20),  -- jpg, png, tif, etc.

    -- Location in document
    section_context TEXT,  -- Which section this figure appears in

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure uniqueness per document
    UNIQUE(document_id, figure_id)
);

-- Indexes for faster queries
CREATE INDEX idx_documents_pmcid ON documents(pmcid);
CREATE INDEX idx_text_elements_document_id ON text_elements(document_id);
CREATE INDEX idx_text_elements_depth ON text_elements(depth);
CREATE INDEX idx_text_elements_path_string ON text_elements(path_string);
CREATE INDEX idx_text_elements_unique_path ON text_elements(unique_path);
CREATE INDEX idx_figures_document_id ON figures(document_id);
CREATE INDEX idx_figures_figure_id ON figures(figure_id);

-- GIN index for array operations on path_list
CREATE INDEX idx_text_elements_path_list ON text_elements USING GIN(path_list);

-- Full-text search index on text content
CREATE INDEX idx_text_elements_text_content_fts ON text_elements USING GIN(to_tsvector('english', text_content));

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE documents IS 'Stores metadata about each XML document (paper)';
COMMENT ON TABLE text_elements IS 'Stores individual text elements (paragraphs) with their hierarchical paths';
COMMENT ON TABLE figures IS 'Stores figure metadata, captions, and image file information';
COMMENT ON COLUMN documents.text_source IS 'Source of text extraction: xml (from XML file) or pdf (from PDF fallback)';
COMMENT ON COLUMN text_elements.unique_path IS 'Unique identifier: {PMCID}/{path_string}/{position}';
COMMENT ON COLUMN text_elements.path_list IS 'Array of section names for hierarchical queries';
COMMENT ON COLUMN text_elements.path_string IS 'Human-readable path representation';
COMMENT ON COLUMN text_elements.position_in_section IS 'Sequential position within the same path';
COMMENT ON COLUMN figures.graphic_ref IS 'Original image filename from XML graphic element';
COMMENT ON COLUMN figures.image_path IS 'Full path to stored image file on disk';
