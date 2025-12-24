"""
SQLAlchemy models for the NLP Histopathology database.
Defines Document and TextElement models with relationships.
"""

from sqlalchemy import (
    Column, Integer, String, Text, TIMESTAMP, ForeignKey, ARRAY, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Document(Base):
    """
    Represents an XML document (paper) in the database.
    Each document has multiple text elements.
    """
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    pmcid = Column(String(50), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    title = Column(Text)
    journal = Column(String(255))
    publication_year = Column(Integer)
    text_source = Column(String(20), default='xml')  # 'xml' or 'pdf'
    processed_at = Column(TIMESTAMP, default=func.now())
    created_at = Column(TIMESTAMP, default=func.now())
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())

    # Relationship to text elements
    text_elements = relationship(
        "TextElement",
        back_populates="document",
        cascade="all, delete-orphan"
    )

    # Relationship to figures
    figures = relationship(
        "Figure",
        back_populates="document",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Document(pmcid='{self.pmcid}', title='{self.title[:50] if self.title else None}...')>"


class TextElement(Base):
    """
    Represents a single text element (paragraph) with its hierarchical path.
    Each text element belongs to one document.
    """
    __tablename__ = 'text_elements'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True)

    # Unique path for this text element
    unique_path = Column(Text, unique=True, nullable=False, index=True)

    # Hierarchical information
    path_list = Column(ARRAY(Text), nullable=False)  # PostgreSQL array
    path_string = Column(Text, nullable=False, index=True)
    depth = Column(Integer, nullable=False, index=True)
    position_in_section = Column(Integer, nullable=False)

    # The actual text content
    text_content = Column(Text, nullable=False)

    # Metadata
    word_count = Column(Integer)
    char_count = Column(Integer)
    created_at = Column(TIMESTAMP, default=func.now())

    # Relationship to document
    document = relationship("Document", back_populates="text_elements")

    # Add a composite unique constraint
    __table_args__ = (
        Index('idx_document_path_position', 'document_id', 'path_string', 'position_in_section', unique=True),
    )

    def __repr__(self):
        text_preview = self.text_content[:50] if self.text_content else ""
        return f"<TextElement(unique_path='{self.unique_path}', text='{text_preview}...')>"

    @property
    def section_hierarchy(self):
        """Returns the full hierarchy as a formatted string."""
        return " > ".join(self.path_list) if self.path_list else "Root"

    def calculate_metadata(self):
        """Calculate and update word_count and char_count."""
        if self.text_content:
            self.char_count = len(self.text_content)
            self.word_count = len(self.text_content.split())


class Figure(Base):
    """
    Represents a figure (image) with caption from a paper.
    Each figure belongs to one document.
    """
    __tablename__ = 'figures'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True)

    # Figure identification
    figure_id = Column(String(100))  # e.g., "fig1", "fig2a"
    figure_label = Column(String(50))  # e.g., "Figure 1", "Fig 2A"

    # Caption and context
    caption_text = Column(Text)

    # Image file information
    graphic_ref = Column(String(255))  # Original filename from XML
    image_filename = Column(String(255))  # Stored filename
    image_path = Column(Text)  # Full path to stored image
    image_format = Column(String(20))  # jpg, png, tif, etc.

    # Location in document
    section_context = Column(Text)  # Which section this figure appears in

    # Metadata
    created_at = Column(TIMESTAMP, default=func.now())

    # Relationship to document
    document = relationship("Document", back_populates="figures")

    # Add a composite unique constraint
    __table_args__ = (
        Index('idx_document_figure_id', 'document_id', 'figure_id', unique=True),
    )

    def __repr__(self):
        return f"<Figure(figure_id='{self.figure_id}', label='{self.figure_label}')>"
