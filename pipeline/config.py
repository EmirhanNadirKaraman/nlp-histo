from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class TableDetectorType(str, Enum):
    TATR = "tatr"
    DOCLING = "docling"
    HYBRID = "hybrid"
    VLM = "vlm"


class BaselineMode(str, Enum):
    MASKED = "masked"
    UNMASKED = "unmasked"
    BOTH = "both"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(slots=True)
class PathConfig:
    project_root: Path = Path(".")
    output_root: Path = Path("out")
    files_root: Path = Path("files")

    # Core outputs
    masked_pdf_dir: Path = Path("out/masked_pdfs")
    docling_full_dir: Path = Path("out/docling_full")
    docling_masked_dir: Path = Path("out/docling_masked")
    text_dir: Path = Path("out/text")
    text_raw_dir: Path = Path("out/text_raw")
    json_dir: Path = Path("out/json")
    vis_dir: Path = Path("out/visualization")

    # Crops / media
    figures_dir: Path = Path("out/figures")
    tables_dir: Path = Path("out/tables")

    # Metadata / bookkeeping
    blacklist_file: Path = Path("out/failed_pdfs_blacklist.json")
    run_metadata_dir: Path = Path("out/run_metadata")

    def ensure_dirs(self) -> None:
        dirs = [
            self.output_root,
            self.files_root,
            self.masked_pdf_dir,
            self.docling_full_dir,
            self.docling_masked_dir,
            self.text_dir,
            self.text_raw_dir,
            self.json_dir,
            self.vis_dir,
            self.figures_dir,
            self.tables_dir,
            self.run_metadata_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class DoclingConfig:
    enabled: bool = True
    do_table_structure: bool = True
    do_ocr: bool = False
    reconstruct_tables_from_lists: bool = False
    export_intermediate_json: bool = True
    timeout_sec: int = 300


@dataclass(slots=True)
class TATRConfig:
    enabled: bool = True
    threshold: float = 0.99
    max_detections_per_page: int = 200
    device: str = "cpu"  # "cpu", "cuda", "mps"
    model_name: str = "microsoft/table-transformer-detection"
    structure_model_name: Optional[str] = None  # optional if you later want structure extraction
    batch_size_pages: int = 1


@dataclass(slots=True)
class MaskingConfig:
    enabled: bool = True
    mask_tables: bool = True
    mask_figures: bool = True
    mask_header_footer_sidebar: bool = True
    merge_overlapping_boxes: bool = True
    merge_iou_threshold: float = 0.3
    expand_box_px: int = 2  # small padding to avoid glyph remnants


@dataclass(slots=True)
class FilteringConfig:
    enabled: bool = True
    apply_ner_filtering: bool = True
    apply_paragraph_relevance_filtering: bool = True
    fix_ligatures: bool = True
    remove_reference_markers: bool = False
    min_paragraph_chars: int = 20


@dataclass(slots=True)
class CroppingConfig:
    enabled: bool = True
    save_figure_crops: bool = True
    save_table_crops: bool = True
    image_format: str = "png"
    dpi: int = 200
    include_captions_in_metadata: bool = True
    panel_counting_enabled: bool = False
    min_figure_pts: int = 50        # minimum width AND height in PDF points; smaller figures are skipped
    merge_figures_by_caption: bool = False  # merge PICTURE elements sharing the same caption number
    merge_tables_by_caption: bool = False   # merge TABLE/detection regions sharing the same caption number
    subfigure_proximity_pts: int = 20       # max edge-to-edge gap to treat adjacent figures as subfigure panels


@dataclass(slots=True)
class TextAssemblyConfig:
    enabled: bool = True
    baseline_mode: BaselineMode = BaselineMode.MASKED
    use_hierarchical_extraction: bool = True
    use_context_aware_stitching: bool = True
    compare_combinations: bool = False
    save_combination_outputs: bool = False
    write_raw_text: bool = False  # dump pre-assembly elements to out/text_raw/


@dataclass(slots=True)
class VisualizationConfig:
    enabled: bool = True
    save_tatr_visualization: bool = True
    save_combined_visualization: bool = True
    max_pages: Optional[int] = None


@dataclass(slots=True)
class DatabaseConfig:
    enabled: bool = False
    db_url: Optional[str] = None
    schema: str = "public"
    create_tables_if_missing: bool = False
    batch_size: int = 100
    connect_timeout_sec: int = 15


@dataclass(slots=True)
class RuntimeConfig:
    log_level: LogLevel = LogLevel.INFO
    fail_fast: bool = False
    skip_blacklisted: bool = True
    skip_existing_in_db: bool = True   # skip documents already in the database
    update_blacklist_on_failure: bool = True
    save_error_traces: bool = True
    seed: int = 42
    num_workers: int = 1


@dataclass(slots=True)
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    docling: DoclingConfig = field(default_factory=DoclingConfig)
    tatr: TATRConfig = field(default_factory=TATRConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    cropping: CroppingConfig = field(default_factory=CroppingConfig)
    text: TextAssemblyConfig = field(default_factory=TextAssemblyConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    table_detector: TableDetectorType = TableDetectorType.HYBRID

    def validate(self) -> None:
        if self.tatr.threshold < 0.0 or self.tatr.threshold > 1.0:
            raise ValueError(f"tatr.threshold must be in [0, 1], got {self.tatr.threshold}")

        if self.cropping.dpi <= 0:
            raise ValueError(f"cropping.dpi must be > 0, got {self.cropping.dpi}")

        if self.runtime.num_workers < 1:
            raise ValueError(f"runtime.num_workers must be >= 1, got {self.runtime.num_workers}")

        if self.database.enabled and not self.database.db_url:
            try:
                from database.db_connection import get_database_url  # type: ignore
                self.database.db_url = get_database_url()
            except Exception:
                raise ValueError(
                    "database.enabled=True but database.db_url is not set "
                    "and no .env / environment variables found"
                )

    def prepare(self) -> None:
        self.validate()
        self.paths.ensure_dirs()