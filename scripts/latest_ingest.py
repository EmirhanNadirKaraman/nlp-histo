def main():
    # Setup logging based on debug flag
    setup_logging(args.debug)

    # Create processor
    processor = ComprehensiveDocumentProcessor(output_dir=args.output_dir)

    # Batch mode takes priority
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        for pdf_file in pdf_files:
            # Extract PMCID from filename
            stem = pdf_file.stem
            if stem.startswith("PMC"):
                pmcid = stem.split('_')[0]
            else:
                logger.warning(f"Skipping {pdf_file.name} - cannot extract PMCID")
                continue

            # Use reconstruction mode if requested
            if args.use_reconstruction:
                processor.process_document_with_reconstruction(
                    pdf_path=pdf_file,
                    pmcid=pmcid,
                    force=args.force,
                    text_only=args.text_only,
                    stitch_paragraphs=args.stitch_paragraphs
                )
            else:
                processor.process_document(
                    pdf_path=pdf_file,
                    pmcid=pmcid,
                    force=args.force,
                    text_only=args.text_only,
                    stitch_paragraphs=args.stitch_paragraphs
                )

        processor.print_stats()

    # Single file mode (default or explicitly specified)
    else:
        if not args.pmcid:
            # Try to extract PMCID from filename
            stem = Path(args.pdf).stem
            if stem.startswith("PMC"):
                args.pmcid = stem.split('_')[0]
            else:
                logger.error("--pmcid is required when filename doesn't start with PMC")
                sys.exit(1)

        # Use reconstruction mode if requested
        if args.use_reconstruction:
            processor.process_document_with_reconstruction(
                pdf_path=Path(args.pdf),
                pmcid=args.pmcid,
                title=args.title,
                journal=args.journal,
                publication_year=args.year,
                force=args.force,
                text_only=args.text_only,
                stitch_paragraphs=args.stitch_paragraphs
            )
        else:
            processor.process_document(
                pdf_path=Path(args.pdf),
                pmcid=args.pmcid,
                title=args.title,
                journal=args.journal,
                publication_year=args.year,
                force=args.force,
                text_only=args.text_only,
                stitch_paragraphs=args.stitch_paragraphs
            )


if __name__ == "__main__":
    main()
