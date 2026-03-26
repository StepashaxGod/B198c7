from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer import RecognizerResult


#Simple container for one detected text span.
@dataclass
class Span:
    label: str
    start: int
    end: int
    text: str
    source: str


class PharmaDeidPipeline:
    """
    Main idea of the pipeline:
    1. Use Microsoft Presidio to detect standard PHI entities.
    2. Use custom logic to detect medication-related entities important
       for downstream pharmacological analytics.
    3. Temporarily mark all detected spans.
    4. In the final output, preserve medication-related entities
       and keep PHI anonymized.
    """

    def __init__(self) -> None:
        """
        Initialize the full pipeline.
        """

        # The small English model is used here because it is lightweight
        # and easy to run in a prototype.
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        )
        nlp_engine = provider.create_engine()

        # Create a recognizer registry and load Presidio built-in recognizers.
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)

        # Add one custom PHI recognizer for medical record numbers.
        # This is useful because MRN is very common in medical notes
        # and is relevant to the present de-identification task.
        registry.add_recognizer(
            PatternRecognizer(
                supported_entity="MEDICAL_RECORD_NUMBER",
                patterns=[
                    Pattern(
                        name="mrn_pattern",
                        regex=r"\b(?:MRN|Medical Record Number|Patient ID|Record Number)\s*[:#]?\s*[A-Z0-9\-]{4,}\b",
                        score=0.8,
                    )
                ],
            )
        )

        # Main Presidio analyzer object.
        self.analyzer = AnalyzerEngine(
            registry=registry,
            nlp_engine=nlp_engine,
            supported_languages=["en"],
        )

        # A small deny-list of drug names used in the prototype.
        # In a larger system, this list could be extended or replaced
        # by a clinical vocabulary or external medication model.
        self.drug_list = {
            "metformin", "insulin", "warfarin", "aspirin",
            "ibuprofen", "acetaminophen", "atorvastatin"
        }

        # A small list of adverse events relevant to the use case.
        self.adverse_event_list = {
            "nausea", "vomiting", "rash", "headache",
            "dizziness", "fatigue", "diarrhea"
        }

        # Regex pattern for dosage expressions.
        # It is designed to capture simple forms such as:
        # 500 mg, 10 ml, 500 mg BID, 20 mcg daily
        self.dosage_pattern = re.compile(
            r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?)\b(?:\s?(?:bid|tid|qid|daily|weekly|po|iv|im))?",
            re.IGNORECASE,
        )

    def preprocess(self, text: str) -> str:
        """
        Normalize whitespace in the input text.
        This simple preprocessing step makes matching more stable
        and keeps offsets more predictable for the later stages.
        """
        return re.sub(r"\s+", " ", text).strip()

    def detect_phi(self, text: str) -> List[Span]:
        """
        Detect standard PHI entities using Presidio.
        """
        results: List[RecognizerResult] = self.analyzer.analyze(
            text=text,
            entities=[
                "PERSON",
                "PHONE_NUMBER",
                "EMAIL_ADDRESS",
                "DATE_TIME",
                "LOCATION",
                "URL",
                "IP_ADDRESS",
                "US_SSN",
                "MEDICAL_RECORD_NUMBER",
            ],
            language="en",
        )

        return [
            Span(r.entity_type, r.start, r.end, text[r.start:r.end], "phi")
            for r in results
        ]

    def detect_clinical_entities(self, text: str) -> List[Span]:
        """
        Detect clinically relevant non-PHI entities.
        Returns:
            A list of Span objects labeled as source="clinical".
        """
        spans: List[Span] = []

        # Detect drug names from the predefined list.
        for drug in self.drug_list:
            for match in re.finditer(rf"\b{re.escape(drug)}\b", text, flags=re.IGNORECASE):
                spans.append(
                    Span("DRUG", match.start(), match.end(), match.group(0), "clinical")
                )

        # Detect dosage patterns such as "500 mg BID".
        for match in self.dosage_pattern.finditer(text):
            spans.append(
                Span("DOSAGE", match.start(), match.end(), match.group(0), "clinical")
            )

        # Detect adverse event terms from the predefined list.
        for event in self.adverse_event_list:
            for match in re.finditer(rf"\b{re.escape(event)}\b", text, flags=re.IGNORECASE):
                spans.append(
                    Span("ADVERSE_EVENT", match.start(), match.end(), match.group(0), "clinical")
                )

        return spans

    def deduplicate(self, spans: List[Span]) -> List[Span]:
        
        spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
        result: List[Span] = []
        last_end = -1

        for span in spans:
            if span.start >= last_end:
                result.append(span)
                last_end = span.end

        return result

    def replace_spans(self, text: str, spans: List[Span], preserve_clinical: bool) -> str:
        """
        Replace spans in the text.
        Returns:
            Transformed text.
        """
        output = []
        last = 0

        for span in spans:
            output.append(text[last:span.start])

            # If preservation is enabled, medication-related entities are restored.
            # Otherwise, everything is tagged.
            if preserve_clinical and span.source == "clinical":
                output.append(span.text)
            else:
                output.append(f"<{span.label}>")

            last = span.end

        output.append(text[last:])
        return "".join(output)

    def run(self, text: str) -> Dict:
        """
        Run the full pipeline on one clinical note.

        Steps:
        1. preprocess input text
        2. detect PHI with Presidio
        3. detect drug-related entities with custom recognizers
        4. merge and deduplicate all spans
        5. generate:
           - fully marked intermediate text
           - final de-identified text

        Returns:
            A dictionary with intermediate and final outputs,
            plus metadata about all detected spans.
        """
        text = self.preprocess(text)

        # Merge PHI spans and custom clinical spans into one list.
        spans = self.deduplicate(
            self.detect_phi(text) + self.detect_clinical_entities(text)
        )

        # Intermediate representation: everything is tagged.
        fully_marked = self.replace_spans(text, spans, preserve_clinical=False)

        # Final output: PHI stays masked, clinical entities are restored.
        final_text = self.replace_spans(text, spans, preserve_clinical=True)

        return {
            "fully_marked_text": fully_marked,
            "final_deidentified_text": final_text,
            "all_candidates": [span.__dict__ for span in spans],
        }


if __name__ == "__main__":
    """
    Example usage of the prototype.
    A short medical note is passed into the pipeline,
    and both the intermediate and final outputs are printed.
    """

    sample_note = """
    Patient Name: Stepan Nikonov
    DOB: 03/14/1978
    Phone number is (415) 555-0198
    Email: stepan.nikonov@email.com
    MRN: A1234567

    On 01/10/2024 the patient was started on metformin 500 mg BID.
    The patient developed nausea and rash after treatment escalation.
    """

    pipeline = PharmaDeidPipeline()
    result = pipeline.run(sample_note)

    print("=== FULLY MARKED TEXT ===")
    print(result["fully_marked_text"])

    print("\n=== FINAL DE-IDENTIFIED TEXT ===")
    print(result["final_deidentified_text"])

    print("\n=== ALL CANDIDATES ===")
    for item in result["all_candidates"]:
        print(item)
